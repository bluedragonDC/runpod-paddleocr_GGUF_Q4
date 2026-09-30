"""Redis-backed per-tenant rate limiter and fair FIFO concurrency gate."""

from __future__ import annotations

import hashlib
import time

from redis import Redis


RATE_LIMIT_LUA = """
local n = redis.call('INCR', KEYS[1])
if n == 1 then redis.call('PEXPIRE', KEYS[1], ARGV[1]) end
if n > tonumber(ARGV[2]) then return {0, redis.call('PTTL', KEYS[1])} end
return {1, redis.call('PTTL', KEYS[1])}
"""

ENQUEUE_LUA = """
local existing = redis.call('ZSCORE', KEYS[1], ARGV[1])
if existing then return {1, redis.call('ZCARD', KEYS[1]) - 1} end
local queued = redis.call('ZCARD', KEYS[1])
if queued >= tonumber(ARGV[2]) then return {0, queued} end
local seq = redis.call('INCR', KEYS[3])
redis.call('ZADD', KEYS[1], seq, ARGV[1])
redis.call('EXPIRE', KEYS[1], 86400)
redis.call('EXPIRE', KEYS[3], 86400)
return {1, queued}
"""

ACQUIRE_LUA = """
local now = tonumber(ARGV[1])
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now)
local first = redis.call('ZRANGE', KEYS[1], 0, 0)
local active = redis.call('ZCARD', KEYS[2])
if first[1] == ARGV[2] and active < tonumber(ARGV[3]) then
  redis.call('ZREM', KEYS[1], ARGV[2])
  redis.call('ZADD', KEYS[2], now + tonumber(ARGV[4]), ARGV[2])
  redis.call('EXPIRE', KEYS[2], 3600)
  return 1
end
return 0
"""

RENEW_LUA = """
if redis.call('ZSCORE', KEYS[1], ARGV[1]) then
  redis.call('ZADD', KEYS[1], tonumber(ARGV[2]) + tonumber(ARGV[3]), ARGV[1])
  return 1
end
return 0
"""

REMOVE_LUA = """
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('ZREM', KEYS[2], ARGV[1])
return 1
"""


class RateLimited(Exception):
    def __init__(self, retry_after_ms: int):
        self.retry_after_ms = max(0, retry_after_ms)


class TenantQueueFull(Exception):
    pass


class RedisTenantQueue:
    def __init__(self, client: Redis, tenant_id: str):
        self.redis = client
        self.prefix = "ocr:tenant:" + hashlib.sha256(tenant_id.encode()).hexdigest()
        self.waiting = f"{self.prefix}:waiting"
        self.active = f"{self.prefix}:active"
        self.sequence = f"{self.prefix}:seq"

    def check_rate(self, limit: int) -> None:
        bucket = int(time.time() // 60)
        ok, ttl = self.redis.eval(RATE_LIMIT_LUA, 1, f"{self.prefix}:rate:{bucket}", 61_000, limit)
        if not ok:
            raise RateLimited(int(ttl))

    def enqueue_and_acquire(
        self,
        job_id: str,
        *,
        max_active: int,
        max_queued: int,
        wait_timeout_s: int,
        lease_ms: int,
    ) -> int:
        admitted, queued_ahead = self.redis.eval(
            ENQUEUE_LUA, 3, self.waiting, self.active, self.sequence, job_id, max_queued
        )
        if not admitted:
            raise TenantQueueFull("User queue is full; retry with a new job later")
        deadline = time.monotonic() + wait_timeout_s
        while time.monotonic() < deadline:
            acquired = self.redis.eval(
                ACQUIRE_LUA,
                2,
                self.waiting,
                self.active,
                int(time.time() * 1000),
                job_id,
                max_active,
                lease_ms,
            )
            if acquired:
                return int(queued_ahead)
            time.sleep(0.25)
        self.remove(job_id)
        raise TenantQueueFull("Waited too long for this user's active-job slot; retry later")

    def renew(self, job_id: str, lease_ms: int) -> bool:
        return bool(self.redis.eval(RENEW_LUA, 1, self.active, job_id, int(time.time() * 1000), lease_ms))

    def remove(self, job_id: str) -> None:
        self.redis.eval(REMOVE_LUA, 2, self.waiting, self.active, job_id)
