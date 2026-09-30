import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import runpod_handler

class TestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            req_json = json.loads(post_data.decode('utf-8'))
            # Wrap the payload as a RunPod job
            job = {
                "id": "test-local-job",
                "input": req_json.get("input", {})
            }

            print(f"Received request. Processing with runpod_handler...")

            # Call the exact same function RunPod calls
            result = runpod_handler.process_job(job)

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # RunPod wraps the handler output in "output"
            response_wrapper = {
                "id": job["id"],
                "status": "COMPLETED" if "error" not in result else "FAILED",
                "output": result
            }
            self.wfile.write(json.dumps(response_wrapper).encode('utf-8'))
            print("Response sent successfully.")

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            print(f"Error: {e}")

if __name__ == '__main__':
    server_address = ('', 8080)
    httpd = ThreadingHTTPServer(server_address, TestHandler)
    print("Starting local test server on http://localhost:8080")
    print("This perfectly simulates the RunPod Serverless environment.")
    print("Use curl or your flutter app to POST to http://localhost:8080")
    httpd.serve_forever()
