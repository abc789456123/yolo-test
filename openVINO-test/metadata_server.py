#!/usr/bin/env python3
"""
간단한 HTTP 서버 - JSON 메타데이터를 받아서 출력하는 서버
"""

import json
import http.server
import socketserver
from datetime import datetime

class MetadataHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/metadata':
            # Content-Length 헤더 읽기
            content_length = int(self.headers['Content-Length'])
            
            # POST 데이터 읽기
            post_data = self.rfile.read(content_length)
            
            try:
                # JSON 파싱
                metadata = json.loads(post_data.decode('utf-8'))
                
                # 현재 시간과 함께 메타데이터 출력
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n[{now}] Received metadata:")
                print(json.dumps(metadata, indent=2, ensure_ascii=False))
                
                # 성공 응답
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {"status": "success", "message": "Metadata received"}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except json.JSONDecodeError as e:
                # JSON 파싱 오류
                print(f"JSON parsing error: {e}")
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {"status": "error", "message": "Invalid JSON"}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
        else:
            # 404 Not Found
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        if self.path == '/':
            # 간단한 상태 페이지
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <html>
            <head><title>Metadata Server</title></head>
            <body>
                <h1>YOLOv5 Metadata Server</h1>
                <p>Server is running and ready to receive metadata.</p>
                <p>POST to /metadata to send detection data.</p>
                <p>Server time: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # 기본 로그 메시지를 억제 (사용자 정의 로깅 사용)
        pass

if __name__ == "__main__":
    PORT = 8080
    
    print(f"Starting metadata server on port {PORT}")
    print(f"Access at: http://localhost:{PORT}")
    print("Send POST requests to: http://localhost:{PORT}/metadata")
    print("Press Ctrl+C to stop the server\n")
    
    with socketserver.TCPServer(("", PORT), MetadataHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
