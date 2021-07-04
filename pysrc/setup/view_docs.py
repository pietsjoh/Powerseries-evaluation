import webbrowser
import socketserver
import threading
from pathlib import Path
import http.server

headDirPath = Path(__file__).parents[2]
docsBuildDirPath = str((headDirPath / "docs" / "build" / "html").resolve())

HOST, PORT = "localhost", 0

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(directory=docsBuildDirPath, *args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        pass

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

def shutdown_server():
    userInput = input("Shutdown server: 'q' or 'exit: ")
    if userInput in ["q", "exit"]:
        return 0
    else:
        return 1

if __name__ == "__main__":
    with ThreadedTCPServer((HOST, PORT), Handler) as server:
        ip, port = server.server_address
        print(f"Serving at port: {port}")
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        print(f"Opening documentation in the web browser.")
        webbrowser.open(f"http://localhost:{port}")
        shutdownServerFlag = shutdown_server()
        while shutdownServerFlag == 1:
            shutdownServerFlag = shutdown_server()
        server.shutdown()
