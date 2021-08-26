"""This script is used to start a webserver at Host: localhost, Port: 0 (defaults to a non-used port).
The files in head/docs/build/html hosted (contains the documentation of the project).
The default webbrowser is opened to view the html files.
To shutdown the server, this script accepts user input in the console where this script has been executed.

Attributes
----------
docsBuildDirPath: pathlib.Path
    path to the html files of the documentation
"""
import webbrowser
import socketserver
import threading
from pathlib import Path
import http.server
from typing import Any

headDirPath: Path = Path(__file__).parents[2]
docsBuildDirPath: Path = (headDirPath / "docs" / "build" / "html").resolve()

HOST: str = "localhost"
PORT: int = 0

class Handler(http.server.SimpleHTTPRequestHandler):
    """Request handler for the webserver. This is only used to suppress log messages.

    In the __init__ method the directory is set to the docsBuildDirPath.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=docsBuildDirPath, **kwargs) # type: ignore

    def log_message(self, format: str, *args: Any) -> None:
        """Overwrite log_message to suppress log messages in the console.
        """
        pass

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """A threaded TCP server is used to run the server thread as a daemon in the background.
    Consequently, the server can be shutdown by user input in the console.
    """
    pass

def shutdown_server():
    """A simple function that returns 0 or 1 based on user input. This used to shutdown the webserver.

    Returns
    -------
    int
        0 if user input is q or exit, 1 otherwise
    """
    userInput: str = input("Shutdown server: 'q' or 'exit: ").lower().replace(" ", "")
    if userInput in ["q", "exit"]:
        return 0
    else:
        return 1

def main():
    """Starts the default webbrowser, displaying the program documentation.
    """
    with ThreadedTCPServer((HOST, PORT), Handler) as server:
        port: int
        _, port = server.server_address
        print(f"Serving at port: {port}")
        server_thread: threading.Thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        print(f"Opening documentation in the web browser.")
        webbrowser.open(f"http://localhost:{port}")
        shutdownServerFlag = shutdown_server()
        while shutdownServerFlag == 1:
            shutdownServerFlag = shutdown_server()
        server.shutdown()

if __name__ == "__main__":
    main()
