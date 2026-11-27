"""A real local HTTP server, so download tests exercise actual retries and status codes."""

import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

# A response is (status, body), or DROP to close the connection without answering.
DROP = "drop"


class LocalHttpServer:
    """Serve scripted responses on 127.0.0.1 and record every requested path.

    Each path has a list of responses; one is used per request and the last one
    repeats. Unknown paths answer 404. ``respond`` can be replaced by a callable
    ``path -> response`` to serve a whole fake site.
    """

    def __init__(self):
        self.routes: dict[str, list] = {}
        self.requested: list[str] = []
        self.respond: Callable[[str], object] = self._scripted
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.requested.append(self.path)
                response = server.respond(self.path)
                if response == DROP:
                    self.close_connection = True
                    return
                status, body = response
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                """Keep test output quiet."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def url(self, path: str) -> str:
        """Return the absolute URL of a path such as ``/image.jpg``."""
        return self.base_url + path

    def serve(self, path: str, *responses) -> None:
        """Answer ``path`` with ``responses`` in order, repeating the last one."""
        self.routes[path] = list(responses)

    def reset(self) -> None:
        """Forget routes and requests, and serve scripted routes again."""
        self.routes.clear()
        self.requested.clear()
        self.respond = self._scripted

    def _scripted(self, path: str):
        """Pop the next scripted response, keeping the last one for repeats."""
        responses = self.routes.get(path)
        if not responses:
            return 404, b""
        return responses.pop(0) if len(responses) > 1 else responses[0]

    def close(self) -> None:
        """Stop serving and release the port."""
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture(scope="module")
def module_http_server():
    """One server per test module; Hypothesis tests may use it across examples."""
    server = LocalHttpServer()
    yield server
    server.close()


@pytest.fixture
def http_server(module_http_server):
    """The module's server with fresh routes and request log for each test."""
    module_http_server.reset()
    return module_http_server
