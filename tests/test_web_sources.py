"""Stage 10B safe deterministic web-source tests without real networking."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import zlib
from email.message import Message
from io import BytesIO
from typing import Any

import pytest

import codeloop.execution.web_sources as web_sources
from codeloop.execution.tools import ToolRegistry
from codeloop.execution.web_sources import (
    MAX_WEB_BODY_BYTES,
    MAX_WEB_CONNECT_ATTEMPTS,
    ExtractedWebPage,
    HopDeadline,
    ResolvedWebTarget,
    StdlibWebTransport,
    SystemWebResolver,
    WebHttpResponse,
    WebPageAdapter,
    WebSourceError,
    _FixedIPConnector,
    _REQUEST_HEADERS,
    _decode_content_encoding,
    _new_connection,
    _read_encoded_body,
    _validate_response_framing,
    normalize_web_url,
)
from codeloop.execution.workspace import Workspace


class FakeResolver:
    def __init__(self, mapping: dict[str, tuple[str, ...]]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, int]] = []

    def resolve(
        self,
        hostname: str,
        port: int,
        deadline: HopDeadline,
    ) -> tuple[str, ...]:
        deadline.remaining_time()
        self.calls.append((hostname, port))
        if hostname not in self.mapping:
            raise WebSourceError("dns_resolution_failed", "not found")
        return self.mapping[hostname]


class FakeTransport:
    def __init__(self, responses: list[WebHttpResponse]) -> None:
        self.responses = iter(responses)
        self.targets: list[ResolvedWebTarget] = []

    def get(
        self,
        target: ResolvedWebTarget,
        deadline: HopDeadline,
    ) -> WebHttpResponse:
        deadline.remaining_time()
        self.targets.append(target)
        return next(self.responses)


class StaticAdapter:
    def __init__(self, page: ExtractedWebPage) -> None:
        self.page = page
        self.urls: list[str] = []

    def extract(self, url: str) -> ExtractedWebPage:
        self.urls.append(url)
        return self.page


class ErrorAdapter:
    def __init__(
        self,
        error_code: str,
        data: dict[str, object] | None = None,
    ) -> None:
        self.error_code = error_code
        self.data = data

    def extract(self, _url: str) -> ExtractedWebPage:
        raise WebSourceError(
            self.error_code,
            "specific web source failure",
            data=self.data,
        )


def _response(
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "text/html; charset=utf-8",
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> WebHttpResponse:
    return WebHttpResponse(
        status=status,
        headers=(("Content-Type", content_type), *extra_headers),
        encoded_body=body,
    )


def _adapter(
    responses: list[WebHttpResponse],
    mapping: dict[str, tuple[str, ...]] | None = None,
) -> tuple[WebPageAdapter, FakeTransport]:
    transport = FakeTransport(responses)
    resolver = FakeResolver(
        mapping
        or {
            "example.com": ("8.8.8.8",),
            "static.example.com": ("1.1.1.1",),
        }
    )
    return WebPageAdapter(resolver=resolver, transport=transport), transport


def test_http_https_normalization_and_stable_public_ip_selection() -> None:
    responses = [
        _response(b"plain", content_type="text/plain"),
        _response(b"secure", content_type="text/plain"),
    ]
    adapter, transport = _adapter(
        responses,
        {"example.com": ("8.8.8.8", "1.1.1.1", "8.8.8.8")},
    )

    first = adapter.extract("HTTP://Example.COM")
    second = adapter.extract("https://example.com/spec#section")

    assert first.requested_url == "http://example.com/"
    assert second.requested_url == "https://example.com/spec"
    assert [target.validated_ips for target in transport.targets] == [
        ("1.1.1.1", "8.8.8.8"),
        ("1.1.1.1", "8.8.8.8"),
    ]
    assert [target.port for target in transport.targets] == [80, 443]

    reversed_adapter, reversed_transport = _adapter(
        [_response(b"plain", content_type="text/plain")],
        {"example.com": ("1.1.1.1", "8.8.8.8")},
    )
    reversed_adapter.extract("http://example.com")
    assert reversed_transport.targets[0].validated_ips == (
        "1.1.1.1",
        "8.8.8.8",
    )


@pytest.mark.parametrize("url", ["file:///tmp/a", "ftp://example.com/a"])
def test_unsupported_url_scheme(url: str) -> None:
    with pytest.raises(WebSourceError) as caught:
        normalize_web_url(url)
    assert caught.value.error_code == "unsupported_url_scheme"


@pytest.mark.parametrize(
    "url",
    [
        "example.com/spec",
        "https://",
        "https://user@example.com/",
        "https://bad host.example/",
    ],
)
def test_invalid_url(url: str) -> None:
    with pytest.raises(WebSourceError) as caught:
        normalize_web_url(url)
    assert caught.value.error_code == "invalid_url"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://service.localhost/",
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://169.254.1.1/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
    ],
)
def test_local_and_non_public_targets_are_rejected(url: str) -> None:
    adapter, _transport = _adapter([])
    with pytest.raises(WebSourceError) as caught:
        adapter.extract(url)
    assert caught.value.error_code == "unsafe_url"


def test_hostname_resolving_to_any_private_address_is_rejected() -> None:
    adapter, transport = _adapter(
        [],
        {"example.com": ("8.8.8.8", "192.168.1.2")},
    )
    with pytest.raises(WebSourceError) as caught:
        adapter.extract("https://example.com/spec")
    assert caught.value.error_code == "unsafe_url"
    assert transport.targets == []


def test_dns_consumes_the_same_hop_deadline() -> None:
    times = iter((0.0, 11.0))
    adapter = WebPageAdapter(
        resolver=FakeResolver({"example.com": ("8.8.8.8",)}),
        transport=FakeTransport([]),
        clock=lambda: next(times),
    )
    with pytest.raises(WebSourceError) as caught:
        adapter.extract("https://example.com/spec")
    assert caught.value.error_code == "request_timeout"


def test_dns_failure_has_bounded_stage_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_resolution(*_args: object, **_kwargs: object) -> object:
        raise socket.gaierror("private resolver detail")

    monkeypatch.setattr(socket, "getaddrinfo", fail_resolution)
    with pytest.raises(WebSourceError) as caught:
        SystemWebResolver().resolve(
            "example.com",
            443,
            HopDeadline(10.0, lambda: 0.0),
        )

    assert caught.value.error_code == "dns_resolution_failed"
    assert caught.value.data == {
        "stage": "dns_resolution",
        "detail_type": "gaierror",
    }


def test_redirect_revalidates_target_and_enforces_limit() -> None:
    private_adapter, private_transport = _adapter(
        [
            WebHttpResponse(
                302,
                (("Location", "http://private.example/spec"),),
                b"",
            )
        ],
        {
            "example.com": ("8.8.8.8",),
            "private.example": ("10.0.0.2",),
        },
    )
    with pytest.raises(WebSourceError) as caught:
        private_adapter.extract("https://example.com/spec")
    assert caught.value.error_code == "unsafe_url"
    assert len(private_transport.targets) == 1

    redirects = [
        WebHttpResponse(302, (("Location", f"/spec/{index}"),), b"")
        for index in range(6)
    ]
    limited, transport = _adapter(redirects)
    with pytest.raises(WebSourceError) as caught:
        limited.extract("https://example.com/spec")
    assert caught.value.error_code == "too_many_redirects"
    assert len(transport.targets) == 6


def test_redirect_returns_requested_and_final_urls() -> None:
    adapter, _transport = _adapter(
        [
            WebHttpResponse(
                302,
                (("Location", "https://static.example.com/spec-v2"),),
                b"",
            ),
            _response(b"Requirements", content_type="text/plain"),
        ]
    )
    page = adapter.extract("https://example.com/spec")
    assert page.requested_url == "https://example.com/spec"
    assert page.final_url == "https://static.example.com/spec-v2"


@pytest.mark.parametrize("status", [404, 500])
def test_http_error_status_is_not_successful_content(status: int) -> None:
    adapter, _transport = _adapter(
        [WebHttpResponse(status, (("Content-Type", "text/plain"),), b"")]
    )
    with pytest.raises(WebSourceError) as caught:
        adapter.extract("https://example.com/spec")
    assert caught.value.error_code == "http_status_error"
    assert caught.value.data == {"status_code": status}


class FailingTransport:
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code

    def get(
        self,
        target: ResolvedWebTarget,
        deadline: HopDeadline,
    ) -> WebHttpResponse:
        del target, deadline
        raise WebSourceError(self.error_code, "failed")


@pytest.mark.parametrize("error_code", ["network_error", "request_timeout"])
def test_transport_error_classification_is_preserved(error_code: str) -> None:
    adapter = WebPageAdapter(
        resolver=FakeResolver({"example.com": ("8.8.8.8",)}),
        transport=FailingTransport(error_code),
    )
    with pytest.raises(WebSourceError) as caught:
        adapter.extract("https://example.com/spec")
    assert caught.value.error_code == error_code


def test_html_xhtml_and_plain_text_normalization() -> None:
    html = b"""
    <html><head><title> Project   Spec </title><style>hidden css</style></head>
    <body><nav>menu noise<br/></nav><h1>Requirements</h1>
    <p>Hello <strong>world</strong>.</p><ul><li>First</li><li>Second</li></ul>
    <table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>Unicode</td></tr></table>
    <script>hidden script</script><noscript>hidden fallback</noscript></body></html>
    """
    adapter, _transport = _adapter(
        [
            _response(html),
            _response(b"<html><body><p>XHTML body</p></body></html>", content_type="application/xhtml+xml"),
            _response("A\r\n\rB  C\n".encode(), content_type="text/plain"),
        ]
    )
    page = adapter.extract("https://example.com/html")
    xhtml = adapter.extract("https://example.com/xhtml")
    plain = adapter.extract("https://example.com/raw")

    assert page.title == "Project Spec"
    assert "Requirements" in page.text
    assert "Hello world." in page.text
    assert "Name\tValue" in page.text
    assert "hidden" not in page.text
    assert xhtml.text == "XHTML body"
    assert plain.title == ""
    assert plain.text == "A\n\nB  C"


def test_unicode_charset_and_empty_javascript_shell() -> None:
    encoded = "<html><body><p>中文需求</p></body></html>".encode("gb18030")
    adapter, _transport = _adapter(
        [
            _response(encoded, content_type="text/html; charset=gb18030"),
            _response(b"<html><body><script>render()</script></body></html>"),
        ]
    )
    assert adapter.extract("https://example.com/chinese").text == "中文需求"
    with pytest.raises(WebSourceError) as caught:
        adapter.extract("https://example.com/app")
    assert caught.value.error_code == "webpage_text_unavailable"


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("application/pdf", "unsupported_content_type"),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "unsupported_content_type"),
        ("image/png", "unsupported_content_type"),
    ],
)
def test_binary_content_types_are_not_remote_documents(
    content_type: str,
    expected: str,
) -> None:
    adapter, _transport = _adapter([_response(b"binary", content_type=content_type)])
    with pytest.raises(WebSourceError) as caught:
        adapter.extract("https://example.com/file")
    assert caught.value.error_code == expected


def test_content_encoding_identity_gzip_and_deflate() -> None:
    body = b"bounded requirement"
    gzip_encoder = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
    gzip_body = gzip_encoder.compress(body) + gzip_encoder.flush()
    zlib_body = zlib.compress(body)
    raw_encoder = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw_body = raw_encoder.compress(body) + raw_encoder.flush()

    assert _decode_content_encoding(_response(body)) == body
    assert _decode_content_encoding(
        _response(body, extra_headers=(("Content-Encoding", "identity"),))
    ) == body
    assert _decode_content_encoding(
        _response(gzip_body, extra_headers=(("Content-Encoding", "gzip"),))
    ) == body
    assert _decode_content_encoding(
        _response(zlib_body, extra_headers=(("Content-Encoding", "deflate"),))
    ) == body
    assert _decode_content_encoding(
        _response(raw_body, extra_headers=(("Content-Encoding", "deflate"),))
    ) == body


@pytest.mark.parametrize("encoding", ["br", "zstd", "compress", "gzip, deflate"])
def test_unsupported_content_encoding_is_distinct(encoding: str) -> None:
    with pytest.raises(WebSourceError) as caught:
        _decode_content_encoding(
            _response(b"encoded", extra_headers=(("Content-Encoding", encoding),))
        )
    assert caught.value.error_code == "unsupported_content_encoding"


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
def test_malformed_content_encoding_is_distinct(encoding: str) -> None:
    with pytest.raises(WebSourceError) as caught:
        _decode_content_encoding(
            _response(b"not compressed", extra_headers=(("Content-Encoding", encoding),))
        )
    assert caught.value.error_code == "malformed_content_encoding"


def test_decoded_body_limit_is_independent() -> None:
    encoded = zlib.compress(b"x" * (MAX_WEB_BODY_BYTES + 1))
    with pytest.raises(WebSourceError) as caught:
        _decode_content_encoding(
            _response(encoded, extra_headers=(("Content-Encoding", "deflate"),))
        )
    assert caught.value.error_code == "response_too_large"

    with pytest.raises(WebSourceError) as caught:
        _decode_content_encoding(_response(b"x" * (MAX_WEB_BODY_BYTES + 1)))
    assert caught.value.error_code == "response_too_large"


@pytest.mark.parametrize(
    "headers",
    [
        (("Transfer-Encoding", "chunked"), ("Content-Length", "3")),
        (("Content-Length", "3"), ("Content-Length", "3")),
        (("Content-Length", "3"), ("Content-Length", "4")),
        (("Content-Length", "-1"),),
        (("Transfer-Encoding", "gzip, chunked"),),
        (("Transfer-Encoding", "chunked"), ("Transfer-Encoding", "chunked")),
    ],
)
def test_ambiguous_framing_is_rejected_before_body_read(
    headers: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises(WebSourceError) as caught:
        _validate_response_framing(headers)
    assert caught.value.error_code == "http_protocol_error"


class FakeResponseSocket:
    def __init__(self, raw_response: bytes) -> None:
        self.stream = BytesIO(raw_response)

    def makefile(self, _mode: str, buffering: int | None = None) -> BytesIO:
        del buffering
        return self.stream


class ReadConnection:
    sock = None
    timeout: float | None = None


def _parsed_response(raw: bytes) -> http.client.HTTPResponse:
    response = http.client.HTTPResponse(FakeResponseSocket(raw))  # type: ignore[arg-type]
    response.begin()
    return response


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello", b"hello"),
        (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n", b"hello"),
        (b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nhello", b"hello"),
    ],
)
def test_standard_http_framing_is_reused(raw: bytes, expected: bytes) -> None:
    response = _parsed_response(raw)
    headers = tuple(response.headers.items())
    content_length = _validate_response_framing(headers)
    body = _read_encoded_body(
        response,
        ReadConnection(),  # type: ignore[arg-type]
        HopDeadline(10.0, lambda: 0.0),
        content_length,
    )
    assert body == expected


def test_chunked_gzip_is_deframed_before_content_decoding() -> None:
    encoder = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
    encoded = encoder.compress(b"hello") + encoder.flush()
    raw = (
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
        b"Content-Encoding: gzip\r\nContent-Type: text/plain\r\n\r\n"
        + f"{len(encoded):X}\r\n".encode()
        + encoded
        + b"\r\n0\r\n\r\n"
    )
    response = _parsed_response(raw)
    headers = tuple(response.headers.items())
    body = _read_encoded_body(
        response,
        ReadConnection(),  # type: ignore[arg-type]
        HopDeadline(10.0, lambda: 0.0),
        _validate_response_framing(headers),
    )
    decoded = _decode_content_encoding(
        WebHttpResponse(response.status, headers, body)
    )
    assert decoded == b"hello"


def test_malformed_chunked_response_is_protocol_error() -> None:
    response = _parsed_response(
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
        b"not-a-size\r\nbody\r\n"
    )
    with pytest.raises(WebSourceError) as caught:
        _read_encoded_body(
            response,
            ReadConnection(),  # type: ignore[arg-type]
            HopDeadline(10.0, lambda: 0.0),
            _validate_response_framing(tuple(response.headers.items())),
        )
    assert caught.value.error_code == "http_protocol_error"


def test_encoded_body_limit_and_hop_deadline() -> None:
    with pytest.raises(WebSourceError) as caught:
        _validate_response_framing(
            (("Content-Length", str(MAX_WEB_BODY_BYTES + 1)),)
        )
    assert caught.value.error_code == "response_too_large"

    deadline = HopDeadline(10.0, lambda: 10.0)
    with pytest.raises(WebSourceError) as caught:
        deadline.remaining_time()
    assert caught.value.error_code == "request_timeout"

    with pytest.raises(WebSourceError) as caught:
        deadline.remaining_time("response_headers")
    assert caught.value.data == {
        "stage": "response_headers",
        "detail_type": "TimeoutError",
    }


class RecordingSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []
        self.options: list[tuple[int, int, int]] = []
        self.sent = bytearray()

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def setsockopt(self, level: int, option: int, value: int) -> None:
        self.options.append((level, option, value))

    def sendall(self, data: bytes) -> None:
        self.sent.extend(data)

    def close(self) -> None:
        return None


class RecordingContext:
    def __init__(self) -> None:
        self.server_hostnames: list[str] = []

    def wrap_socket(
        self,
        raw_socket: RecordingSocket,
        *,
        server_hostname: str,
    ) -> RecordingSocket:
        self.server_hostnames.append(server_hostname)
        return raw_socket


def _target(
    *,
    scheme: str = "https",
    validated_ips: tuple[str, ...] = ("8.8.8.8",),
) -> ResolvedWebTarget:
    port = 443 if scheme == "https" else 80
    return ResolvedWebTarget(
        url=f"{scheme}://example.com/spec",
        scheme=scheme,
        hostname="example.com",
        port=port,
        request_target="/spec",
        validated_ips=validated_ips,
    )


def test_fixed_ip_connector_preserves_logical_host_timeout_and_source_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, int], float, object]] = []
    created = RecordingSocket()

    def create_connection(
        address: tuple[str, int],
        timeout: float,
        source_address: object,
    ) -> RecordingSocket:
        calls.append((address, timeout, source_address))
        return created

    monkeypatch.setattr(socket, "create_connection", create_connection)
    deadline = HopDeadline(10.0, lambda: 0.0)
    target = _target(scheme="http", validated_ips=("1.1.1.1",))
    connection = _new_connection(target, "1.1.1.1", deadline)
    connection.source_address = ("0.0.0.0", 0)
    connection.connect()

    assert connection.host == "example.com"
    assert calls == [(('1.1.1.1', 80), 10.0, ("0.0.0.0", 0))]
    assert isinstance(connection._create_connection, _FixedIPConnector)


def test_https_uses_stdlib_connect_and_original_hostname_for_tls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addresses: list[tuple[str, int]] = []
    raw_socket = RecordingSocket()
    context = RecordingContext()

    def create_connection(
        address: tuple[str, int],
        timeout: float,
        source_address: object,
    ) -> RecordingSocket:
        del timeout, source_address
        addresses.append(address)
        return raw_socket

    monkeypatch.setattr(socket, "create_connection", create_connection)
    deadline = HopDeadline(10.0, lambda: 0.0)
    https_connection = _new_connection(
        _target(), "8.8.8.8", deadline
    )
    https_connection._context = context  # type: ignore[assignment]
    https_connection.connect()

    assert type(https_connection) is http.client.HTTPSConnection
    assert https_connection.host == "example.com"
    assert addresses == [("8.8.8.8", 443)]
    assert context.server_hostnames == ["example.com"]


def test_http_client_generates_host_and_declared_request_headers() -> None:
    connection = http.client.HTTPConnection("example.com", 8080, timeout=10.0)
    recording_socket = RecordingSocket()
    connection.sock = recording_socket  # type: ignore[assignment]

    connection.request("GET", "/spec", headers=_REQUEST_HEADERS)

    request = recording_socket.sent.decode("latin-1")
    assert "Host: example.com:8080\r\n" in request
    assert _REQUEST_HEADERS["User-Agent"].startswith("CodeLoop/")
    assert _REQUEST_HEADERS["User-Agent"] != "CodeLoop/"
    assert f"User-Agent: {_REQUEST_HEADERS['User-Agent']}\r\n" in request
    assert (
        "Accept: text/html, application/xhtml+xml, "
        "text/plain;q=0.9, */*;q=0.1\r\n"
    ) in request
    assert "Accept-Encoding: identity\r\n" in request
    assert "Connection: close\r\n" in request


class ScriptedResponse:
    def __init__(
        self,
        body: bytes = b"ok",
        *,
        read_error: BaseException | None = None,
    ) -> None:
        self.status = 200
        self.headers = Message()
        self.headers["Content-Length"] = str(len(body))
        self._body = body
        self._read_error = read_error
        self._read = False

    def read1(self, _size: int) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        if self._read:
            return b""
        self._read = True
        return self._body

    def close(self) -> None:
        return None


class ScriptedConnection:
    def __init__(
        self,
        events: list[str],
        label: str,
        *,
        connect_error: BaseException | None = None,
        request_error: BaseException | None = None,
        headers_error: BaseException | None = None,
        response: ScriptedResponse | None = None,
    ) -> None:
        self.events = events
        self.label = label
        self.connect_error = connect_error
        self.request_error = request_error
        self.headers_error = headers_error
        self.response = response or ScriptedResponse()
        self.sock: RecordingSocket | None = None
        self.timeout: float | None = None

    def connect(self) -> None:
        self.events.append(f"connect:{self.label}")
        if self.connect_error is not None:
            raise self.connect_error
        self.sock = RecordingSocket()

    def request(
        self,
        method: str,
        target: str,
        *,
        headers: dict[str, str],
    ) -> None:
        del method, target, headers
        self.events.append(f"request:{self.label}")
        if self.request_error is not None:
            raise self.request_error

    def getresponse(self) -> ScriptedResponse:
        self.events.append(f"headers:{self.label}")
        if self.headers_error is not None:
            raise self.headers_error
        return self.response

    def close(self) -> None:
        self.events.append(f"close:{self.label}")


def test_multiple_ips_fallback_only_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    outcomes: dict[str, BaseException | None] = {
        "1.1.1.1": ConnectionRefusedError("private detail"),
        "8.8.4.4": ssl.SSLError(1, "private detail"),
        "8.8.8.8": None,
    }
    deadlines: list[HopDeadline] = []

    def connection_factory(
        _target_value: ResolvedWebTarget,
        connected_ip: str,
        deadline: HopDeadline,
    ) -> ScriptedConnection:
        deadlines.append(deadline)
        return ScriptedConnection(
            events,
            connected_ip,
            connect_error=outcomes[connected_ip],
        )

    monkeypatch.setattr(web_sources, "_new_connection", connection_factory)
    deadline = HopDeadline(10.0, lambda: 0.0)
    result = StdlibWebTransport().get(
        _target(validated_ips=("1.1.1.1", "8.8.4.4", "8.8.8.8")),
        deadline,
    )

    assert result.encoded_body == b"ok"
    assert events == [
        "connect:1.1.1.1",
        "close:1.1.1.1",
        "connect:8.8.4.4",
        "close:8.8.4.4",
        "connect:8.8.8.8",
        "request:8.8.8.8",
        "headers:8.8.8.8",
        "close:8.8.8.8",
    ]
    assert deadlines == [deadline, deadline, deadline]


def test_only_first_three_validated_ips_are_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []

    def connection_factory(
        _target_value: ResolvedWebTarget,
        connected_ip: str,
        _deadline: HopDeadline,
    ) -> ScriptedConnection:
        attempted.append(connected_ip)
        return ScriptedConnection(
            [],
            connected_ip,
            connect_error=ConnectionResetError("private detail"),
        )

    monkeypatch.setattr(web_sources, "_new_connection", connection_factory)
    with pytest.raises(WebSourceError) as caught:
        StdlibWebTransport().get(
            _target(
                validated_ips=(
                    "1.1.1.1",
                    "8.8.4.4",
                    "8.8.8.8",
                    "9.9.9.9",
                )
            ),
            HopDeadline(10.0, lambda: 0.0),
        )

    assert attempted == ["1.1.1.1", "8.8.4.4", "8.8.8.8"]
    assert len(attempted) == MAX_WEB_CONNECT_ATTEMPTS
    assert caught.value.error_code == "network_error"
    assert caught.value.data == {
        "stage": "tcp_connect",
        "detail_type": "ConnectionResetError",
    }


def test_certificate_error_is_tls_stage_and_not_plain_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def connection_factory(
        _target_value: ResolvedWebTarget,
        connected_ip: str,
        _deadline: HopDeadline,
    ) -> ScriptedConnection:
        return ScriptedConnection(
            [],
            connected_ip,
            connect_error=ssl.SSLCertVerificationError(
                1,
                "private certificate detail",
            ),
        )

    monkeypatch.setattr(web_sources, "_new_connection", connection_factory)
    with pytest.raises(WebSourceError) as caught:
        StdlibWebTransport().get(
            _target(),
            HopDeadline(10.0, lambda: 0.0),
        )

    assert caught.value.error_code == "network_error"
    assert caught.value.data == {
        "stage": "tls_handshake",
        "detail_type": "SSLCertVerificationError",
    }


def test_request_failure_never_falls_back_to_another_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []

    def connection_factory(
        _target_value: ResolvedWebTarget,
        connected_ip: str,
        _deadline: HopDeadline,
    ) -> ScriptedConnection:
        created.append(connected_ip)
        return ScriptedConnection(
            [],
            connected_ip,
            request_error=ConnectionResetError("private request detail"),
        )

    monkeypatch.setattr(web_sources, "_new_connection", connection_factory)
    with pytest.raises(WebSourceError) as caught:
        StdlibWebTransport().get(
            _target(validated_ips=("1.1.1.1", "8.8.8.8")),
            HopDeadline(10.0, lambda: 0.0),
        )

    assert created == ["1.1.1.1"]
    assert caught.value.data == {
        "stage": "request_send",
        "detail_type": "ConnectionResetError",
    }


@pytest.mark.parametrize(
    ("failure", "expected_code", "expected_stage"),
    [
        (
            ConnectionResetError("private header detail"),
            "network_error",
            "response_headers",
        ),
        (
            http.client.BadStatusLine("private protocol detail"),
            "http_protocol_error",
            "response_headers",
        ),
    ],
)
def test_response_header_diagnostics_do_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_code: str,
    expected_stage: str,
) -> None:
    created: list[str] = []

    def connection_factory(
        _target_value: ResolvedWebTarget,
        connected_ip: str,
        _deadline: HopDeadline,
    ) -> ScriptedConnection:
        created.append(connected_ip)
        return ScriptedConnection([], connected_ip, headers_error=failure)

    monkeypatch.setattr(web_sources, "_new_connection", connection_factory)
    with pytest.raises(WebSourceError) as caught:
        StdlibWebTransport().get(
            _target(validated_ips=("1.1.1.1", "8.8.8.8")),
            HopDeadline(10.0, lambda: 0.0),
        )

    assert created == ["1.1.1.1"]
    assert caught.value.error_code == expected_code
    assert caught.value.data == {
        "stage": expected_stage,
        "detail_type": type(failure).__name__,
    }
    assert "private" not in json.dumps(caught.value.data)


def test_response_body_diagnostic_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []

    def connection_factory(
        _target_value: ResolvedWebTarget,
        connected_ip: str,
        _deadline: HopDeadline,
    ) -> ScriptedConnection:
        created.append(connected_ip)
        return ScriptedConnection(
            [],
            connected_ip,
            response=ScriptedResponse(
                read_error=ConnectionResetError("private body detail")
            ),
        )

    monkeypatch.setattr(web_sources, "_new_connection", connection_factory)
    with pytest.raises(WebSourceError) as caught:
        StdlibWebTransport().get(
            _target(validated_ips=("1.1.1.1", "8.8.8.8")),
            HopDeadline(10.0, lambda: 0.0),
        )

    assert created == ["1.1.1.1"]
    assert caught.value.data == {
        "stage": "response_body",
        "detail_type": "ConnectionResetError",
    }


def test_tool_cursor_contract_determinism_and_read_only(tmp_path: Any) -> None:
    source = "abcdef"
    page = ExtractedWebPage(
        requested_url="https://example.com/spec",
        final_url="https://example.com/spec",
        title="Spec",
        content_type="text/plain",
        text=source,
    )
    adapter = StaticAdapter(page)
    registry = ToolRegistry(Workspace(tmp_path), webpage_adapter=adapter)  # type: ignore[arg-type]
    before = tuple(tmp_path.iterdir())

    first = registry.dispatch(
        "read_webpage",
        json.dumps({"url": page.requested_url, "cursor": 0, "max_chars": 3}),
    )
    repeated = registry.dispatch(
        "read_webpage",
        json.dumps({"url": page.requested_url, "cursor": 0, "max_chars": 3}),
    )
    eof = registry.dispatch(
        "read_webpage",
        json.dumps({"url": page.requested_url, "cursor": 6}),
    )
    overflow = registry.dispatch(
        "read_webpage",
        json.dumps({"url": page.requested_url, "cursor": 7}),
    )

    assert first == repeated
    assert first["data"]["text"] == "abc"
    assert first["data"]["next_cursor"] == 3
    assert eof["data"]["text"] == ""
    assert eof["data"]["truncated"] is False
    assert eof["data"]["next_cursor"] is None
    assert overflow["error_code"] == "invalid_arguments"
    assert tuple(tmp_path.iterdir()) == before


@pytest.mark.parametrize(
    "error_code",
    [
        "webpage_text_unavailable",
        "unsupported_content_type",
        "unsupported_content_encoding",
        "malformed_content_encoding",
        "http_protocol_error",
        "network_error",
    ],
)
def test_tool_result_preserves_specific_web_error(
    tmp_path: Any,
    error_code: str,
) -> None:
    result = ToolRegistry(
        Workspace(tmp_path),  # type: ignore[arg-type]
        webpage_adapter=ErrorAdapter(error_code),  # type: ignore[arg-type]
    ).dispatch("read_webpage", '{"url":"https://example.com/spec"}')
    assert result == {
        "ok": False,
        "error_code": error_code,
        "message": "specific web source failure",
    }


def test_tool_result_preserves_only_bounded_transport_diagnostics(
    tmp_path: Any,
) -> None:
    diagnostics = {
        "stage": "tls_handshake",
        "detail_type": "SSLCertVerificationError",
    }
    result = ToolRegistry(
        Workspace(tmp_path),  # type: ignore[arg-type]
        webpage_adapter=ErrorAdapter("network_error", diagnostics),  # type: ignore[arg-type]
    ).dispatch("read_webpage", '{"url":"https://example.com/spec"}')

    assert result == {
        "ok": False,
        "error_code": "network_error",
        "message": "specific web source failure",
        "data": diagnostics,
    }


@pytest.mark.parametrize("max_chars", [0, 20_001, True])
def test_tool_max_chars_validation(tmp_path: Any, max_chars: object) -> None:
    page = ExtractedWebPage(
        "https://example.com/",
        "https://example.com/",
        "",
        "text/plain",
        "text",
    )
    result = ToolRegistry(
        Workspace(tmp_path),  # type: ignore[arg-type]
        webpage_adapter=StaticAdapter(page),
    ).dispatch(
        "read_webpage",
        json.dumps({"url": page.requested_url, "max_chars": max_chars}),
    )
    assert result["error_code"] == "invalid_arguments"


@pytest.mark.parametrize("cursor", [-1, 1.5, True])
def test_tool_cursor_type_and_lower_bound_validation(
    tmp_path: Any,
    cursor: object,
) -> None:
    page = ExtractedWebPage(
        "https://example.com/",
        "https://example.com/",
        "",
        "text/plain",
        "text",
    )
    result = ToolRegistry(
        Workspace(tmp_path),  # type: ignore[arg-type]
        webpage_adapter=StaticAdapter(page),
    ).dispatch(
        "read_webpage",
        json.dumps({"url": page.requested_url, "cursor": cursor}),
    )
    assert result["error_code"] == "invalid_arguments"
