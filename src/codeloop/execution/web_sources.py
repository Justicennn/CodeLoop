"""Safe bounded adapters for explicit HTTP and HTTPS textual sources."""

from __future__ import annotations

import http.client
import ipaddress
import queue
import re
import socket
import ssl
import threading
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from email.message import Message
from html.parser import HTMLParser
from importlib.metadata import PackageNotFoundError, version
from time import monotonic
from typing import Protocol
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

MAX_WEB_URL_CHARS = 2_000
MAX_WEB_BODY_BYTES = 2 * 1024 * 1024
MAX_WEB_TITLE_CHARS = 500
WEB_HOP_TIMEOUT_SECONDS = 10.0
MAX_WEB_REDIRECTS = 5
MAX_WEB_CONNECT_ATTEMPTS = 3
_READ_CHUNK_BYTES = 64 * 1024
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}
_SUPPORTED_CONTENT_TYPES = {*_HTML_CONTENT_TYPES, "text/plain"}
_IGNORED_TAGS = {"script", "style", "noscript", "template", "nav"}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "ul",
}
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")
_DECIMAL = re.compile(r"[0-9]+\Z")
_META_CHARSET = re.compile(
    br"<meta\s+[^>]*charset\s*=\s*['\"]?\s*([A-Za-z0-9._:-]+)",
    re.IGNORECASE,
)
_META_HTTP_EQUIV_CHARSET = re.compile(
    br"<meta\s+[^>]*content\s*=\s*['\"][^'\"]*charset\s*=\s*"
    br"([A-Za-z0-9._:-]+)[^'\"]*['\"]",
    re.IGNORECASE,
)

try:
    _CODELOOP_VERSION = version("codeloop")
except PackageNotFoundError:
    _CODELOOP_VERSION = "0.1.0"

_REQUEST_HEADERS = {
    "Accept": "text/html, application/xhtml+xml, text/plain;q=0.9, */*;q=0.1",
    "Accept-Encoding": "identity",
    "Connection": "close",
    "User-Agent": f"CodeLoop/{_CODELOOP_VERSION}",
}


class WebSourceError(Exception):
    """A stable, safe failure while reading an explicit web source."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        data: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.data = data


@dataclass(frozen=True)
class HopDeadline:
    expires_at: float
    clock: Callable[[], float]

    def remaining_time(self, stage: str | None = None) -> float:
        remaining = self.expires_at - self.clock()
        if remaining <= 0:
            raise WebSourceError(
                "request_timeout",
                "The webpage request hop exceeded its time limit.",
                data=(
                    {"stage": stage, "detail_type": "TimeoutError"}
                    if stage is not None
                    else None
                ),
            )
        return remaining


@dataclass(frozen=True)
class ResolvedWebTarget:
    url: str
    scheme: str
    hostname: str
    port: int
    request_target: str
    validated_ips: tuple[str, ...]


@dataclass(frozen=True)
class WebHttpResponse:
    status: int
    headers: tuple[tuple[str, str], ...]
    encoded_body: bytes

    def header_values(self, name: str) -> tuple[str, ...]:
        folded = name.casefold()
        return tuple(value for key, value in self.headers if key.casefold() == folded)


@dataclass(frozen=True)
class ExtractedWebPage:
    requested_url: str
    final_url: str
    title: str
    content_type: str
    text: str


class WebResolver(Protocol):
    def resolve(
        self,
        hostname: str,
        port: int,
        deadline: HopDeadline,
    ) -> tuple[str, ...]: ...


class WebTransport(Protocol):
    def get(
        self,
        target: ResolvedWebTarget,
        deadline: HopDeadline,
    ) -> WebHttpResponse: ...


class SystemWebResolver:
    """Run blocking system DNS behind the hop's bounded wait."""

    def resolve(
        self,
        hostname: str,
        port: int,
        deadline: HopDeadline,
    ) -> tuple[str, ...]:
        outcomes: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def worker() -> None:
            try:
                records = socket.getaddrinfo(
                    hostname,
                    port,
                    type=socket.SOCK_STREAM,
                    proto=socket.IPPROTO_TCP,
                )
            except OSError as exc:
                outcomes.put((False, exc))
            else:
                outcomes.put((True, tuple(record[4][0] for record in records)))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            ok, value = outcomes.get(
                timeout=deadline.remaining_time("dns_resolution")
            )
        except queue.Empty as exc:
            raise WebSourceError(
                "request_timeout",
                "DNS resolution exceeded the webpage request hop time limit.",
                data={
                    "stage": "dns_resolution",
                    "detail_type": type(exc).__name__,
                },
            ) from exc
        if not ok:
            error = WebSourceError(
                "dns_resolution_failed",
                "The webpage hostname could not be resolved.",
                data={
                    "stage": "dns_resolution",
                    "detail_type": (
                        type(value).__name__
                        if isinstance(value, BaseException)
                        else "DNSResolutionError"
                    ),
                },
            )
            if isinstance(value, BaseException):
                raise error from value
            raise error
        addresses = value
        if not isinstance(addresses, tuple) or not addresses:
            raise WebSourceError(
                "dns_resolution_failed",
                "The webpage hostname returned no addresses.",
                data={
                    "stage": "dns_resolution",
                    "detail_type": "NoAddressResult",
                },
            )
        return tuple(str(address) for address in addresses)


@dataclass
class _FixedIPConnector:
    """Replace only HTTPConnection's physical socket target."""

    connected_ip: str
    port: int
    deadline: HopDeadline

    def __call__(
        self,
        _logical_address: tuple[str, int],
        timeout: float,
        source_address: tuple[str, int] | None,
    ) -> socket.socket:
        self.deadline.remaining_time("tcp_connect")
        return socket.create_connection(
            (self.connected_ip, self.port),
            timeout,
            source_address,
        )


def _diagnostic_data(stage: str, exc: BaseException) -> dict[str, object]:
    return {"stage": stage, "detail_type": type(exc).__name__}


def _request_timeout(stage: str, exc: BaseException) -> WebSourceError:
    return WebSourceError(
        "request_timeout",
        "The webpage request hop exceeded its time limit.",
        data=_diagnostic_data(stage, exc),
    )


def _network_error(stage: str, exc: BaseException) -> WebSourceError:
    return WebSourceError(
        "network_error",
        "The webpage network request failed.",
        data=_diagnostic_data(stage, exc),
    )


def _refresh_socket_timeout(
    connection: http.client.HTTPConnection,
    deadline: HopDeadline,
    stage: str,
) -> None:
    timeout = deadline.remaining_time(stage)
    connection.timeout = timeout
    if connection.sock is not None:
        connection.sock.settimeout(timeout)


def _new_connection(
    target: ResolvedWebTarget,
    connected_ip: str,
    deadline: HopDeadline,
) -> http.client.HTTPConnection:
    timeout = deadline.remaining_time("tcp_connect")
    if target.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            target.hostname,
            target.port,
            timeout=timeout,
        )
    else:
        connection = http.client.HTTPConnection(
            target.hostname,
            target.port,
            timeout=timeout,
        )
    connection._create_connection = _FixedIPConnector(  # type: ignore[attr-defined]
        connected_ip,
        target.port,
        deadline,
    )
    return connection


class StdlibWebTransport:
    """HTTP/1.1 transport with fixed-IP sockets and standard framing parsing."""

    def get(
        self,
        target: ResolvedWebTarget,
        deadline: HopDeadline,
    ) -> WebHttpResponse:
        last_connect_error: WebSourceError | None = None
        for connected_ip in target.validated_ips[:MAX_WEB_CONNECT_ATTEMPTS]:
            connection = _new_connection(target, connected_ip, deadline)
            try:
                try:
                    connection.connect()
                except (socket.timeout, TimeoutError) as exc:
                    last_connect_error = _request_timeout("tcp_connect", exc)
                    continue
                except ssl.SSLError as exc:
                    last_connect_error = _network_error("tls_handshake", exc)
                    continue
                except OSError as exc:
                    last_connect_error = _network_error("tcp_connect", exc)
                    continue

                return self._request_connected(connection, target, deadline)
            finally:
                try:
                    connection.close()
                except OSError:
                    pass

        if last_connect_error is not None:
            raise last_connect_error
        raise WebSourceError(
            "network_error",
            "The webpage network request failed.",
            data={"stage": "tcp_connect", "detail_type": "NoConnectionAttempt"},
        )

    def _request_connected(
        self,
        connection: http.client.HTTPConnection,
        target: ResolvedWebTarget,
        deadline: HopDeadline,
    ) -> WebHttpResponse:
        response: http.client.HTTPResponse | None = None
        try:
            try:
                _refresh_socket_timeout(connection, deadline, "request_send")
                connection.request(
                    "GET",
                    target.request_target,
                    headers=_REQUEST_HEADERS,
                )
            except WebSourceError:
                raise
            except (socket.timeout, TimeoutError) as exc:
                raise _request_timeout("request_send", exc) from exc
            except (http.client.HTTPException, ssl.SSLError, OSError) as exc:
                raise _network_error("request_send", exc) from exc

            try:
                _refresh_socket_timeout(connection, deadline, "response_headers")
                response = connection.getresponse()
            except WebSourceError:
                raise
            except (socket.timeout, TimeoutError) as exc:
                raise _request_timeout("response_headers", exc) from exc
            except http.client.HTTPException as exc:
                raise WebSourceError(
                    "http_protocol_error",
                    "The webpage returned an invalid HTTP response.",
                    data=_diagnostic_data("response_headers", exc),
                ) from exc
            except (ssl.SSLError, OSError) as exc:
                raise _network_error("response_headers", exc) from exc

            headers = tuple((key, value) for key, value in response.headers.items())
            try:
                content_length = _validate_response_framing(headers)
            except WebSourceError as exc:
                if exc.data is not None:
                    raise
                raise WebSourceError(
                    exc.error_code,
                    exc.message,
                    data={
                        "stage": "response_headers",
                        "detail_type": "FramingPolicyError",
                    },
                ) from exc
            should_read_body = (
                200 <= response.status < 300
                and response.status not in {204, 205}
            )
            encoded_body = (
                _read_encoded_body(
                    response,
                    connection,
                    deadline,
                    content_length,
                )
                if should_read_body
                else b""
            )
            return WebHttpResponse(response.status, headers, encoded_body)
        finally:
            if response is not None:
                try:
                    response.close()
                except OSError:
                    pass


class WebPageAdapter:
    """Fetch and normalize one user-provided textual webpage."""

    def __init__(
        self,
        *,
        resolver: WebResolver | None = None,
        transport: WebTransport | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._resolver = resolver or SystemWebResolver()
        self._transport = transport or StdlibWebTransport()
        self._clock = clock

    def extract(self, url: str) -> ExtractedWebPage:
        requested_url = normalize_web_url(url)
        current_url = requested_url
        redirects = 0
        while True:
            deadline = HopDeadline(
                self._clock() + WEB_HOP_TIMEOUT_SECONDS,
                self._clock,
            )
            target = _resolve_target(current_url, self._resolver, deadline)
            response = self._transport.get(target, deadline)
            if response.status in _REDIRECT_STATUSES:
                locations = response.header_values("Location")
                if len(locations) != 1 or not locations[0].strip():
                    raise WebSourceError(
                        "http_protocol_error",
                        "The webpage redirect must contain exactly one Location.",
                    )
                if redirects >= MAX_WEB_REDIRECTS:
                    raise WebSourceError(
                        "too_many_redirects",
                        "The webpage exceeded the redirect limit.",
                    )
                redirects += 1
                current_url = normalize_web_url(
                    urljoin(current_url, locations[0].strip())
                )
                continue
            if not 200 <= response.status < 300:
                raise WebSourceError(
                    "http_status_error",
                    f"The webpage returned HTTP status {response.status}.",
                    data={"status_code": response.status},
                )
            if response.status in {204, 205}:
                raise WebSourceError(
                    "webpage_text_unavailable",
                    "The webpage response contains no textual representation.",
                )
            decoded_body = _decode_content_encoding(response)
            content_type, charset = _content_type(response)
            decoded_text = _decode_text(decoded_body, charset, content_type)
            if content_type in _HTML_CONTENT_TYPES:
                title, text = _extract_html(decoded_text)
            else:
                title = ""
                text = _normalize_plain_text(decoded_text)
            if not text.strip():
                raise WebSourceError(
                    "webpage_text_unavailable",
                    "The webpage contains no extractable textual content.",
                )
            return ExtractedWebPage(
                requested_url=requested_url,
                final_url=current_url,
                title=title,
                content_type=content_type,
                text=text,
            )


def normalize_web_url(url: str) -> str:
    """Validate and deterministically normalize one absolute web URL."""
    if not isinstance(url, str) or not url or len(url) > MAX_WEB_URL_CHARS:
        raise WebSourceError("invalid_url", "url must be a non-empty bounded string.")
    if _CONTROL_CHARACTER.search(url):
        raise WebSourceError("invalid_url", "url cannot contain control characters.")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise WebSourceError("invalid_url", "The webpage URL is malformed.") from exc
    scheme = parsed.scheme.casefold()
    if not scheme:
        raise WebSourceError("invalid_url", "The webpage URL must be absolute.")
    if scheme not in {"http", "https"}:
        raise WebSourceError(
            "unsupported_url_scheme",
            "read_webpage supports only http and https URLs.",
        )
    if parsed.username is not None or parsed.password is not None:
        raise WebSourceError("invalid_url", "Credential-bearing URLs are unsupported.")
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise WebSourceError("invalid_url", "The webpage URL port is invalid.") from exc
    if not hostname:
        raise WebSourceError("invalid_url", "The webpage URL requires a hostname.")
    try:
        normalized_host = hostname.encode("idna").decode("ascii").casefold()
    except UnicodeError as exc:
        raise WebSourceError("invalid_url", "The webpage hostname is invalid.") from exc
    if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
        raise WebSourceError("unsafe_url", "Localhost webpage targets are forbidden.")
    default_port = 443 if scheme == "https" else 80
    effective_port = port if port is not None else default_port
    if effective_port < 1 or effective_port > 65_535:
        raise WebSourceError("invalid_url", "The webpage URL port is invalid.")
    display_host = normalized_host
    try:
        literal = ipaddress.ip_address(normalized_host)
    except ValueError:
        literal = None
        labels = normalized_host.rstrip(".").split(".")
        if (
            len(normalized_host.rstrip(".")) > 253
            or not labels
            or any(
                not label
                or len(label) > 63
                or label.startswith("-")
                or label.endswith("-")
                or re.fullmatch(r"[a-z0-9-]+", label) is None
                for label in labels
            )
        ):
            raise WebSourceError(
                "invalid_url",
                "The webpage hostname is invalid.",
            )
    if isinstance(literal, ipaddress.IPv6Address):
        display_host = f"[{literal.compressed}]"
    elif literal is not None:
        display_host = literal.compressed
    netloc = display_host
    if effective_port != default_port:
        netloc = f"{netloc}:{effective_port}"
    path = quote(parsed.path or "/", safe="/%:@-._~!$&'()*+,;=")
    query = quote(parsed.query, safe="=&?/:@-._~!$'()*+,;%")
    normalized = urlunsplit((scheme, netloc, path, query, ""))
    if len(normalized) > MAX_WEB_URL_CHARS:
        raise WebSourceError("invalid_url", "The normalized webpage URL is too long.")
    return normalized


def _resolve_target(
    url: str,
    resolver: WebResolver,
    deadline: HopDeadline,
) -> ResolvedWebTarget:
    parsed = urlsplit(url)
    hostname = parsed.hostname
    if hostname is None:  # normalize_web_url already guarantees this.
        raise WebSourceError("invalid_url", "The webpage URL requires a hostname.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        raw_addresses = resolver.resolve(hostname, port, deadline)
    else:
        raw_addresses = (literal.compressed,)
    addresses: dict[tuple[int, bytes], ipaddress.IPv4Address | ipaddress.IPv6Address] = {}
    for raw_address in raw_addresses:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as exc:
            raise WebSourceError(
                "dns_resolution_failed",
                "DNS returned an invalid IP address.",
            ) from exc
        _require_public_address(address)
        addresses[(address.version, address.packed)] = address
    if not addresses:
        raise WebSourceError(
            "dns_resolution_failed",
            "The webpage hostname returned no addresses.",
        )
    validated_ips = tuple(
        addresses[key].compressed for key in sorted(addresses)
    )
    request_target = parsed.path or "/"
    if parsed.query:
        request_target += f"?{parsed.query}"
    return ResolvedWebTarget(
        url=url,
        scheme=parsed.scheme,
        hostname=hostname,
        port=port,
        request_target=request_target,
        validated_ips=validated_ips,
    )


def _require_public_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> None:
    mapped = (
        address.ipv4_mapped
        if isinstance(address, ipaddress.IPv6Address)
        else None
    )
    if not address.is_global or (mapped is not None and not mapped.is_global):
        raise WebSourceError(
            "unsafe_url",
            "The webpage target resolves to a non-public IP address.",
        )


def _validate_response_framing(
    headers: tuple[tuple[str, str], ...],
) -> int | None:
    transfer_values = _header_values(headers, "Transfer-Encoding")
    length_values = _header_values(headers, "Content-Length")
    if transfer_values and length_values:
        raise WebSourceError(
            "http_protocol_error",
            "Conflicting Transfer-Encoding and Content-Length headers.",
        )
    if len(length_values) > 1:
        raise WebSourceError(
            "http_protocol_error",
            "Repeated Content-Length headers are unsupported.",
        )
    if len(transfer_values) > 1 or (
        transfer_values and transfer_values[0].strip().casefold() != "chunked"
    ):
        raise WebSourceError(
            "http_protocol_error",
            "Unsupported or ambiguous Transfer-Encoding framing.",
        )
    if not length_values:
        return None
    value = length_values[0].strip()
    if len(value) > 20 or _DECIMAL.fullmatch(value) is None:
        raise WebSourceError(
            "http_protocol_error",
            "Content-Length must be one non-negative decimal integer.",
        )
    length = int(value)
    if length > MAX_WEB_BODY_BYTES:
        raise WebSourceError(
            "response_too_large",
            "The encoded webpage response exceeds the byte limit.",
        )
    return length


def _read_encoded_body(
    response: http.client.HTTPResponse,
    connection: http.client.HTTPConnection,
    deadline: HopDeadline,
    content_length: int | None,
) -> bytes:
    body = bytearray()
    try:
        while True:
            _refresh_socket_timeout(connection, deadline, "response_body")
            allowance = MAX_WEB_BODY_BYTES + 1 - len(body)
            if allowance <= 0:
                raise WebSourceError(
                    "response_too_large",
                    "The encoded webpage response exceeds the byte limit.",
                )
            chunk = response.read1(min(_READ_CHUNK_BYTES, allowance))
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > MAX_WEB_BODY_BYTES:
                raise WebSourceError(
                    "response_too_large",
                    "The encoded webpage response exceeds the byte limit.",
                )
    except WebSourceError:
        raise
    except (socket.timeout, TimeoutError) as exc:
        raise _request_timeout("response_body", exc) from exc
    except (http.client.HTTPException, ValueError) as exc:
        try:
            deadline.remaining_time("response_body")
        except WebSourceError as timeout:
            raise timeout from exc
        raise WebSourceError(
            "http_protocol_error",
            "The webpage response body framing is invalid.",
            data=_diagnostic_data("response_body", exc),
        ) from exc
    except (ssl.SSLError, OSError) as exc:
        raise _network_error("response_body", exc) from exc
    if content_length is not None and len(body) != content_length:
        raise WebSourceError(
            "http_protocol_error",
            "The webpage response ended before Content-Length bytes arrived.",
            data={
                "stage": "response_body",
                "detail_type": "IncompleteBodyError",
            },
        )
    return bytes(body)


def _decode_content_encoding(response: WebHttpResponse) -> bytes:
    if len(response.encoded_body) > MAX_WEB_BODY_BYTES:
        raise WebSourceError(
            "response_too_large",
            "The encoded webpage response exceeds the byte limit.",
        )
    values = response.header_values("Content-Encoding")
    if not values:
        return response.encoded_body
    if len(values) != 1:
        raise WebSourceError(
            "unsupported_content_encoding",
            "Multiple Content-Encoding headers are unsupported.",
        )
    encoding = values[0].strip().casefold()
    if encoding == "identity":
        return response.encoded_body
    if "," in encoding or encoding not in {"gzip", "deflate"}:
        raise WebSourceError(
            "unsupported_content_encoding",
            f"Unsupported webpage Content-Encoding: {encoding or 'empty'}.",
        )
    if encoding == "gzip":
        return _bounded_decompress(response.encoded_body, 16 + zlib.MAX_WBITS)
    try:
        return _bounded_decompress(response.encoded_body, zlib.MAX_WBITS)
    except _DeflateFormatError:
        return _bounded_decompress(response.encoded_body, -zlib.MAX_WBITS)


class _DeflateFormatError(Exception):
    pass


def _bounded_decompress(data: bytes, window_bits: int) -> bytes:
    try:
        decompressor = zlib.decompressobj(window_bits)
        output = bytearray()
        for offset in range(0, len(data), _READ_CHUNK_BYTES):
            pending = data[offset : offset + _READ_CHUNK_BYTES]
            while pending:
                remaining = MAX_WEB_BODY_BYTES + 1 - len(output)
                part = decompressor.decompress(pending, remaining)
                output.extend(part)
                if len(output) > MAX_WEB_BODY_BYTES:
                    raise WebSourceError(
                        "response_too_large",
                        "The decoded webpage response exceeds the byte limit.",
                    )
                next_pending = decompressor.unconsumed_tail
                if next_pending == pending:
                    raise WebSourceError(
                        "response_too_large",
                        "The decoded webpage response exceeds the byte limit.",
                    )
                pending = next_pending
        output.extend(
            decompressor.flush(MAX_WEB_BODY_BYTES + 1 - len(output))
        )
        if len(output) > MAX_WEB_BODY_BYTES:
            raise WebSourceError(
                "response_too_large",
                "The decoded webpage response exceeds the byte limit.",
            )
        if not decompressor.eof or decompressor.unused_data:
            raise zlib.error("incomplete or trailing compressed stream")
        return bytes(output)
    except WebSourceError:
        raise
    except zlib.error as exc:
        if window_bits == zlib.MAX_WBITS:
            raise _DeflateFormatError from exc
        raise WebSourceError(
            "malformed_content_encoding",
            "The webpage compressed response is malformed.",
        ) from exc


def _content_type(response: WebHttpResponse) -> tuple[str, str | None]:
    values = response.header_values("Content-Type")
    if len(values) != 1:
        raise WebSourceError(
            "unsupported_content_type",
            "The webpage must provide one supported Content-Type.",
        )
    content_type = values[0].split(";", 1)[0].strip().casefold()
    if content_type not in _SUPPORTED_CONTENT_TYPES:
        raise WebSourceError(
            "unsupported_content_type",
            f"Unsupported webpage Content-Type: {content_type or 'empty'}.",
        )
    message = Message()
    message["content-type"] = values[0]
    return content_type, message.get_content_charset()


def _decode_text(
    body: bytes,
    charset: str | None,
    content_type: str,
) -> str:
    selected = charset
    if selected is None and content_type in _HTML_CONTENT_TYPES:
        prefix = body[:4_096]
        match = _META_CHARSET.search(prefix) or _META_HTTP_EQUIV_CHARSET.search(prefix)
        if match is not None:
            selected = match.group(1).decode("ascii", errors="ignore")
    try:
        return body.decode(selected or "utf-8", errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fragments: list[str] = []
        self.title_fragments: list[str] = []
        self.ignored_depth = 0
        self.head_depth = 0
        self.in_title = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        tag = tag.casefold()
        if self.ignored_depth:
            if tag not in _VOID_TAGS:
                self.ignored_depth += 1
            return
        if tag in _IGNORED_TAGS:
            self.ignored_depth = 1
            return
        if tag == "head":
            self.head_depth += 1
        elif tag == "title":
            self.in_title = True
        elif not self.head_depth:
            if tag in _BLOCK_TAGS or tag in {"br", "tr"}:
                self.fragments.append("\n")

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        was_ignored = self.ignored_depth > 0
        self.handle_starttag(tag, attrs)
        if was_ignored and tag.casefold() in _VOID_TAGS:
            return
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self.ignored_depth:
            self.ignored_depth -= 1
            return
        if tag == "title":
            self.in_title = False
        elif tag == "head":
            self.head_depth = max(0, self.head_depth - 1)
        elif not self.head_depth:
            if tag in _BLOCK_TAGS or tag == "tr":
                self.fragments.append("\n")
            elif tag in {"td", "th"}:
                self.fragments.append("\t")

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        if self.in_title:
            self.title_fragments.append(data)
            return
        if not self.head_depth:
            self.fragments.append(re.sub(r"\s+", " ", data))


def _extract_html(text: str) -> tuple[str, str]:
    parser = _ReadableHTMLParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:
        raise WebSourceError(
            "webpage_text_unavailable",
            "The webpage HTML could not be converted to readable text.",
        ) from exc
    title = re.sub(r"\s+", " ", "".join(parser.title_fragments)).strip()
    title = title[:MAX_WEB_TITLE_CHARS]
    joined = "".join(parser.fragments).replace("\r\n", "\n").replace("\r", "\n")
    joined = re.sub(r" *\t *", "\t", joined)
    joined = re.sub(r" *\n *", "\n", joined)
    lines = [line.strip(" \t") for line in joined.split("\n")]
    body = "\n".join(line for line in lines if line)
    return title, body


def _normalize_plain_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _header_values(
    headers: Iterable[tuple[str, str]],
    name: str,
) -> tuple[str, ...]:
    folded = name.casefold()
    return tuple(value for key, value in headers if key.casefold() == folded)
