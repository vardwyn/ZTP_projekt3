import hashlib
import io
import zipfile

import pandas as pd
import pytest
import requests

from data_preparation import download_gios_archive


# Pseudo-wrapper dla zastąpienia requests.get i requests.raise_for_status w testach
class DummyResponse:
    def __init__(self, content=b"", raise_exc=None, headers=None):
        self.content = content
        self._raise_exc = raise_exc
        self.headers = headers or {}

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc


# Tworzy poprawne archiwum zip używając zipfile
def make_zip_bytes(file_map):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in file_map.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_download_success_with_hash(monkeypatch):
    content = make_zip_bytes({"data.xlsx": b"dummy"})
    expected_hash = hashlib.sha256(content).hexdigest()

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "application/zip"})

    def fake_read_excel(f, header=None, **kwargs):
        assert header is None
        assert hasattr(f, "read")
        assert kwargs.get("dtype") == {0: str}
        return pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    df = download_gios_archive("https://example.com/archive.zip", "data.xlsx", sha256=expected_hash)
    assert df.equals(pd.DataFrame({"ok": [1]}))


def test_download_success_without_hash(monkeypatch):
    content = make_zip_bytes({"data.xlsx": b"dummy"})

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "application/zip"})

    def fake_read_excel(f, header=None, **kwargs):
        assert kwargs.get("dtype") == {0: str}
        return pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    df = download_gios_archive("https://example.com/archive.zip", "data.xlsx", sha256=None)
    assert df.equals(pd.DataFrame({"ok": [1]}))


def test_bad_hash_raises_value_error(monkeypatch):
    content = make_zip_bytes({"data.xlsx": b"dummy"})

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "application/zip"})

    def fake_read_excel(f, header=None):
        return pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    with pytest.raises(ValueError):
        download_gios_archive("https://example.com/archive.zip", "data.xlsx", sha256="deadbeef")


def test_missing_file_in_zip_raises_file_not_found(monkeypatch):
    content = make_zip_bytes({"other.xlsx": b"dummy"})

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "application/zip"})

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(FileNotFoundError):
        download_gios_archive("https://example.com/archive.zip", "data.xlsx", sha256=None)


def test_non_zip_content_raises_bad_zip(monkeypatch):
    content = b"this is not a zip"

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "application/zip"})

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(zipfile.BadZipFile):
        download_gios_archive("https://example.com/archive.zip", "data.xlsx", sha256=None)


def test_excel_read_failure_is_wrapped(monkeypatch):
    content = make_zip_bytes({"data.xlsx": b"dummy"})

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "application/zip"})

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(RuntimeError):
        download_gios_archive("https://example.com/archive.zip", "data.xlsx", sha256=None)


def test_http_error_propagates(monkeypatch):
    def fake_get(url):
        return DummyResponse(raise_exc=requests.HTTPError("404"))

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(requests.HTTPError):
        download_gios_archive("https://example.com/archive.zip", "data.xlsx", sha256=None)


def test_request_exception_propagates(monkeypatch):
    def fake_get(url):
        raise requests.exceptions.RequestException("network error")

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(requests.exceptions.RequestException):
        download_gios_archive("https://example.com/archive.zip", "data.xlsx", sha256=None)


def test_invalid_url_raises_missing_schema():
    with pytest.raises(requests.exceptions.MissingSchema):
        download_gios_archive("not-a-url", "data.xlsx", sha256=None)


def test_text_html_content_type_raises_value_error(monkeypatch):
    content = b"<html>not a zip</html>"

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "text/html; charset=utf-8"})

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(ValueError):
        download_gios_archive("https://example.com/archive.zip", "data.xlsx", sha256=None)
