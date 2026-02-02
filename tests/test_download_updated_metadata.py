import hashlib

import pandas as pd
import pytest
import requests

from pm25.data_preparation import download_updated_metadata


class DummyResponse:
    def __init__(self, content=b"", raise_exc=None, headers=None):
        self.content = content
        self._raise_exc = raise_exc
        self.headers = headers or {}

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc


def test_download_success_with_hash(monkeypatch):
    content = b"fake-xlsx-bytes"
    expected_hash = hashlib.sha256(content).hexdigest()

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})

    def fake_read_excel(buf):
        return pd.DataFrame({"Kod stacji": ["X1"], "Stary Kod stacji \n(o ile inny od aktualnego)": ["Y1"]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    df = download_updated_metadata("https://example.com/meta.xlsx", sha256=expected_hash)
    assert "Stary Kod stacji" in df.columns
    assert df["Kod stacji"].iloc[0] == "X1"


def test_download_success_without_hash(monkeypatch):
    content = b"fake-xlsx-bytes"

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})

    def fake_read_excel(buf):
        return pd.DataFrame({"Kod stacji": ["X1"]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    df = download_updated_metadata("https://example.com/meta.xlsx", sha256=None)
    assert df["Kod stacji"].iloc[0] == "X1"


def test_bad_hash_raises_value_error(monkeypatch):
    content = b"fake-xlsx-bytes"

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})

    def fake_read_excel(buf):
        return pd.DataFrame({"Kod stacji": ["X1"]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    with pytest.raises(ValueError):
        download_updated_metadata("https://example.com/meta.xlsx", sha256="deadbeef")

def test_excel_read_failure_is_wrapped(monkeypatch):
    content = b"fake-xlsx-bytes"

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})

    def fake_read_excel(buf):
        raise exc_type("read failed")

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    with pytest.raises(RuntimeError):
        download_updated_metadata("https://example.com/meta.xlsx", sha256=None)


def test_text_html_content_type_raises_value_error(monkeypatch):
    content = b"<html>not an xlsx</html>"

    def fake_get(url):
        return DummyResponse(content=content, headers={"Content-Type": "text/html; charset=utf-8"})

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(ValueError):
        download_updated_metadata("https://example.com/meta.xlsx", sha256=None)


def test_http_error_propagates(monkeypatch):
    def fake_get(url):
        return DummyResponse(raise_exc=requests.HTTPError("404"))

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(requests.HTTPError):
        download_updated_metadata("https://example.com/meta.xlsx", sha256=None)


def test_request_exception_propagates(monkeypatch):
    def fake_get(url):
        raise requests.exceptions.RequestException("network error")

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(requests.exceptions.RequestException):
        download_updated_metadata("https://example.com/meta.xlsx", sha256=None)


def test_invalid_url_raises_missing_schema():
    with pytest.raises(requests.exceptions.MissingSchema):
        download_updated_metadata("not-a-url", sha256=None)
