# tests/test_auth.py — vlm/api/auth.py X-API-Key 검증 테스트

import pytest
from fastapi import HTTPException

from vlm.api import auth


@pytest.fixture
def with_keys(monkeypatch):
    monkeypatch.setattr(auth, "_API_KEYS", {"key-A", "key-B"})


@pytest.fixture
def without_keys(monkeypatch):
    monkeypatch.setattr(auth, "_API_KEYS", set())


@pytest.mark.asyncio
async def test_no_keys_configured_allows_all(without_keys):
    # 키 미설정 시 인증 비활성화 (개발 모드)
    assert auth.is_auth_enabled() is False
    assert await auth.verify_api_key(x_api_key=None) is True
    assert await auth.verify_api_key(x_api_key="anything") is True


@pytest.mark.asyncio
async def test_valid_key_passes(with_keys):
    assert auth.is_auth_enabled() is True
    assert await auth.verify_api_key(x_api_key="key-A") is True
    assert await auth.verify_api_key(x_api_key="key-B") is True


@pytest.mark.asyncio
async def test_invalid_key_rejected(with_keys):
    with pytest.raises(HTTPException) as exc:
        await auth.verify_api_key(x_api_key="wrong-key")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_missing_header_rejected_when_enabled(with_keys):
    with pytest.raises(HTTPException) as exc:
        await auth.verify_api_key(x_api_key=None)
    assert exc.value.status_code == 401
    assert "필요합니다" in exc.value.detail
