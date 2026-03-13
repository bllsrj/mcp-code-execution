"""Unit tests for the reusable function library."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from mce.errors import CompileError
from mce.runtime.cache import CacheStore, _detect_servers_used, _human_readable_time

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
async def cache(tmp_path: Path) -> CacheStore:
    """Create a test cache store."""
    store = CacheStore(
        db_path=str(tmp_path / "test_functions.db"),
        max_functions=10,
        ttl_days=30,
    )
    await store.initialize()
    # Set a mock executor
    mock_executor = AsyncMock()
    mock_executor.validate_code_only = AsyncMock(return_value=(True, None))
    store.set_executor(mock_executor)
    return store


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_detect_servers_used_single_import() -> None:
    """Detect single server from import statement."""
    code = "from mirth.functions import get_channels\nresult = get_channels()"
    assert _detect_servers_used(code) == ["mirth"]


def test_detect_servers_used_multiple_imports() -> None:
    """Detect multiple servers from different imports."""
    code = """
from mirth.functions import get_channels
from weather.functions import get_forecast
result = get_channels() + get_forecast()
"""
    assert _detect_servers_used(code) == ["mirth", "weather"]


def test_detect_servers_used_import_module_style() -> None:
    """Detect servers from 'import server.functions' style."""
    code = "import mirth.functions\nresult = mirth.functions.get_channels()"
    assert _detect_servers_used(code) == ["mirth"]


def test_detect_servers_used_no_imports() -> None:
    """No servers detected when no imports present."""
    code = "result = 42"
    assert _detect_servers_used(code) == []


def test_human_readable_time_just_now() -> None:
    """Recent timestamp shows as 'just now'."""
    now = time.time()
    assert _human_readable_time(now) == "just now"


def test_human_readable_time_minutes() -> None:
    """Timestamp from minutes ago."""
    two_min_ago = time.time() - 120
    assert "2 minutes ago" in _human_readable_time(two_min_ago)


def test_human_readable_time_hours() -> None:
    """Timestamp from hours ago."""
    two_hours_ago = time.time() - 7200
    assert "2 hours ago" in _human_readable_time(two_hours_ago)


def test_human_readable_time_days() -> None:
    """Timestamp from days ago."""
    three_days_ago = time.time() - (3 * 86400)
    assert "3 days ago" in _human_readable_time(three_days_ago)


# ---------------------------------------------------------------------------
# CacheStore methods
# ---------------------------------------------------------------------------


async def test_save_reusable_function(cache: CacheStore) -> None:
    """Save new function successfully."""
    code = "from mirth.functions import get_channels\nresult = get_channels()"
    func = await cache.save_reusable_function(
        name="list_all_channels",
        description="List all Mirth channels",
        code=code,
    )

    assert func.name == "list_all_channels"
    assert func.description == "List all Mirth channels"
    assert func.code == code
    assert func.servers_used == ["mirth"]
    assert func.times_used == 1


async def test_save_duplicate_name_raises(cache: CacheStore) -> None:
    """Saving function with existing name raises ValueError."""
    code = "result = 1"
    await cache.save_reusable_function("duplicate", "first", code)

    with pytest.raises(ValueError, match="already exists"):
        await cache.save_reusable_function("duplicate", "second", code)


async def test_save_invalid_code_raises(cache: CacheStore) -> None:
    """Saving invalid code raises CompileError."""
    # Override the mock to return validation error
    cache._executor.validate_code_only = AsyncMock(return_value=(False, "Syntax error"))

    with pytest.raises(CompileError, match="Code validation failed"):
        await cache.save_reusable_function("bad", "invalid code", "bad syntax!!")


async def test_get_reusable_function(cache: CacheStore) -> None:
    """Retrieve function by name."""
    code = "result = 42"
    await cache.save_reusable_function("test_func", "test", code)

    func = await cache.get_reusable_function("test_func")
    assert func is not None
    assert func.name == "test_func"
    assert func.code == code


async def test_get_nonexistent_returns_none(cache: CacheStore) -> None:
    """Getting non-existent function returns None."""
    result = await cache.get_reusable_function("nonexistent")
    assert result is None


async def test_overwrite_reusable_function(cache: CacheStore) -> None:
    """Overwrite existing function with new implementation."""
    code1 = "result = 1"
    await cache.save_reusable_function("func", "original", code1)

    code2 = "result = 2"
    func = await cache.overwrite_reusable_function("func", "updated", code2, "improved logic")

    assert func.code == code2
    assert func.description == "updated"
    assert func.times_used == 1  # Reset to 1


async def test_overwrite_nonexistent_raises(cache: CacheStore) -> None:
    """Overwriting non-existent function raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        await cache.overwrite_reusable_function("missing", "desc", "code", "reason")


async def test_list_reusable_functions(cache: CacheStore) -> None:
    """List all functions sorted by times_used."""
    await cache.save_reusable_function("func1", "first", "result = 1")
    await cache.save_reusable_function("func2", "second", "result = 2")

    # Manually increment usage for func2
    await cache.increment_usage("func2", "result = 2")

    functions = await cache.list_reusable_functions()
    assert len(functions) == 2
    # func2 should be first (higher times_used)
    assert functions[0]["name"] == "func2"
    assert functions[0]["times_used"] == 2
    assert functions[1]["name"] == "func1"


async def test_list_reusable_functions_with_filter(cache: CacheStore) -> None:
    """List functions filtered by server."""
    await cache.save_reusable_function("m1", "mirth 1", "from mirth.functions import x")
    await cache.save_reusable_function("w1", "weather 1", "from weather.functions import y")
    await cache.save_reusable_function("m2", "mirth 2", "from mirth.functions import z")

    functions = await cache.list_reusable_functions(server_filter=["mirth"])
    assert len(functions) == 2
    names = {f["name"] for f in functions}
    assert names == {"m1", "m2"}


async def test_list_single_server_filter_omits_servers_used(cache: CacheStore) -> None:
    """Single server filter omits servers_used field."""
    await cache.save_reusable_function("func", "test", "from mirth.functions import x")

    functions = await cache.list_reusable_functions(server_filter=["mirth"])
    assert len(functions) == 1
    assert "servers_used" not in functions[0]


async def test_list_multiple_server_filter_includes_servers_used(cache: CacheStore) -> None:
    """Multiple server filter includes servers_used field."""
    await cache.save_reusable_function("func", "test", "from mirth.functions import x")

    functions = await cache.list_reusable_functions(server_filter=["mirth", "weather"])
    assert len(functions) == 1
    assert "servers_used" in functions[0]


async def test_increment_usage(cache: CacheStore) -> None:
    """Increment usage counter and update last_used_at."""
    code = "result = 1"
    await cache.save_reusable_function("func", "test", code)

    await cache.increment_usage("func", code)

    func = await cache.get_reusable_function("func")
    assert func is not None
    assert func.times_used == 2


async def test_increment_usage_merges_servers(cache: CacheStore) -> None:
    """Increment usage merges new servers with existing."""
    code1 = "from mirth.functions import x"
    await cache.save_reusable_function("func", "test", code1)

    code2 = "from mirth.functions import x\nfrom weather.functions import y"
    await cache.increment_usage("func", code2)

    func = await cache.get_reusable_function("func")
    assert func is not None
    assert set(func.servers_used) == {"mirth", "weather"}


async def test_delete_reusable_function(cache: CacheStore) -> None:
    """Delete function by name."""
    await cache.save_reusable_function("func", "test", "result = 1")

    deleted = await cache.delete_reusable_function("func", "obsolete")
    assert deleted is True

    func = await cache.get_reusable_function("func")
    assert func is None


async def test_delete_nonexistent_returns_false(cache: CacheStore) -> None:
    """Deleting non-existent function returns False."""
    deleted = await cache.delete_reusable_function("missing", "reason")
    assert deleted is False


async def test_prune_old_functions(tmp_path: Path) -> None:
    """Prune functions unused for ttl_days."""
    cache = CacheStore(
        db_path=str(tmp_path / "prune_cache.db"),
        max_functions=50,
        ttl_days=0,  # 0 days = immediate expiry
    )
    await cache.initialize()
    mock_executor = AsyncMock()
    mock_executor.validate_code_only = AsyncMock(return_value=(True, None))
    cache.set_executor(mock_executor)

    await cache.save_reusable_function("old", "old func", "result = 1")
    time.sleep(0.1)  # Ensure some time passes

    count = await cache.prune_old_functions(ttl_days=0)
    assert count == 1

    func = await cache.get_reusable_function("old")
    assert func is None


async def test_enforce_global_cap(tmp_path: Path) -> None:
    """Enforce global cap by deleting least-used functions."""
    cache = CacheStore(
        db_path=str(tmp_path / "cap_cache.db"),
        max_functions=10,  # High cap so save doesn't trigger pruning
        ttl_days=30,
    )
    await cache.initialize()
    mock_executor = AsyncMock()
    mock_executor.validate_code_only = AsyncMock(return_value=(True, None))
    cache.set_executor(mock_executor)

    # Add 5 functions
    for i in range(5):
        await cache.save_reusable_function(f"func{i}", f"test {i}", f"result = {i}")

    # Manually increment usage for func4 to make it most used
    await cache.increment_usage("func4", "result = 4")

    # Enforce cap of 3
    deleted = await cache.enforce_global_cap(limit=3)
    assert deleted == 2

    # func4 should still exist (most used)
    func4 = await cache.get_reusable_function("func4")
    assert func4 is not None


async def test_auto_prune(tmp_path: Path) -> None:
    """Auto-prune runs both age and capacity strategies."""
    cache = CacheStore(
        db_path=str(tmp_path / "auto_prune.db"),
        max_functions=2,
        ttl_days=0,
    )
    await cache.initialize()
    mock_executor = AsyncMock()
    mock_executor.validate_code_only = AsyncMock(return_value=(True, None))
    cache.set_executor(mock_executor)

    # Add 3 functions
    for i in range(3):
        await cache.save_reusable_function(f"func{i}", f"test {i}", f"result = {i}")

    time.sleep(0.1)

    result = await cache.auto_prune(ttl_days=0, cap=2)
    # Should prune at least 1 (either by age or cap)
    assert (result["old_deleted"] + result["cap_deleted"]) >= 1


async def test_save_enforces_cap_before_saving(tmp_path: Path) -> None:
    """Cap enforcement can be triggered manually when needed."""
    cache = CacheStore(
        db_path=str(tmp_path / "save_cap.db"),
        max_functions=2,
        ttl_days=30,
    )
    await cache.initialize()
    mock_executor = AsyncMock()
    mock_executor.validate_code_only = AsyncMock(return_value=(True, None))
    cache.set_executor(mock_executor)

    # Add 3 functions (exceeds cap of 2)
    await cache.save_reusable_function("func1", "test 1", "result = 1")
    await cache.save_reusable_function("func2", "test 2", "result = 2")
    await cache.save_reusable_function("func3", "test 3", "result = 3")

    # Manually enforce cap
    deleted = await cache.enforce_global_cap()
    assert deleted == 1

    # Should now have exactly 2 functions
    functions = await cache.list_reusable_functions()
    assert len(functions) == 2
