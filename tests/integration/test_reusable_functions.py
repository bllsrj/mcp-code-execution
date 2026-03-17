"""Integration tests for the reusable function library end-to-end flow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mce.runtime.cache import CacheStore
from mce.runtime.executor import CodeExecutor

if TYPE_CHECKING:
    from pathlib import Path

    from mce.config import MCEConfig


@pytest.fixture
async def integrated_cache(tmp_path: Path, mce_config: MCEConfig) -> CacheStore:
    """Create cache with real executor integration."""
    cache = CacheStore(
        db_path=str(tmp_path / "integration.db"),
        max_functions=50,
        ttl_days=30,
    )
    await cache.initialize()

    # Create executor and wire it up
    executor = CodeExecutor(mce_config, cache)
    cache.set_executor(executor)

    return cache


async def test_save_and_run_reusable_function(integrated_cache: CacheStore) -> None:
    """End-to-end: save function with validation, then retrieve and check."""
    code = """
# Simple function that returns a constant
result = 42
"""

    func = await integrated_cache.save_reusable_function(
        name="return_constant",
        description="Returns the number 42",
        code=code,
    )

    assert func.name == "return_constant"
    assert func.code == code
    assert func.times_used == 1

    # Retrieve it
    retrieved = await integrated_cache.get_reusable_function("return_constant")
    assert retrieved is not None
    assert retrieved.code == code


async def test_validation_rejects_malicious_code(integrated_cache: CacheStore) -> None:
    """Validation rejects code with banned imports."""
    from mce.errors import CompileError, SecurityViolationError  # noqa: PLC0415

    malicious_code = """
import os
os.system('rm -rf /')
result = "hacked"
"""

    # Should raise due to AST security guard
    with pytest.raises((SecurityViolationError, CompileError)):
        await integrated_cache.save_reusable_function(
            name="malicious",
            description="Bad code",
            code=malicious_code,
        )


async def test_overwrite_updates_function(integrated_cache: CacheStore) -> None:
    """Overwriting function updates code and resets usage."""
    code1 = "result = 1"
    await integrated_cache.save_reusable_function("func", "version 1", code1)

    # Increment usage
    await integrated_cache.increment_usage("func", code1)
    func_v1 = await integrated_cache.get_reusable_function("func")
    assert func_v1 is not None
    assert func_v1.times_used == 2

    # Overwrite
    code2 = "result = 2"
    func_v2 = await integrated_cache.overwrite_reusable_function(
        "func", "version 2", code2, "improved implementation"
    )

    assert func_v2.code == code2
    assert func_v2.times_used == 1  # Reset
    assert func_v2.description == "version 2"


async def test_list_filters_by_instance(integrated_cache: CacheStore) -> None:
    """List functions filtered by instance usage."""
    await integrated_cache.save_reusable_function(
        "prod_func",
        "Prod function",
        'from mirth.functions import get_channels\nresult = get_channels(instance="prod")',
    )

    await integrated_cache.save_reusable_function(
        "staging_func",
        "Staging function",
        'from mirth.functions import get_forecast\nresult = get_forecast(instance="staging")',
    )

    await integrated_cache.save_reusable_function(
        "multi_func",
        "Multi-instance function",
        'from mirth.functions import x\nx(instance="prod")\nx(instance="staging")',
    )

    # Filter by prod only
    prod_funcs = await integrated_cache.list_reusable_functions(instance_filter=["prod"])
    names = {f["name"] for f in prod_funcs}
    assert "prod_func" in names
    assert "multi_func" in names
    assert "staging_func" not in names


async def test_usage_statistics_sorting(integrated_cache: CacheStore) -> None:
    """Functions are sorted by times_used descending."""
    await integrated_cache.save_reusable_function("func_a", "A", "result = 'a'")
    await integrated_cache.save_reusable_function("func_b", "B", "result = 'b'")
    await integrated_cache.save_reusable_function("func_c", "C", "result = 'c'")

    # Increment usage: func_c = 3, func_a = 2, func_b = 1
    await integrated_cache.increment_usage("func_c", "result = 'c'")
    await integrated_cache.increment_usage("func_c", "result = 'c'")
    await integrated_cache.increment_usage("func_a", "result = 'a'")

    functions = await integrated_cache.list_reusable_functions()
    names = [f["name"] for f in functions]

    # Should be sorted by times_used DESC
    assert names[0] == "func_c"  # 3 uses
    assert names[1] == "func_a"  # 2 uses
    assert names[2] == "func_b"  # 1 use


async def test_auto_prune_on_capacity(tmp_path: Path, mce_config: MCEConfig) -> None:
    """Auto-pruning removes least-used functions when enforcing capacity."""
    cache = CacheStore(
        db_path=str(tmp_path / "prune_test.db"),
        max_functions=10,  # High enough to not auto-prune during saves
        ttl_days=30,
    )
    await cache.initialize()
    executor = CodeExecutor(mce_config, cache)
    cache.set_executor(executor)

    # Add 5 functions
    for i in range(5):
        await cache.save_reusable_function(f"func_{i}", f"Function {i}", f"result = {i}")

    # Manually boost usage for func_4
    await cache.increment_usage("func_4", "result = 4")
    await cache.increment_usage("func_4", "result = 4")

    # Manually enforce cap of 3
    deleted = await cache.enforce_global_cap(limit=3)
    assert deleted == 2

    # Should now have 3 functions
    functions = await cache.list_reusable_functions()
    assert len(functions) == 3

    # func_4 should still exist (most used)
    func_4 = await cache.get_reusable_function("func_4")
    assert func_4 is not None
    assert func_4.times_used == 3


async def test_instance_detection_auto_updates(integrated_cache: CacheStore) -> None:
    """Instance detection automatically updates on each execution."""
    # Start with single instance
    code1 = 'from mirth.functions import x\nresult = x(instance="prod")'
    await integrated_cache.save_reusable_function("func", "test", code1)

    func = await integrated_cache.get_reusable_function("func")
    assert func is not None
    assert func.instances_used == ["prod"]

    # Run with additional instance
    code2 = 'from mirth.functions import x\nx(instance="prod")\nx(instance="staging")'
    await integrated_cache.increment_usage("func", code2)

    func = await integrated_cache.get_reusable_function("func")
    assert func is not None
    # Should merge both instances
    assert set(func.instances_used) == {"prod", "staging"}
