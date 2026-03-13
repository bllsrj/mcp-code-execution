"""MCE FastMCP server — registers the 5 MCP tools and 1 prompt exposed to LLMs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP
from toon_format import encode as _toon_encode

from mce.errors import (
    CacheError,
    CompileError,
    ExecutionError,
    ExecutionTimeoutError,
    FunctionNotFoundError,
    LintError,
    SecurityViolationError,
    ServerNotFoundError,
)
from mce.runtime.cache import CacheStore
from mce.runtime.executor import CodeExecutor
from mce.runtime.registry import Registry
from mce.utils.logging import get_logger

if TYPE_CHECKING:
    from mce.config import MCEConfig

logger = get_logger(__name__)


def create_server(
    config: MCEConfig,
    registry: Registry | None = None,
    cache: CacheStore | None = None,
) -> FastMCP:
    """Create and configure the MCE FastMCP server with all 5 tools and 1 prompt.

    Args:
        config: MCE configuration instance.
        registry: Pre-loaded Registry. If None, a new one is created from config.
        cache: Pre-initialized CacheStore. If None, a new one is created from config.

    Returns:
        Configured FastMCP server ready to run.
    """
    mcp: FastMCP = FastMCP(
        name="MCE — MCP Code Execution",
        instructions="""\
# MCE — MCP Code Execution: Usage Guide

## MANDATORY RULE

**`get_functions` BEFORE writing code** — You MUST call `get_functions` before
using any server function. Never write `from <server>.functions import <fn>`
without first calling `get_functions` in the same session.

## Workflow (follow in order)

1. **`list_servers`** — Discover available API servers and their function names.

2. **`get_functions`** — Fetch the signature, parameters, and return schema for
   1–5 functions at once. The response includes a ready-to-use `import_statement`.

3. **`execute_code`** — Run Python code in a sandboxed Docker container.
   - Use the exact `import_statement` from `get_functions`.
   - Every dynamic value (city, ID, date, name…) MUST be a top-level variable.
   - `main()` takes NO arguments — it reads those top-level variables as globals.
   - NEVER hardcode any entity or value inside `main()`.

4. **Reusable Function Library** — Build a persistent library of useful patterns:
   - **`save_reusable_function`** — Save successful code for cross-session reuse.
     Use semantic names like "deploy_channels_batch" or "get_patient_vitals".
   - **`list_reusable_functions`** — Discover existing functions (sorted by popularity).
   - **`run_reusable_function`** — Execute a saved function with new params.
   - **`overwrite_reusable_function`** — Update an existing function.
   - **`delete_reusable_function`** — Remove obsolete functions.

## Rules

- NEVER guess function signatures. Always call `get_functions` first.
- NEVER import a server module without the `import_statement` from `get_functions`.
- Keep `execute_code` payloads minimal — extract only the fields you need.
- Save useful patterns to the reusable function library for future sessions.
- If execution fails, re-read the `get_functions` output before retrying.
""",
    )

    if registry is None:
        registry = Registry(config.compiled_output_dir)
        registry.load()
    if cache is None:
        cache = CacheStore(
            config.cache_db_path,
            max_functions=config.reusable_function_max_count,
            ttl_days=config.reusable_function_ttl_days,
        )
    executor = CodeExecutor(config, cache)
    cache.set_executor(executor)

    try:
        _sandbox_libraries = [
            line.strip() for line in Path(config.sandbox_requirements_path).read_text().splitlines() if line.strip()
        ]
    except OSError:
        _sandbox_libraries = []

    @mcp.tool()
    async def list_servers() -> str:
        """List all available API servers and their functions.

        Returns a compact overview of each server with:
        - Server name and description
        - List of available functions with one-line summaries

        Use this to discover what APIs are available before getting function details.
        """
        try:
            servers = registry.list_servers()
            logger.info("tool_list_servers_called", server_count=len(servers))
            return str(
                _toon_encode(
                    {
                        "sandbox_libraries": _sandbox_libraries,
                        "servers": [
                            {
                                "name": s.name,
                                "description": s.description,
                                "functions": [
                                    {"name": fn, "summary": s.function_summaries.get(fn, "")} for fn in s.functions
                                ],
                            }
                            for s in servers
                        ],
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("list_servers_unexpected_error")
            return str(_toon_encode({"error": "Internal error loading servers", "detail": str(exc)}))

    @mcp.tool()
    async def get_functions(functions: list[dict[str, str]]) -> str:
        """Get detailed function signatures and return schemas for 1–5 functions at once.

        Args:
            functions: List of 1–5 items, each with:
                - server_name: Name of the server (from list_servers).
                - function_name: Name of the function to inspect.

        Returns each function's parameters, types, and response data structure
        so you can write Python code that calls them correctly.
        Requesting more than 5 functions at once returns a validation error.
        """
        if not functions:
            return str(_toon_encode({"error": "Provide at least 1 function.", "error_type": "validation"}))
        if len(functions) > 5:
            return str(
                _toon_encode({"error": "At most 5 functions can be requested at once.", "error_type": "validation"})
            )

        results = []
        for item in functions:
            server_name = item.get("server_name", "")
            function_name = item.get("function_name", "")
            try:
                fn = registry.get_function(server_name, function_name)
                logger.info("tool_get_function_called", server=server_name, function=function_name)
                results.append(
                    {
                        "server": server_name,
                        "function": fn.function_name,
                        "summary": fn.summary,
                        "method": fn.method,
                        "path": fn.path,
                        "parameters": [p.model_dump() for p in fn.parameters],
                        "return_type": fn.return_type,
                        "response_fields": [r.model_dump() for r in fn.response_fields],
                        "usage_example": fn.source_code,
                        "import_statement": f"from {server_name}.functions import {function_name}",
                    }
                )
            except ServerNotFoundError as exc:
                results.append(
                    {
                        "server": server_name,
                        "function": function_name,
                        "error": str(exc),
                        "error_type": "server_not_found",
                    }
                )
            except FunctionNotFoundError as exc:
                results.append(
                    {
                        "server": server_name,
                        "function": function_name,
                        "error": str(exc),
                        "error_type": "function_not_found",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("get_function_unexpected_error")
                results.append(
                    {
                        "server": server_name,
                        "function": function_name,
                        "error": "Internal error",
                        "error_type": "internal",
                        "detail": str(exc),
                    }
                )
        return str(_toon_encode({"functions": results}))

    @mcp.tool()
    async def execute_code(code: str, description: str) -> dict[str, Any]:
        """Execute Python code in a sandboxed environment.

        The code runs in an isolated Docker container with access to API server functions.
        Code MUST define either a `main()` function that returns a result,
        or a `result` variable containing the output.

        Available imports in sandbox:
        - Server functions: `from {server_name}.functions import {function_name}`
        - Standard: httpx, json, datetime, re, math, dataclasses, typing, collections

        Args:
            code: Valid Python code to execute. Must be self-contained.
            description: Brief description of what this code does (used for caching).

        Run multiple functions in single code block and return result.
        Returns execution result with data or error details.
        Keep responses minimal — extract only the fields you need.

        ## Reusable Code Guide

        WRONG — hardcoded value inside main(), not reusable:
            def main():
                return geocoding_search(name="Colombo, Sri Lanka")  # BAD

        CORRECT — top-level variable, reusable via save_reusable_function:
            location_name = "Colombo, Sri Lanka"   # top-level param

            def main():
                return geocoding_search(name=location_name)  # reads global

            result = main()

        After execute_code succeeds, consider saving useful patterns with:
            save_reusable_function(
                name="get_location_coordinates",
                description="Get lat/lon for a location name",
                code=<your code>
            )

        Then reuse in future sessions with:
            run_reusable_function(name="get_location_coordinates", params={"location_name": "Galle"})

        Rules:
        - ALL dynamic values (city, ID, date, name…) → top-level variables
        - main() NEVER takes arguments; it reads globals only
        - description: "action + entity + key param", no specific values or dates
        """
        try:
            result = await executor.execute(code, description)
            logger.info("tool_execute_code_called", success=result.success, description=description[:60])
            response = result.model_dump()
            # Suggest saving as reusable function on success
            if result.success:
                response["suggestion"] = (
                    "Consider saving this as a reusable function with save_reusable_function() "
                    "if it could be useful in future sessions."
                )
            return response
        except SecurityViolationError as exc:
            return {"success": False, "error": f"Security violation: {exc}", "error_type": "security"}
        except LintError as exc:
            return {
                "success": False,
                "error": f"Code has issues: {exc}",
                "lint_output": exc.lint_output,
                "error_type": "lint",
            }
        except ExecutionTimeoutError:
            return {
                "success": False,
                "error": f"Execution timed out after {config.execution_timeout_seconds}s",
                "error_type": "timeout",
            }
        except ExecutionError as exc:
            return {"success": False, "error": str(exc), "stderr": exc.stderr, "error_type": "execution"}
        except Exception:  # noqa: BLE001
            logger.exception("execute_code_unexpected_error")
            return {"success": False, "error": "Internal error occurred", "error_type": "internal"}

    @mcp.tool()
    async def save_reusable_function(name: str, description: str, code: str) -> dict[str, Any]:
        """Save a new reusable function to the persistent library.

        Validates code, auto-detects which servers it uses, and stores it for future reuse.
        Functions must have semantic names like "deploy_channels_batch" or "get_patient_latest_vitals".

        Args:
            name: Unique semantic name (e.g., "deploy_channels_batch").
            description: What this function does and when to use it.
            code: Complete Python code (must pass validation).

        Returns:
            Success response with function metadata, or error details.
        """
        try:
            func = await cache.save_reusable_function(name, description, code)
            logger.info("tool_save_reusable_function_called", name=name)
            return {
                "success": True,
                "function": {
                    "name": func.name,
                    "description": func.description,
                    "servers_used": func.servers_used,
                    "times_used": func.times_used,
                },
            }
        except ValueError as exc:
            return {"success": False, "error": str(exc), "error_type": "validation"}
        except CompileError as exc:
            return {"success": False, "error": str(exc), "error_type": "validation"}
        except CacheError as exc:
            return {"success": False, "error": f"Cache error: {exc}", "error_type": "cache"}
        except Exception:  # noqa: BLE001
            logger.exception("save_reusable_function_unexpected_error")
            return {"success": False, "error": "Internal error occurred", "error_type": "internal"}

    @mcp.tool()
    async def overwrite_reusable_function(name: str, description: str, code: str, reason: str) -> dict[str, Any]:
        """Overwrite an existing reusable function with a new implementation.

        Resets usage statistics and updates code/description. Use when improving
        an existing function or fixing bugs.

        Args:
            name: Name of existing function to overwrite.
            description: Updated description.
            code: New implementation (must pass validation).
            reason: Why overwriting (e.g., "fixed parameter handling", "added error check").

        Returns:
            Success response with updated function metadata, or error details.
        """
        try:
            func = await cache.overwrite_reusable_function(name, description, code, reason)
            logger.info("tool_overwrite_reusable_function_called", name=name, reason=reason)
            return {
                "success": True,
                "function": {
                    "name": func.name,
                    "description": func.description,
                    "servers_used": func.servers_used,
                    "times_used": func.times_used,
                },
            }
        except ValueError as exc:
            return {"success": False, "error": str(exc), "error_type": "validation"}
        except CompileError as exc:
            return {"success": False, "error": str(exc), "error_type": "validation"}
        except CacheError as exc:
            return {"success": False, "error": f"Cache error: {exc}", "error_type": "cache"}
        except Exception:  # noqa: BLE001
            logger.exception("overwrite_reusable_function_unexpected_error")
            return {"success": False, "error": "Internal error occurred", "error_type": "internal"}

    @mcp.tool()
    async def list_reusable_functions(server_filter: list[str] | None = None) -> str:
        """List all reusable functions in the library, optionally filtered by server(s).

        Returns functions sorted by popularity (times_used DESC). For single-server
        filter, omits servers_used field since it's redundant.

        Args:
            server_filter: Optional list of server names to filter by (e.g., ["mirth"]).

        Returns:
            List of functions with name, description, times_used, last_used, servers_used.
        """
        try:
            functions = await cache.list_reusable_functions(server_filter)
            logger.info("tool_list_reusable_functions_called", count=len(functions), filter=server_filter)
            return str(_toon_encode({"functions": functions}))
        except CacheError as exc:
            return str(_toon_encode({"error": f"Cache error: {exc}", "error_type": "cache"}))
        except Exception:  # noqa: BLE001
            logger.exception("list_reusable_functions_unexpected_error")
            return str(_toon_encode({"error": "Internal error occurred", "error_type": "internal"}))

    @mcp.tool()
    async def run_reusable_function(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a reusable function from the library with optional parameter overrides.

        Fetches function code, increments usage counter, and executes it with injected params.

        Args:
            name: Name of the reusable function to run.
            params: Optional key→value overrides injected as top-level variables.

        Returns:
            Same structure as execute_code: success, data, error, execution_time_ms.
        """
        try:
            func = await cache.get_reusable_function(name)
        except CacheError as exc:
            return {"success": False, "error": f"Cache error: {exc}", "error_type": "cache"}

        if func is None:
            return {
                "success": False,
                "error": f"Reusable function '{name}' not found",
                "error_type": "not_found",
            }

        code = func.code
        if params:
            # Inject params as top-level variables
            param_lines = "\n".join(f"{k} = {v!r}" for k, v in params.items())
            rerun = "try:\n    result = main()\nexcept NameError:\n    pass"
            code = f"{code}\n\n# --- injected parameter overrides ---\n_params = {params!r}\n{param_lines}\n{rerun}\n"

        logger.info("tool_run_reusable_function_called", name=name, has_params=bool(params))

        try:
            result = await executor.execute(code, func.description)
            # Increment usage counter on success
            if result.success:
                await cache.increment_usage(name, func.code)
            return result.model_dump()
        except SecurityViolationError as exc:
            return {"success": False, "error": f"Security violation: {exc}", "error_type": "security"}
        except LintError as exc:
            return {
                "success": False,
                "error": f"Code has issues: {exc}",
                "lint_output": exc.lint_output,
                "error_type": "lint",
            }
        except ExecutionTimeoutError:
            return {
                "success": False,
                "error": f"Execution timed out after {config.execution_timeout_seconds}s",
                "error_type": "timeout",
            }
        except ExecutionError as exc:
            return {"success": False, "error": str(exc), "stderr": exc.stderr, "error_type": "execution"}
        except Exception:  # noqa: BLE001
            logger.exception("run_reusable_function_unexpected_error")
            return {"success": False, "error": "Internal error occurred", "error_type": "internal"}

    @mcp.tool()
    async def delete_reusable_function(name: str, reason: str) -> dict[str, Any]:
        """Delete a reusable function from the library.

        Args:
            name: Name of the function to delete.
            reason: Why deleting (e.g., "obsolete", "buggy", "replaced by X").

        Returns:
            Success status.
        """
        try:
            deleted = await cache.delete_reusable_function(name, reason)
            logger.info("tool_delete_reusable_function_called", name=name, reason=reason)
            if deleted:
                return {"success": True, "message": f"Function '{name}' deleted"}
            else:
                return {"success": False, "error": f"Function '{name}' not found", "error_type": "not_found"}
        except CacheError as exc:
            return {"success": False, "error": f"Cache error: {exc}", "error_type": "cache"}
        except Exception:  # noqa: BLE001
            logger.exception("delete_reusable_function_unexpected_error")
            return {"success": False, "error": "Internal error occurred", "error_type": "internal"}

    @mcp.prompt()
    def reusable_code_guide() -> str:
        """Guide for writing reusable functions for the persistent library."""
        return (
            "# Reusable Function Library Guide\n\n"
            "Save useful code patterns as named functions for cross-session reuse.\n\n"
            "## Writing Reusable Functions\n\n"
            "WRONG — hardcoded values inside main():\n"
            "    def main(): return fn(name='Colombo')  # BAD - not reusable\n\n"
            "CORRECT — top-level variables for parameterization:\n"
            "    location_name = 'Colombo'   # top-level param\n"
            "    def main():\n"
            "        return fn(name=location_name)  # reads global\n"
            "    result = main()\n\n"
            "## Workflow\n\n"
            "1. Write and test code with execute_code()\n"
            "2. If useful for future, save with save_reusable_function(\n"
            "       name='get_location_coordinates',\n"
            "       description='Get lat/lon for a location name',\n"
            "       code=<your code>\n"
            "   )\n"
            "3. Reuse with run_reusable_function(\n"
            "       name='get_location_coordinates',\n"
            "       params={'location_name': 'Galle'}\n"
            "   )\n\n"
            "## Naming\n\n"
            "Use semantic names describing the use case:\n"
            "- deploy_channels_batch (good)\n"
            "- get_patient_latest_vitals (good)\n"
            "- code_abc123 (bad)\n\n"
            "Rules:\n"
            "- ALL dynamic values → top-level variables\n"
            "- main() NEVER takes arguments\n"
            "- description: describe the use case, not specific values"
        )

    return mcp


async def initialize_server(config: MCEConfig, mcp: FastMCP) -> None:
    """Run startup initialization: load registry, initialize cache, and auto-prune.

    Args:
        config: MCE configuration.
        mcp: FastMCP server instance (used to access registry/cache via closure — handled elsewhere).
    """
    # Registry and cache are loaded via the create_server closure
    # This function can be used for pre-flight checks
    registry = Registry(config.compiled_output_dir)
    registry.load()

    cache = CacheStore(
        config.cache_db_path,
        max_functions=config.reusable_function_max_count,
        ttl_days=config.reusable_function_ttl_days,
    )
    await cache.initialize()

    # Auto-prune old and excess functions on startup
    prune_result = await cache.auto_prune()
    if prune_result["old_deleted"] > 0 or prune_result["cap_deleted"] > 0:
        logger.info(
            "startup_auto_prune_completed",
            old_deleted=prune_result["old_deleted"],
            cap_deleted=prune_result["cap_deleted"],
        )

    logger.info(
        "mce_server_initialized",
        compiled_dir=config.compiled_output_dir,
        cache_db=config.cache_db_path,
        log_level=config.log_level,
    )
