"""MCE FastMCP server — registers the 5 MCP tools and 1 prompt exposed to LLMs."""

from __future__ import annotations

import importlib.util
import sys
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


def _load_top_level_tools(compiled_dir: str | Path) -> list[dict[str, Any]]:
    """Scan the compiled directory for ``top_level_functions.py`` files and load them.

    Each file exposes a ``_TOP_LEVEL_TOOLS`` list of ``{"name", "fn", "server"}``
    dicts that ``create_server`` uses to register direct FastMCP tools.

    The compiled directory is added to ``sys.path`` (once) so the generated
    files can import their sibling ``functions.py`` modules.

    Args:
        compiled_dir: Path to the compiled output directory.

    Returns:
        List of tool descriptor dicts ready to register with FastMCP.
    """
    compiled_path = Path(compiled_dir)
    tools: list[dict[str, Any]] = []

    tlf_paths = sorted(compiled_path.glob("*/top_level_functions.py"))
    if not tlf_paths:
        return tools

    # Make compiled dir importable so `from <server>.functions import …` works
    compiled_str = str(compiled_path.resolve())
    if compiled_str not in sys.path:
        sys.path.insert(0, compiled_str)

    for tlf_path in tlf_paths:
        server_name = tlf_path.parent.name
        module_key = f"_mce_tlf_{server_name}"
        try:
            spec = importlib.util.spec_from_file_location(module_key, tlf_path)
            if spec is None or spec.loader is None:
                logger.warning("top_level_functions_spec_invalid", path=str(tlf_path))
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            tool_list: list[dict[str, Any]] = getattr(module, "_TOP_LEVEL_TOOLS", [])
            tools.extend(tool_list)
            logger.info("top_level_tools_loaded", server=server_name, count=len(tool_list))
        except Exception as exc:  # noqa: BLE001
            logger.warning("top_level_tools_load_failed", server=server_name, error=str(exc))

    return tools


_BASE_INSTRUCTIONS = """\
# MCE — MCP Code Execution: Usage Guide

## MANDATORY RULE

**`get_functions` BEFORE writing code** — You MUST call `get_functions` to discover
available endpoints before using any server function.

## Workflow (follow in order)

1. **`list_servers`** — Discover available API servers and their instances.
   - Returns server names, descriptions, and instance information.
   - Note which instances are read-only to avoid attempting destructive operations.

2. **`get_functions`** — Get list of available functions for a server.
   - Pass `server_name` and `read_only` boolean.
   - If `read_only=True`, returns only GET methods (safe for read-only instances).
   - If `read_only=False`, returns only write methods (POST, PUT, DELETE, PATCH).
   - Returns function names, summaries, methods, and paths.

3. **`get_function_details`** — Get detailed signature for a specific function.
   - Pass `server_name` and `function_name`.
   - Returns parameters, types, response schema, usage example, and import statement.
   - Use this to understand how to call specific functions before writing code.

4. **`execute_code`** — Run Python code in a sandboxed Docker container.
   - Use the exact `import_statement` from `get_function_details`.
   - For multi-instance servers, pass `instance="instance_name"` as a keyword parameter
     (e.g., `get_channels(instance="prod")` or `deploy_channel(instance="staging", ...)`).
   - Every dynamic value (city, ID, date, name…) MUST be a top-level variable.
   - `main()` takes NO arguments — it reads those top-level variables as globals.
   - NEVER hardcode any entity or value inside `main()`.

5. **Reusable Function Library** — Build a persistent library of useful patterns:
   - **`save_reusable_function`** — Save successful code for cross-session reuse.
     Use semantic names like "deploy_channels_batch" or "get_patient_vitals".
   - **`list_reusable_functions`** — Discover existing functions (sorted by popularity).
   - **`run_reusable_function`** — Execute a saved function with new params.
   - **`overwrite_reusable_function`** — Update an existing function.
   - **`delete_reusable_function`** — Remove obsolete functions.

## Rules

- NEVER guess function signatures. Always discover with `get_functions` first.
- Call `get_function_details` for any function you plan to use to get the exact signature.
- Keep `execute_code` payloads minimal — extract only the fields you need.
- Save useful patterns to the reusable function library for future sessions.
- **API responses are automatically parsed** — XML responses are converted to Python dicts for easy navigation.

## Multi-Instance Server Considerations

When working with multi-instance servers (e.g., Mirth across prod/staging/dev):

- **Instance-specific data**: Channels, channel IDs, configuration objects, and other
  entities may differ between instances. A channel ID from prod will NOT work on staging.

- **Read-only enforcement**: Some instances (typically production) are marked read-only.
  Attempts to modify data on read-only instances will fail at runtime with a clear error.
  Use `get_functions` with `read_only=True` to see only safe operations.

- **Reusable functions are shared**: The reusable function library is shared across ALL
  instances. When a saved function calls server APIs, you can pass different instance
  names to target different environments with the same workflow.
"""


def _build_instructions(
    registry: Registry,
    servers_with_skills: list[str],
    top_level_tools: list[dict[str, Any]] | None = None,
) -> str:
    """Build the FastMCP instructions string.

    Skills content for each entry in ``servers_with_skills`` is embedded inline
    so the LLM receives it automatically via the MCP ``initialize`` response —
    no explicit resource fetch required.  When the list is empty the base
    instructions are returned as-is, spending zero extra tokens.

    Args:
        registry: Registry used to resolve each server's skills file path.
        servers_with_skills: Pre-computed list of server module names that have
            a ``skills.md`` on disk.  Computed once in ``create_server`` so this
            function never calls ``registry.list_servers()`` itself.
    """
    # Prepend a direct-tools section when top-level tools are registered so the
    # LLM knows it can call them immediately — no workflow required.
    direct_section = ""
    if top_level_tools:
        lines = [
            "\n\n## Direct API Tools\n\n"
            "The following API functions are registered as **direct MCP tools** "
            "and can be called immediately — no `list_servers` → `get_functions` "
            "→ `execute_code` workflow is needed.\n"
        ]
        # Group by server for readability
        by_server: dict[str, list[str]] = {}
        for entry in top_level_tools:
            srv = entry.get("server", "unknown")
            by_server.setdefault(srv, []).append(entry["name"])
        for srv, names in by_server.items():
            lines.append(f"\n**`{srv}`**: " + ", ".join(f"`{n}`" for n in names))
        direct_section = "".join(lines) + "\n"

    if not servers_with_skills:
        return _BASE_INSTRUCTIONS + direct_section

    skills_blocks: list[str] = []
    for sn in servers_with_skills:
        path = registry.skills_path(sn)
        if path is not None:
            skills_blocks.append(f"### `{sn}`\n\n{path.read_text(encoding='utf-8')}")

    if not skills_blocks:
        return _BASE_INSTRUCTIONS + direct_section

    divider = "\n\n---\n\n"
    skills_section = (
        "\n\n## Server Skills\n\n"
        "The following server-specific guides are pre-loaded. "
        "Apply their guidance whenever you use that server's tools.\n\n" + divider.join(skills_blocks) + "\n"
    )
    return _BASE_INSTRUCTIONS + direct_section + skills_section


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
    # Initialise registry before FastMCP so we can inspect skills availability
    # and tailor the server instructions accordingly.
    if registry is None:
        registry = Registry(config.compiled_output_dir)
        registry.load()
    if cache is None:
        cache = CacheStore(
            config.cache_db_path,
            max_functions=config.reusable_function_max_count,
            ttl_days=config.reusable_function_ttl_days,
        )

    # Compute once; guard so a broken registry at startup doesn't crash the server.
    try:
        servers_with_skills = [s.name for s in registry.list_servers() if registry.has_skills(s.name)]
    except Exception:  # noqa: BLE001
        logger.warning("skills_discovery_failed")
        servers_with_skills = []

    # Load top-level tool definitions from compiled directories (if any).
    # Done before FastMCP construction so their names appear in the instructions.
    top_level_tools = _load_top_level_tools(config.compiled_output_dir)

    mcp: FastMCP = FastMCP(
        name="MCE — MCP Code Execution",
        instructions=_build_instructions(registry, servers_with_skills, top_level_tools),
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
        """List all available API servers and their instances.

        Returns server names, descriptions, instance information, and read-only status.
        Use this to discover available API servers before calling get_functions.
        """
        try:
            servers = registry.list_servers()
            logger.info("tool_list_servers_called", count=len(servers))
            return str(
                _toon_encode(
                    {
                        "sandbox_libraries": _sandbox_libraries,
                        "servers": [
                            {
                                "name": srv.name,
                                "description": srv.description,
                                "instances": [
                                    {
                                        "instance_name": inst.instance_name,
                                        "is_read_only": inst.is_read_only,
                                    }
                                    for inst in srv.instances
                                ],
                            }
                            for srv in servers
                        ],
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("list_servers_unexpected_error")
            return str(_toon_encode({"error": "Internal error loading servers", "detail": str(exc)}))

    @mcp.tool()
    async def get_functions(server_name: str, read_only: bool) -> str:
        """Get list of available functions for a server, filtered by access type.

        Args:
            server_name: Name of the API server.
            read_only: If True, return only GET methods (safe for read-only instances).
                      If False, return only write methods (POST, PUT, DELETE, PATCH).

        Returns a list of functions with their summaries, methods, and paths.
        Use this to discover available endpoints before getting detailed signatures.
        """
        try:
            # Validate server exists and get its function list
            servers = registry.list_servers()
            server_info = next((s for s in servers if s.name == server_name), None)
            if server_info is None:
                return str(_toon_encode({
                    "error": f"Server '{server_name}' not found",
                    "error_type": "server_not_found",
                }))

            # Filter functions by method type
            functions = []
            for fn_name in server_info.functions:
                try:
                    fn = registry.get_function(server_name, fn_name)
                    is_get_method = fn.method.upper() == "GET"

                    # Include if: read_only=True and GET, OR read_only=False and not GET
                    if read_only == is_get_method:
                        functions.append(
                            {
                                "name": fn.function_name,
                                "summary": fn.summary,
                                "method": fn.method,
                                "path": fn.path,
                            }
                        )
                except Exception:  # noqa: BLE001
                    # Skip functions that can't be loaded
                    continue

            logger.info("tool_get_functions_called", server=server_name, read_only=read_only, count=len(functions))
            return str(_toon_encode({"server": server_name, "read_only": read_only, "functions": functions}))

        except ServerNotFoundError as exc:
            return str(_toon_encode({"error": str(exc), "error_type": "server_not_found"}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_functions_unexpected_error")
            return str(_toon_encode({"error": "Internal error", "error_type": "internal", "detail": str(exc)}))

    @mcp.tool()
    async def get_function_details(server_name: str, function_name: str) -> str:
        """Get detailed signature and schema for a specific function.

        Args:
            server_name: Name of the API server.
            function_name: Name of the function.

        Returns detailed parameter types, response schema, and usage example.
        Use this after discovering functions with get_functions to understand how to call them.
        """
        try:
            fn = registry.get_function(server_name, function_name)
            logger.info("tool_get_function_details_called", server=server_name, function=function_name)
            return str(
                _toon_encode(
                    {
                        "function": fn.function_name,
                        "summary": fn.summary,
                        "method": fn.method,
                        "path": fn.path,
                        "parameters": [p.model_dump() for p in fn.parameters],
                        "return_type": fn.return_type,
                        "response_fields": [r.model_dump() for r in fn.response_fields],
                        "usage_example": fn.source_code,
                        "import_statement": f"from {fn.server_name}.functions import {function_name}",
                    }
                )
            )
        except ServerNotFoundError as exc:
            return str(_toon_encode({"error": str(exc), "error_type": "server_not_found"}))
        except FunctionNotFoundError as exc:
            return str(_toon_encode({"error": str(exc), "error_type": "function_not_found"}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_function_details_unexpected_error")
            return str(_toon_encode({"error": "Internal error", "error_type": "internal", "detail": str(exc)}))

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
                    "instances_used": func.instances_used,
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
                    "instances_used": func.instances_used,
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
    async def list_reusable_functions(instance_filter: list[str] | None = None) -> str:
        """List all reusable functions in the library, optionally filtered by instance(s).

        Returns functions sorted by popularity (times_used DESC).

        Args:
            instance_filter: Optional list of instance names to filter by (e.g., ["prod", "staging"]).
                           Leave empty to see all reusable functions.

        Returns:
            List of functions with name, description, times_used, last_used, and which instances they've been used with.
        """
        try:
            functions = await cache.list_reusable_functions(instance_filter)
            logger.info("tool_list_reusable_functions_called", count=len(functions), filter=instance_filter)
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

    # Register one concrete static resource per server that has a skills document.
    # Static resources (no URI-template params) appear in resources/list, making them
    # immediately discoverable.  Keeping registration conditional avoids surfacing
    # empty resources when no skills are configured.
    def _make_skills_resource(sn: str) -> None:
        @mcp.resource(
            f"skills://{sn}",
            name=f"{sn}_skills",
            description=f"Skills guide for {sn}: usage patterns, best practices, and worked examples.",
            mime_type="text/markdown",
        )
        def _get_skills() -> str:
            """Return the skills guide for this API server."""
            skills_file = registry.skills_path(sn)
            if skills_file is None:
                logger.debug("skills_resource_miss", server=sn)
                return f"No skills documentation is available for server '{sn}'."
            logger.debug("skills_resource_served", server=sn)
            return skills_file.read_text(encoding="utf-8")

    for _sn in servers_with_skills:
        _make_skills_resource(_sn)

    # Register top-level tools as first-class FastMCP tools.
    # Each tool is an async function defined in compiled/<server>/top_level_functions.py.
    # The function's __name__ becomes the MCP tool name; its docstring the description.
    _registered_tool_names: set[str] = set()
    for _entry in top_level_tools:
        _tool_fn = _entry["fn"]
        _tool_name: str = _entry.get("name", _tool_fn.__name__)
        _tool_server: str = _entry.get("server", "?")
        if _tool_name in _registered_tool_names:
            logger.warning(
                "top_level_tool_name_conflict",
                name=_tool_name,
                server=_tool_server,
                detail="Skipping duplicate tool name — rename the function in swaggers.yaml",
            )
            continue
        try:
            mcp.tool()(_tool_fn)
            _registered_tool_names.add(_tool_name)
            logger.info("top_level_tool_registered", name=_tool_name, server=_tool_server)
        except Exception as _exc:  # noqa: BLE001
            logger.warning(
                "top_level_tool_registration_failed",
                name=_tool_name,
                server=_tool_server,
                error=str(_exc),
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
