"""Runtime registry — loads compiled manifests and provides fast function lookup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mce.errors import FunctionNotFoundError, ServerNotFoundError
from mce.models import (
    EndpointManifest,
    FunctionInfo,
    InstanceInfo,
    ParamSchema,
    ResponseField,
    ServerInfo,
    ServerManifest,
)
from mce.utils.logging import get_logger

logger = get_logger(__name__)


class Registry:
    """Loads and indexes compiled API manifests for fast LLM tool lookups."""

    def __init__(self, compiled_dir: str) -> None:
        """Initialize the registry from the compiled output directory.

        Args:
            compiled_dir: Path to directory containing compiled server subdirectories.
        """
        self._compiled_dir = Path(compiled_dir)
        self._manifests: dict[str, ServerManifest] = {}
        self._function_source_cache: dict[str, str] = {}

    def load(self) -> None:
        """Load all compiled manifests from server subdirectories.

        Scans for manifest.json files in each subdirectory of compiled_dir.
        """
        self._manifests = {}
        self._function_source_cache.clear()

        if not self._compiled_dir.exists():
            logger.warning("compiled_dir_not_found", path=str(self._compiled_dir))
            return

        for manifest_path in sorted(self._compiled_dir.glob("*/manifest.json")):
            server_name = manifest_path.parent.name
            try:
                self._load_manifest(server_name, manifest_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("manifest_load_failed", path=str(manifest_path), error=str(exc))

        total = sum(len(m.endpoints) for m in self._manifests.values())
        if self._manifests:
            logger.info("registry_loaded", servers=len(self._manifests), total_functions=total)
        else:
            logger.warning("no_manifest_loaded")

    def _load_manifest(self, server_name: str, manifest_path: Path) -> None:
        """Load a single manifest file.

        Args:
            server_name: Directory name used as the server key.
            manifest_path: Path to manifest.json file.
        """
        with open(manifest_path, encoding="utf-8") as f:
            raw = json.load(f)

        self._manifests[server_name] = ServerManifest(**raw)
        logger.debug("manifest_loaded", server=server_name, endpoints=len(self._manifests[server_name].endpoints))

    def list_servers(self) -> list[ServerInfo]:
        """Return summary metadata for all loaded servers.

        Returns:
            List of ServerInfo objects sorted by server name.
        """
        result: list[ServerInfo] = []
        for server_name, manifest in sorted(self._manifests.items()):
            instances = [
                InstanceInfo(
                    instance_name=inst["instance_name"],
                    base_url=inst["base_url"],
                    is_read_only=inst["is_read_only"],
                )
                for inst in manifest.instances
            ]
            result.append(
                ServerInfo(
                    name=server_name,
                    description=manifest.description,
                    functions=[ep.function_name for ep in manifest.endpoints],
                    function_summaries={ep.function_name: ep.summary for ep in manifest.endpoints},
                    instances=instances,
                )
            )
        return result

    def list_instances(self) -> list[Any]:
        """Return list of available instances across all servers.

        Returns:
            Flat list of InstanceInfo objects.
        """
        instances: list[Any] = []
        for manifest in self._manifests.values():
            for inst in manifest.instances:
                instances.append(
                    InstanceInfo(
                        instance_name=inst["instance_name"],
                        base_url=inst["base_url"],
                        is_read_only=inst["is_read_only"],
                    )
                )
        return instances

    def list_function_names(self, server_name: str | None = None) -> list[str]:
        """Return list of all function names, optionally filtered by server.

        Args:
            server_name: If provided, returns only functions for that server.

        Returns:
            List of function names.
        """
        if server_name is not None:
            manifest = self._manifests.get(server_name)
            if manifest is None:
                return []
            return [ep.function_name for ep in manifest.endpoints]

        names: list[str] = []
        for manifest in self._manifests.values():
            names.extend(ep.function_name for ep in manifest.endpoints)
        return names

    def get_function(self, server_name: str, function_name: str) -> FunctionInfo:
        """Get detailed function information by server and function name.

        Args:
            server_name: Name of the server (compiled directory name).
            function_name: Name of the function.

        Returns:
            FunctionInfo with full parameter and response schema.

        Raises:
            ServerNotFoundError: If the server doesn't exist in the registry.
            FunctionNotFoundError: If the function doesn't exist.
        """
        manifest = self._manifests.get(server_name)
        if manifest is None:
            available = list(self._manifests.keys())
            raise ServerNotFoundError(f"Server '{server_name}' not found. Available: {available}")

        endpoint = self._find_endpoint(server_name, function_name)
        source_code = self._get_function_source(server_name, function_name)
        parameters = self._parse_parameters_summary(endpoint.parameters_summary)
        response_fields = self._parse_response_summary(endpoint.response_summary)

        return FunctionInfo(
            server_name=server_name,
            function_name=function_name,
            summary=endpoint.summary,
            parameters=parameters,
            response_fields=response_fields,
            return_type=endpoint.return_type,
            source_code=source_code,
            method=endpoint.method,
            path=endpoint.path,
        )

    def get_function_source(self, server_name: str, function_name: str) -> str:
        """Get the Python source code for a specific function.

        Args:
            server_name: Name of the server.
            function_name: Function name.

        Returns:
            Python source code string for the function.

        Raises:
            ServerNotFoundError: If server not found.
            FunctionNotFoundError: If function not found.
        """
        if server_name not in self._manifests:
            available = list(self._manifests.keys())
            raise ServerNotFoundError(f"Server '{server_name}' not found. Available: {available}")
        return self._get_function_source(server_name, function_name)

    def get_swagger_hash(self, server_name: str) -> str:
        """Get the swagger hash for a compiled server.

        Args:
            server_name: Name of the server.

        Returns:
            Swagger hash string.

        Raises:
            ServerNotFoundError: If server not found.
        """
        manifest = self._manifests.get(server_name)
        if manifest is None:
            available = list(self._manifests.keys())
            raise ServerNotFoundError(f"Server '{server_name}' not found. Available: {available}")
        return manifest.swagger_hash

    def has_skills(self, server_name: str | None = None) -> bool:
        """Return True if a compiled skills.md document exists for the given server.

        If server_name is None, returns True if ANY server has skills.

        Args:
            server_name: Optional server name to check. If None, checks all servers.

        Returns:
            True when skills.md is present on disk for the given server.
        """
        if server_name is not None:
            return (self._compiled_dir / server_name / "skills.md").is_file()
        # Check all servers
        return any((self._compiled_dir / sn / "skills.md").is_file() for sn in self._manifests)

    def skills_path(self, server_name: str | None = None) -> Path | None:
        """Return the Path to skills.md for the given server, or None if absent.

        If server_name is None, returns the first skills.md found across all servers.

        Args:
            server_name: Optional server name. If None, returns first skills file found.

        Returns:
            Absolute Path to skills.md, or None.
        """
        if server_name is not None:
            path = self._compiled_dir / server_name / "skills.md"
            return path if path.is_file() else None
        # Return first skills file found
        for sn in sorted(self._manifests.keys()):
            path = self._compiled_dir / sn / "skills.md"
            if path.is_file():
                return path
        return None

    def _find_endpoint(self, server_name: str, function_name: str) -> EndpointManifest:
        """Find an endpoint entry in the manifest.

        Args:
            server_name: Server name.
            function_name: Function name to find.

        Returns:
            Endpoint manifest entry.

        Raises:
            FunctionNotFoundError: If function not in manifest.
        """
        manifest = self._manifests.get(server_name)
        if not manifest:
            raise FunctionNotFoundError(f"No manifest for server '{server_name}'")

        for ep in manifest.endpoints:
            if ep.function_name == function_name:
                return ep

        available = [ep.function_name for ep in manifest.endpoints]
        raise FunctionNotFoundError(f"Function '{function_name}' not found in '{server_name}'. Available: {available}")

    def _get_function_source(self, server_name: str, function_name: str) -> str:
        """Extract the Python source for a function from functions.py.

        Args:
            server_name: Server name.
            function_name: Function name.

        Returns:
            Source code of the specific function, or full file if extraction fails.
        """
        cache_key = f"{server_name}.{function_name}"
        if cache_key in self._function_source_cache:
            return self._function_source_cache[cache_key]

        functions_file = self._compiled_dir / server_name / "functions.py"
        if not functions_file.exists():
            return f"# Source not found for {function_name}"

        full_source = functions_file.read_text(encoding="utf-8")
        snippet = self._extract_function_snippet(full_source, function_name)
        self._function_source_cache[cache_key] = snippet
        return snippet

    def _extract_function_snippet(self, source: str, function_name: str) -> str:
        """Extract a function definition and its associated TypedDict classes from a Python source file.

        Args:
            source: Full Python source file content.
            function_name: Name of function to extract.

        Returns:
            TypedDict class definitions (if any) followed by the function source,
            or full source on failure.
        """
        import ast  # noqa: PLC0415

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        lines = source.splitlines()

        # Derive the PascalCase prefix used for TypedDict class names
        pascal_prefix = "".join(word.capitalize() for word in function_name.split("_"))

        # Collect TypedDict classes whose names match this function's response types
        class_snippets: list[str] = []
        func_snippet: str | None = None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith(pascal_prefix):
                start = node.lineno - 1
                end = node.end_lineno or (start + 1)
                class_snippets.append("\n".join(lines[start:end]))
            elif isinstance(node, ast.FunctionDef) and node.name == function_name:
                start = node.lineno - 1
                end = node.end_lineno or (start + 1)
                func_snippet = "\n".join(lines[start:end])

        if func_snippet is None:
            return source  # Fallback to full source

        parts = class_snippets + [func_snippet]
        return "\n\n".join(parts)

    def _parse_parameters_summary(self, summary: str) -> list[ParamSchema]:
        """Parse the human-readable parameters_summary string into ParamSchema objects.

        Args:
            summary: e.g. "city (string, required), date (string, optional)"

        Returns:
            List of ParamSchema objects.
        """
        import re  # noqa: PLC0415

        if not summary.strip():
            return []

        params: list[ParamSchema] = []
        for part in re.split(r"\),\s*", summary.rstrip(")")):
            part = part.strip()  # noqa: PLW2901
            if not part:
                continue
            try:
                name, rest = part.split("(", 1)
                type_str, req_str = rest.split(",", 1)
                params.append(
                    ParamSchema(
                        name=name.strip(),
                        location="query",
                        param_type=type_str.strip(),
                        required="required" in req_str.lower(),
                    )
                )
            except ValueError:
                params.append(ParamSchema(name=part, location="query", param_type="string"))

        return params

    def _parse_response_summary(self, summary: str) -> list[ResponseField]:
        """Parse response_summary string into ResponseField list.

        Args:
            summary: e.g. "id, name, price"

        Returns:
            List of ResponseField objects.
        """
        if not summary or summary == "response data":
            return []
        return [ResponseField(name=field.strip(), field_type="string") for field in summary.split(",") if field.strip()]

