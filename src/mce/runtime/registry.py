"""Runtime registry — loads compiled manifests and provides fast function lookup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mce.errors import FunctionNotFoundError, ServerNotFoundError
from mce.models import EndpointManifest, FunctionInfo, ParamSchema, ResponseField, ServerInfo, ServerManifest
from mce.utils.logging import get_logger

logger = get_logger(__name__)


class Registry:
    """Loads and indexes the compiled API manifest for fast LLM tool lookups."""

    def __init__(self, compiled_dir: str) -> None:
        """Initialize the registry from the compiled output directory.

        Args:
            compiled_dir: Path to directory containing compiled mirth subdirectory.
        """
        self._compiled_dir = Path(compiled_dir)
        self._manifest: ServerManifest | None = None
        self._function_source_cache: dict[str, str] = {}

    def load(self) -> None:
        """Load the compiled manifest from the mirth directory.

        Loads the manifest.json file from compiled/mirth/ into memory for fast lookup.
        """
        self._manifest = None
        self._function_source_cache.clear()

        if not self._compiled_dir.exists():
            logger.warning("compiled_dir_not_found", path=str(self._compiled_dir))
            return

        # Load the single mirth manifest
        manifest_path = self._compiled_dir / "mirth" / "manifest.json"
        if manifest_path.exists():
            try:
                self._load_manifest(manifest_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("manifest_load_failed", path=str(manifest_path), error=str(exc))

        if self._manifest:
            logger.info("registry_loaded", total_functions=len(self._manifest.endpoints))
        else:
            logger.warning("no_manifest_loaded")

    def _load_manifest(self, manifest_path: Path) -> None:
        """Load the manifest file.

        Args:
            manifest_path: Path to manifest.json file.
        """
        with open(manifest_path, encoding="utf-8") as f:
            raw = json.load(f)

        self._manifest = ServerManifest(**raw)
        logger.debug("manifest_loaded", endpoints=len(self._manifest.endpoints))

    def list_instances(self) -> list[Any]:
        """Return list of available instances.

        Returns:
            List of InstanceInfo objects.
        """
        from mce.models import InstanceInfo

        if not self._manifest:
            return []

        return [
            InstanceInfo(
                instance_name=inst["instance_name"],
                base_url=inst["base_url"],
                is_read_only=inst["is_read_only"],
            )
            for inst in self._manifest.instances
        ]

    def list_function_names(self) -> list[str]:
        """Return list of all function names.

        Returns:
            List of function names.
        """
        if not self._manifest:
            return []
        return [ep.function_name for ep in self._manifest.endpoints]


    def get_function(self, function_name: str) -> FunctionInfo:
        """Get detailed function information by function name.

        Args:
            function_name: Name of the function.

        Returns:
            FunctionInfo with full parameter and response schema.

        Raises:
            FunctionNotFoundError: If function doesn't exist.
        """
        if not self._manifest:
            raise FunctionNotFoundError(f"No manifest loaded")

        endpoint = self._find_endpoint(function_name)
        source_code = self._get_function_source(function_name)
        parameters = self._parse_parameters_summary(endpoint.parameters_summary)
        response_fields = self._parse_response_summary(endpoint.response_summary)

        return FunctionInfo(
            server_name="mirth",  # Hardcoded since there's only one API
            function_name=function_name,
            summary=endpoint.summary,
            parameters=parameters,
            response_fields=response_fields,
            return_type=endpoint.return_type,
            source_code=source_code,
            method=endpoint.method,
            path=endpoint.path,
        )

    def get_function_source(self, function_name: str) -> str:
        """Get the Python source code for a specific function.

        Args:
            function_name: Function name.

        Returns:
            Python source code string for the function.

        Raises:
            FunctionNotFoundError: If function not found.
        """
        return self._get_function_source(function_name)

    def get_swagger_hash(self) -> str:
        """Get the swagger hash for the compiled API.

        Returns:
            Swagger hash string, or empty string if no manifest loaded.
        """
        if not self._manifest:
            return ""
        return self._manifest.swagger_hash

    def has_skills(self) -> bool:
        """Return True if a compiled skills.md document exists.

        Uses file-system presence as the authoritative check so that the result
        stays correct even after an incremental skills-only refresh.

        Returns:
            True when skills.md is present on disk.
        """
        return (self._compiled_dir / "mirth" / "skills.md").is_file()

    def skills_path(self) -> Path | None:
        """Return the Path to skills.md, or None if it does not exist.

        Returns:
            Absolute Path to skills.md, or None.
        """
        path = self._compiled_dir / "mirth" / "skills.md"
        return path if path.is_file() else None

    def _find_endpoint(self, function_name: str) -> EndpointManifest:
        """Find an endpoint entry in the manifest.

        Args:
            function_name: Function name to find.

        Returns:
            Endpoint manifest entry.

        Raises:
            FunctionNotFoundError: If function not in manifest.
        """
        if not self._manifest:
            raise FunctionNotFoundError(f"No manifest loaded")

        for ep in self._manifest.endpoints:
            if ep.function_name == function_name:
                return ep

        available = [ep.function_name for ep in self._manifest.endpoints]
        raise FunctionNotFoundError(f"Function '{function_name}' not found. Available: {available}")

    def _get_function_source(self, function_name: str) -> str:
        """Extract the Python source for a function from functions.py.

        Args:
            function_name: Function name.

        Returns:
            Source code of the specific function, or full file if extraction fails.
        """
        if function_name in self._function_source_cache:
            return self._function_source_cache[function_name]

        functions_file = self._compiled_dir / "mirth" / "functions.py"
        if not functions_file.exists():
            return f"# Source not found for {function_name}"

        full_source = functions_file.read_text(encoding="utf-8")
        snippet = self._extract_function_snippet(full_source, function_name)
        self._function_source_cache[function_name] = snippet
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
