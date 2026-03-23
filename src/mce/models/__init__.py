"""Pydantic models for MCE — swagger/OpenAPI, functions, execution, and cache."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Swagger / OpenAPI models (swagger.py namespace)
# ---------------------------------------------------------------------------


class ParamSchema(BaseModel):
    """Represents a single parameter to an API endpoint."""

    name: str
    location: str  # "query" | "path" | "header" | "body"
    param_type: str  # "string" | "integer" | "number" | "boolean" | "object" | "array"
    required: bool = False
    description: str = ""
    default: str | None = None
    enum: list[str] | None = None


class ResponseField(BaseModel):
    """Represents a field in an API response schema."""

    name: str
    field_type: str
    description: str = ""
    required: bool = True  # Whether this field is required per swagger "required" array
    nested: list[ResponseField] | None = None  # 1 level only


class EndpointSpec(BaseModel):
    """Normalized representation of a single API endpoint."""

    path: str
    method: str
    operation_id: str
    summary: str
    description: str = ""
    parameters: list[ParamSchema] = Field(default_factory=list)
    request_body_schema: dict[str, Any] | None = None
    request_content_types: list[str] = Field(default_factory=list)  # Available request content types, e.g., ["application/xml", "application/json"]
    response_schema: list[ResponseField] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    base_url: str = ""  # Override base URL for this endpoint (from operation-level servers)
    response_content_types: list[str] = Field(default_factory=list)  # Available response content types, e.g., ["application/xml", "application/json"]


class ServerSpec(BaseModel):
    """Normalized representation of a complete API server from a swagger doc."""

    name: str
    description: str
    base_url: str
    auth_type: str = "jwt"
    is_read_only: bool
    endpoints: list[EndpointSpec] = Field(default_factory=list)
    swagger_hash: str


# ---------------------------------------------------------------------------
# Swagger source config model
# ---------------------------------------------------------------------------


class ServerInstance(BaseModel):
    """Configuration for a specific deployment instance of an API server."""

    instance_name: str
    base_url: str
    session_credentials: dict[str, str] = Field(default_factory=dict)
    is_read_only: bool = False


class SwaggerSource(BaseModel):
    """Configuration for an API server with its OpenAPI spec and auth settings.

    Supports both single-instance (top-level base_url / credentials) and
    multi-instance (instances list) deployments.  All instances share the same
    API surface defined by swagger_url.
    """

    # Required fields
    name: str
    swagger_url: str
    base_url: str = ""

    # Server-level flags
    is_read_only: bool = False
    auth_type: str = "jwt"

    # JWT auth — top-level auth header value (may contain ${ENV_VAR} references)
    auth_header: str = ""

    # Session auth — shared across all instances
    session_endpoint: str | None = None
    session_cookie_name: str | None = None
    # Top-level session credentials (used when no instances are defined)
    session_credentials: dict[str, str] = Field(default_factory=dict)

    extra_headers: dict[str, str] = Field(default_factory=dict)
    headers: str = ""  # "[key1:value1,key2:value2]" format; parsed into extra_headers
    skills_url: str | None = None  # Optional: local path or HTTP URL to a skills.md document
    top_level_functions: list[str] = Field(
        default_factory=list
    )  # Optional: function names to expose as direct MCP tools

    # Optional multi-instance definitions
    instances: list[ServerInstance] = Field(default_factory=list)

    @field_validator("auth_type")
    @classmethod
    def validate_auth_type(cls, v: str) -> str:
        """Validate auth_type is either jwt or session."""
        if v not in ("jwt", "session"):
            raise ValueError(f"auth_type must be 'jwt' or 'session', got {v}")
        return v

    @model_validator(mode="after")
    def _parse_headers(self) -> SwaggerSource:
        if self.headers:
            raw = self.headers.strip().strip("[]")
            for pair in raw.split(","):
                if ":" in pair:
                    k, _, v = pair.partition(":")
                    self.extra_headers[k.strip()] = v.strip()
        return self

    @model_validator(mode="after")
    def validate_auth_fields(self) -> SwaggerSource:
        """Ensure session auth has all required fields."""
        if self.auth_type == "session":
            if not self.session_cookie_name:
                raise ValueError("session auth requires session_cookie_name")
            if not self.session_endpoint:
                raise ValueError("session auth requires session_endpoint")

            # Credentials must be supplied either at top level or per instance
            has_top_level_creds = bool(self.session_credentials)
            has_instance_creds = any(inst.session_credentials for inst in self.instances)
            if not has_top_level_creds and not has_instance_creds:
                raise ValueError("session auth requires session_credentials")

        return self


# ---------------------------------------------------------------------------
# Function info models (function.py namespace)
# ---------------------------------------------------------------------------


class FunctionInfo(BaseModel):
    """Complete metadata for a compiled API function."""

    server_name: str
    function_name: str
    summary: str
    description: str = ""
    parameters: list[ParamSchema] = Field(default_factory=list)
    response_fields: list[ResponseField] = Field(default_factory=list)
    return_type: str = "Any"
    source_code: str
    method: str = ""
    path: str = ""


class InstanceInfo(BaseModel):
    """Instance information for list_servers response."""

    instance_name: str
    base_url: str
    is_read_only: bool


class ServerInfo(BaseModel):
    """Summary metadata for a compiled server (used in list_servers response)."""

    name: str
    description: str
    functions: list[str] = Field(default_factory=list)
    function_summaries: dict[str, str] = Field(default_factory=dict)
    instances: list[InstanceInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Execution models (execution.py namespace)
# ---------------------------------------------------------------------------


class ExecutionResult(BaseModel):
    """Result from sandboxed code execution."""

    success: bool
    data: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    error: str | None = None
    traceback: str | None = None  # Only populated in debug mode
    prints: str | None = None  # Captured stdout from print() calls in user code
    execution_time_ms: int = 0
    cache_id: str | None = None  # Set when execution result is stored in the code cache


# ---------------------------------------------------------------------------
# Reusable Function Library models (cache.py namespace)
# ---------------------------------------------------------------------------


class ReusableFunction(BaseModel):
    """Persistent, use-case-specific function in the library."""

    name: str
    description: str
    code: str
    instances_used: list[str] = Field(default_factory=list)
    times_used: int = 1
    created_at: float
    last_used_at: float


# ---------------------------------------------------------------------------
# Manifest model
# ---------------------------------------------------------------------------


class EndpointManifest(BaseModel):
    """Manifest entry for a single compiled endpoint."""

    function_name: str
    summary: str
    method: str
    path: str
    parameters_summary: str
    response_summary: str
    return_type: str = "Any"


class ServerManifest(BaseModel):
    """Compiled server manifest written to disk."""

    server_name: str
    description: str
    swagger_hash: str
    template_hash: str = ""
    compiled_at: str
    base_url: str
    is_read_only: bool
    endpoints: list[EndpointManifest] = Field(default_factory=list)
    instances: list[dict[str, Any]] = Field(default_factory=list)  # Simplified instance data for manifest
