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
    response_schema: list[ResponseField] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    base_url: str = ""  # Override base URL for this endpoint (from operation-level servers)


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
    """Configuration for API instances with shared OpenAPI spec.

    All instances share the same API surface defined by swagger_url.
    """

    swagger_url: str

    # Shared config across all instances
    auth_type: str = "jwt"
    session_endpoint: str | None = None
    session_cookie_name: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    headers: str = ""  # "[key1:value1,key2:value2]" format; parsed into extra_headers
    skills_url: str | None = None  # Optional: local path or HTTP URL to a skills.md document
    top_level_functions: list[str] = Field(
        default_factory=list
    )  # Optional: function names to expose as direct MCP tools

    # Instance definitions (required)
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
        """Ensure session auth has required fields and instances are defined."""
        if not self.instances:
            raise ValueError("At least one instance must be defined")

        # Validate session auth for each instance
        if self.auth_type == "session":
            if not self.session_cookie_name:
                raise ValueError("Session auth requires session_cookie_name")
            if not self.session_endpoint:
                raise ValueError("Session auth requires session_endpoint")

            for instance in self.instances:
                if not instance.session_credentials:
                    raise ValueError(
                        f"Instance '{instance.instance_name}': session auth requires session_credentials"
                    )

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
