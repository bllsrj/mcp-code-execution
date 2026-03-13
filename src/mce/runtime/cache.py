"""SQLite-backed reusable function library."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

from mce.errors import CacheError, CompileError
from mce.models import ReusableFunction
from mce.utils.logging import get_logger

if TYPE_CHECKING:
    from mce.runtime.executor import CodeExecutor

logger = get_logger(__name__)

# Pattern to detect server function imports in user code
_SERVER_IMPORT_RE = re.compile(r"from\s+(\w+)\.functions\s+import|import\s+(\w+)\.functions")


def _detect_servers_used(code: str) -> list[str]:
    """Detect which server function modules are imported in the code.

    Args:
        code: Python source code.

    Returns:
        List of server names referenced by from <name>.functions import.
    """
    servers: set[str] = set()
    for match in _SERVER_IMPORT_RE.finditer(code):
        name = match.group(1) or match.group(2)
        if name:
            servers.add(name)
    return sorted(servers)


def _human_readable_time(timestamp: float) -> str:
    """Convert timestamp to human-readable relative time.

    Args:
        timestamp: Unix timestamp in seconds.

    Returns:
        Human-readable string like "2 hours ago", "3 days ago".
    """
    now = time.time()
    delta = now - timestamp

    if delta < 60:
        return "just now"
    elif delta < 3600:
        mins = int(delta / 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    elif delta < 86400:
        hours = int(delta / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(delta / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS reusable_functions (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    code TEXT NOT NULL,
    servers_used TEXT NOT NULL,
    times_used INTEGER DEFAULT 1,
    created_at REAL NOT NULL,
    last_used_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_last_used ON reusable_functions(last_used_at);
CREATE INDEX IF NOT EXISTS idx_times_used ON reusable_functions(times_used DESC);
"""

_DROP_OLD_TABLE_SQL = """
DROP TABLE IF EXISTS code_cache;
"""


class CacheStore:
    """Reusable function library with auto-pruning and capacity management."""

    def __init__(
        self,
        db_path: str,
        executor: CodeExecutor | None = None,
        max_functions: int = 50,
        ttl_days: int = 30,
    ) -> None:
        """Initialize the reusable function library.

        Args:
            db_path: Filesystem path for the SQLite database file.
            executor: Code executor for validation (injected after init).
            max_functions: Maximum number of functions before pruning.
            ttl_days: Days of inactivity before auto-deletion.
        """
        self._db_path = db_path
        self._executor = executor
        self._max_functions = max_functions
        self._ttl_days = ttl_days

    def set_executor(self, executor: CodeExecutor) -> None:
        """Set the executor reference (called after both are initialized).

        Args:
            executor: Code executor instance.
        """
        self._executor = executor

    async def initialize(self) -> None:
        """Create database tables and drop old cache table.

        Raises:
            CacheError: If database initialization fails.
        """
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            async with aiosqlite.connect(self._db_path) as db:
                # Drop old cache table
                await db.executescript(_DROP_OLD_TABLE_SQL)
                # Create new reusable_functions table
                await db.executescript(_CREATE_TABLE_SQL)
                await db.commit()
            logger.info("reusable_function_library_initialized", path=self._db_path)
        except aiosqlite.Error as exc:
            raise CacheError(f"Failed to initialize reusable function library: {exc}") from exc

    async def save_reusable_function(
        self,
        name: str,
        description: str,
        code: str,
    ) -> ReusableFunction:
        """Save new reusable function after validation.

        Args:
            name: Unique semantic name.
            description: What this function does.
            code: Python code (must pass validation).

        Returns:
            Saved ReusableFunction.

        Raises:
            ValueError: If name already exists.
            CompileError: If code validation fails.
            CacheError: If database operation fails.
        """
        # Check if name already exists
        existing = await self.get_reusable_function(name)
        if existing:
            raise ValueError(f"Function '{name}' already exists")

        # Validate code
        if self._executor:
            is_valid, error = await self._executor.validate_code_only(code)
            if not is_valid:
                raise CompileError(f"Code validation failed: {error}")

        # Auto-detect servers
        servers_used = _detect_servers_used(code)

        # Check capacity and prune if needed
        await self._enforce_global_cap_if_needed()

        # Save to DB
        now = time.time()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """
                    INSERT INTO reusable_functions
                        (name, description, code, servers_used, times_used, created_at, last_used_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    """,
                    (name, description, code, json.dumps(servers_used), now, now),
                )
                await db.commit()
            logger.info("function_saved", name=name, servers=servers_used)
            return ReusableFunction(
                name=name,
                description=description,
                code=code,
                servers_used=servers_used,
                times_used=1,
                created_at=now,
                last_used_at=now,
            )
        except aiosqlite.Error as exc:
            raise CacheError(f"Failed to save function: {exc}") from exc

    async def overwrite_reusable_function(
        self,
        name: str,
        description: str,
        code: str,
        reason: str,
    ) -> ReusableFunction:
        """Overwrite existing function, resetting usage stats.

        Args:
            name: Name of existing function.
            description: Updated description.
            code: New implementation.
            reason: Why overwriting (for logging).

        Returns:
            Updated ReusableFunction.

        Raises:
            ValueError: If name doesn't exist.
            CompileError: If code validation fails.
            CacheError: If database operation fails.
        """
        # Check if exists
        existing = await self.get_reusable_function(name)
        if not existing:
            raise ValueError(f"Function '{name}' not found")

        # Validate code
        if self._executor:
            is_valid, error = await self._executor.validate_code_only(code)
            if not is_valid:
                raise CompileError(f"Code validation failed: {error}")

        # Auto-detect servers
        servers_used = _detect_servers_used(code)

        # Log reason
        logger.info("function_overwritten", name=name, reason=reason)

        # Update in DB, reset times_used to 1
        now = time.time()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """
                    UPDATE reusable_functions
                    SET description = ?, code = ?, servers_used = ?, times_used = 1,
                        created_at = ?, last_used_at = ?
                    WHERE name = ?
                    """,
                    (description, code, json.dumps(servers_used), now, now, name),
                )
                await db.commit()
            return ReusableFunction(
                name=name,
                description=description,
                code=code,
                servers_used=servers_used,
                times_used=1,
                created_at=now,
                last_used_at=now,
            )
        except aiosqlite.Error as exc:
            raise CacheError(f"Failed to overwrite function: {exc}") from exc

    async def get_reusable_function(self, name: str) -> ReusableFunction | None:
        """Fetch function by name.

        Args:
            name: Function name.

        Returns:
            ReusableFunction if found, None otherwise.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM reusable_functions WHERE name = ?", (name,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        return None
                    return ReusableFunction(
                        name=row["name"],
                        description=row["description"],
                        code=row["code"],
                        servers_used=json.loads(row["servers_used"]),
                        times_used=row["times_used"],
                        created_at=row["created_at"],
                        last_used_at=row["last_used_at"],
                    )
        except aiosqlite.Error as exc:
            logger.error("get_function_failed", name=name, error=str(exc))
            return None

    async def list_reusable_functions(
        self, server_filter: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """List all functions, optionally filtered by server(s).

        Args:
            server_filter: Optional list of server names to filter by.

        Returns:
            List of dicts with function metadata, sorted by times_used DESC.
            If single server filter: omit 'servers_used' field from response.
            Otherwise: include 'servers_used' field.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row

                # Build query
                if server_filter:
                    # Filter functions that use any of the specified servers
                    query = """
                    SELECT * FROM reusable_functions
                    ORDER BY times_used DESC
                    """
                    async with db.execute(query) as cursor:
                        rows = await cursor.fetchall()
                        # Filter in Python (SQLite JSON querying is complex)
                        results = []
                        for row in rows:
                            servers = json.loads(row["servers_used"])
                            if any(s in server_filter for s in servers):
                                results.append(row)
                else:
                    query = "SELECT * FROM reusable_functions ORDER BY times_used DESC"
                    async with db.execute(query) as cursor:
                        results = list(await cursor.fetchall())

                # Format response
                include_servers = not server_filter or len(server_filter) != 1
                output = []
                for row in results:
                    item: dict[str, Any] = {
                        "name": row["name"],
                        "description": row["description"],
                        "times_used": row["times_used"],
                        "last_used": _human_readable_time(row["last_used_at"]),
                    }
                    if include_servers:
                        item["servers_used"] = json.loads(row["servers_used"])
                    output.append(item)
                return output
        except aiosqlite.Error as exc:
            logger.error("list_functions_failed", error=str(exc))
            return []

    async def increment_usage(self, name: str, code: str) -> None:
        """Increment times_used, update last_used_at, and refresh servers_used.

        Args:
            name: Function name.
            code: Function code (to re-detect servers).
        """
        # Detect current servers
        new_servers = _detect_servers_used(code)

        try:
            async with aiosqlite.connect(self._db_path) as db:
                # Fetch existing servers
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT servers_used FROM reusable_functions WHERE name = ?", (name,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        return
                    existing_servers = json.loads(row["servers_used"])

                # Merge servers
                updated_servers = sorted(set(existing_servers) | set(new_servers))

                # Update
                await db.execute(
                    """
                    UPDATE reusable_functions
                    SET times_used = times_used + 1,
                        last_used_at = ?,
                        servers_used = ?
                    WHERE name = ?
                    """,
                    (time.time(), json.dumps(updated_servers), name),
                )
                await db.commit()

                if updated_servers != existing_servers:
                    logger.info("function_servers_updated", name=name, servers=updated_servers)
        except aiosqlite.Error as exc:
            logger.error("increment_usage_failed", name=name, error=str(exc))

    async def delete_reusable_function(self, name: str, reason: str) -> bool:
        """Delete function by name.

        Args:
            name: Function name.
            reason: Why deleting (for logging).

        Returns:
            True if deleted, False if not found.
        """
        logger.info("function_deleted", name=name, reason=reason)
        try:
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM reusable_functions WHERE name = ?", (name,)
                )
                await db.commit()
                return cursor.rowcount > 0
        except aiosqlite.Error as exc:
            logger.error("delete_function_failed", name=name, error=str(exc))
            return False

    async def prune_old_functions(self, ttl_days: int | None = None) -> int:
        """Delete functions unused for ttl_days or more.

        Args:
            ttl_days: Days of inactivity before deletion (uses instance default if None).

        Returns:
            Number of functions deleted.
        """
        days = ttl_days if ttl_days is not None else self._ttl_days
        cutoff = time.time() - (days * 86400)

        try:
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM reusable_functions WHERE last_used_at < ?", (cutoff,)
                )
                await db.commit()
                count = cursor.rowcount
                if count > 0:
                    logger.info("old_functions_pruned", count=count, ttl_days=days)
                return count
        except aiosqlite.Error as exc:
            logger.error("prune_old_failed", error=str(exc))
            return 0

    async def enforce_global_cap(self, limit: int | None = None) -> int:
        """Delete least-used functions if count exceeds limit.

        Args:
            limit: Maximum number of functions to keep (uses instance default if None).

        Returns:
            Number of functions deleted.
        """
        cap = limit if limit is not None else self._max_functions

        try:
            async with aiosqlite.connect(self._db_path) as db:
                # Check count
                async with db.execute("SELECT COUNT(*) as count FROM reusable_functions") as cursor:
                    row = await cursor.fetchone()
                    current_count = row[0] if row else 0

                if current_count <= cap:
                    return 0

                # Delete least-used
                to_delete = current_count - cap
                await db.execute(
                    """
                    DELETE FROM reusable_functions
                    WHERE name IN (
                        SELECT name FROM reusable_functions
                        ORDER BY times_used ASC, last_used_at ASC
                        LIMIT ?
                    )
                    """,
                    (to_delete,),
                )
                await db.commit()
                logger.info("cap_enforcement_pruned", count=to_delete, limit=cap)
                return to_delete
        except aiosqlite.Error as exc:
            logger.error("enforce_cap_failed", error=str(exc))
            return 0

    async def _enforce_global_cap_if_needed(self) -> None:
        """Check capacity and prune if at limit (before saving new function)."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute("SELECT COUNT(*) as count FROM reusable_functions") as cursor:
                    row = await cursor.fetchone()
                    current_count = row[0] if row else 0

                # If adding one more would exceed, enforce cap now
                if current_count + 1 > self._max_functions:
                    await self.enforce_global_cap()
        except aiosqlite.Error as exc:
            logger.error("cap_check_failed", error=str(exc))

    async def auto_prune(
        self, ttl_days: int | None = None, cap: int | None = None
    ) -> dict[str, int]:
        """Run both pruning strategies.

        Args:
            ttl_days: Age threshold for deletion.
            cap: Maximum function count.

        Returns:
            {"old_deleted": N, "cap_deleted": M}
        """
        old_deleted = await self.prune_old_functions(ttl_days)
        cap_deleted = await self.enforce_global_cap(cap)
        return {"old_deleted": old_deleted, "cap_deleted": cap_deleted}
