"""Local SQLite storage module for attendance, tracking, and metrics."""

from attendance_agent.storage.database import (
    DatabaseManager,
    get_connection,
    init_db,
)

__all__ = ["DatabaseManager", "get_connection", "init_db"]

