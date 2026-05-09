"""
BaseAgent: Parent class for all BVMT agents.

Provides:
  - DB connection via asyncpg
  - MCP tool calling helpers (for when servers are running)
  - Logging shortcuts
  - Abstract run() method that every agent must implement
"""

import asyncpg
import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from agents.agent_state import AgentState

load_dotenv()


class BaseAgent(ABC):
    """Abstract base class. All 7 agents inherit from this."""

    def __init__(self, name: str):
        self.name = name
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", 5432)),
            "database": os.getenv("DB_NAME", "bvmt_db"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", ""),
        }

    async def get_db(self) -> asyncpg.Connection:
        """Get an async DB connection. Remember to close it after use."""
        return await asyncpg.connect(**self.db_config)

    async def query_db(self, sql: str, *params) -> list:
        """Run a SELECT query. Returns list of Record objects."""
        conn = await self.get_db()
        try:
            return await conn.fetch(sql, *params)
        finally:
            await conn.close()

    async def execute_db(self, sql: str, *params):
        """Run an INSERT/UPDATE/DELETE query."""
        conn = await self.get_db()
        try:
            await conn.execute(sql, *params)
        finally:
            await conn.close()

    @abstractmethod
    async def run(self, state: AgentState) -> AgentState:
        """
        Every agent must implement this method.
        It receives the shared AgentState, does its work,
        writes its output to state.agent_outputs[self.name],
        and returns the updated state.
        """
        pass