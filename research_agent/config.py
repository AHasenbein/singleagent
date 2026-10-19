"""Runtime settings, read explicitly from the environment.

Nothing here runs at import time: call Settings.from_env() from an entrypoint
(after loading .env) and pass the result into the services that need it.
"""

import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_SEC_USER_AGENT = "Nittany AI ELP educational-project contact@example.com"
DEFAULT_DB_PATH = "team1_evidence.db"
DEFAULT_MODEL = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class MissingSettingsError(RuntimeError):
    """Raised when a required API key is not configured."""


@dataclass(frozen=True)
class Settings:
    tavily_api_key: Optional[str]
    deepseek_api_key: Optional[str]
    sec_user_agent: str = DEFAULT_SEC_USER_AGENT
    db_path: str = DEFAULT_DB_PATH
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            sec_user_agent=os.getenv("SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT),
            db_path=os.getenv("EVIDENCE_DB", DEFAULT_DB_PATH),
            model=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        )

    def require_api_keys(self) -> None:
        missing = [
            name
            for name, value in (
                ("TAVILY_API_KEY", self.tavily_api_key),
                ("DEEPSEEK_API_KEY", self.deepseek_api_key),
            )
            if not value
        ]
        if missing:
            raise MissingSettingsError(
                f"Missing API key. Add {' and '.join(missing)} "
                "to a .env file beside this script."
            )
