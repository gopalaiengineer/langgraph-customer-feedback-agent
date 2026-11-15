
import os
from pathlib import Path
import re
import certifi
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.constants.constant import (
    DEFAULT_MANAGER_EMAILS,
    DEFAULT_SUPPORT_EMAILS,
)

# Set the CA bundle path for HTTPS requests
os.environ["SSL_CERT_FILE"] = certifi.where()

BASE_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = BASE_DIR / ".env"
EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
)


class Settings(BaseSettings):

    # LLM providers
    ENABLE_GEMINI: bool = False
    ENABLE_GROQ: bool = False
    ENABLE_OPENAI: bool = False

    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = "ap21atil@gmail.com"
    SMTP_PASSWORD: str | None = None
    SUPPORT_EMAILS: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SUPPORT_EMAILS)
    )
    MANAGER_EMAILS: list[str] = Field(
        default_factory=lambda: list(DEFAULT_MANAGER_EMAILS)
    )

    MAX_EMAIL_REVISIONS: int = Field(default=3, ge=1)
    LLM_MAX_RETRIES: int = Field(default=3, ge=0)
    LLM_REQUEST_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)

    # Gemini
    GOOGLE_API_KEY: str | None = None
    GOOGLE_MODEL: str = "gemini-3.5-flash-lite"

    # Groq
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    GROQ_API_URL: str = "https://api.groq.com/openai/v1"

    # OpenAI
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o"

    # Tavily
    ENABLE_WEB_SEARCH: bool = False
    TAVILY_API_KEY: str | None = None

    # LangSmith
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "customer-feedback-dev"

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @field_validator("SUPPORT_EMAILS", "MANAGER_EMAILS")
    @classmethod
    def validate_email_recipients(cls, recipients: list[str]) -> list[str]:
        normalized = [recipient.strip() for recipient in recipients]
        if not normalized or any(
            not recipient or not EMAIL_PATTERN.fullmatch(recipient)
            for recipient in normalized
        ):
            raise ValueError(
                "Recipient settings must contain one or more valid email addresses"
            )
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_required_settings(self):

        errors = []

        # Gemini validation
        if self.ENABLE_GEMINI:
            if not self.GOOGLE_API_KEY or not self.GOOGLE_API_KEY.strip():
                errors.append("GOOGLE_API_KEY is required")

        # Groq validation
        if self.ENABLE_GROQ:
            if not self.GROQ_API_KEY or not self.GROQ_API_KEY.strip():
                errors.append("GROQ_API_KEY is required")

        # OpenAI validation
        if self.ENABLE_OPENAI:
            if not self.OPENAI_API_KEY or not self.OPENAI_API_KEY.strip():
                errors.append("OPENAI_API_KEY is required")

        # Tavily validation
        if self.ENABLE_WEB_SEARCH:
            if not self.TAVILY_API_KEY or not self.TAVILY_API_KEY.strip():
                errors.append("TAVILY_API_KEY is required")

        # LangSmith validation
        if self.LANGSMITH_TRACING:
            if not self.LANGSMITH_API_KEY or not self.LANGSMITH_API_KEY.strip():
                errors.append("LANGSMITH_API_KEY is required")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n- "
                + "\n- ".join(errors)
            )

        if not any([
            self.ENABLE_GEMINI,
            self.ENABLE_GROQ,
            self.ENABLE_OPENAI
        ]):
            raise ValueError("At least one LLM provider must be enabled")

        return self


settings = Settings()
