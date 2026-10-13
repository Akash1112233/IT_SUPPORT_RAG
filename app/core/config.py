
import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Application
    app_name: str = "Enterprise IT Support RAG"
    app_env: str = "development"

    # Provider credentials
    groq_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_api_key: str = ""

    # Pinecone configuration
    pinecone_index_name: str = "fde-it-support-rag"
    pinecone_namespace: str = "company-it-kb"

    # Embeddings and generation
    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    groq_model: str = "openai/gpt-oss-20b"

    # Retrieval
    top_k: int = Field(default=4, ge=1, le=20)

    # Workflow limits
    max_retries: int = Field(default=1, ge=0, le=2)
    request_timeout_seconds: int = Field(
        default=45, ge=5, le=180
    )

    # Admin security: no insecure default
    admin_api_key: str = ""

    # Local paths
    audit_db_path: str = str(
        BASE_DIR / "data" / "audit.db"
    )
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(
        BASE_DIR / "data" / "sample_kb"
    )

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_production_settings(self):
        if self.app_env.lower() == "production":
            required = {
                "GROQ_API_KEY": self.groq_api_key,
                "PINECONE_API_KEY": self.pinecone_api_key,
                "ADMIN_API_KEY": self.admin_api_key,
            }

            missing = [
                name
                for name, value in required.items()
                if not value.strip()
            ]

            if missing:
                raise ValueError(
                    "Missing required production settings: "
                    + ", ".join(missing)
                )

            if len(self.admin_api_key) < 32:
                raise ValueError(
                    "ADMIN_API_KEY must be at least "
                    "32 characters in production."
                )

            if self.admin_api_key.lower() in {
                "change-me",
                "admin",
                "password",
            }:
                raise ValueError(
                    "Choose a strong, unique ADMIN_API_KEY."
                )

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()






# from functools import lru_cache
# from pathlib import Path
# from pydantic_settings import BaseSettings,SettingsConfigDict

# BASE_DIR = Path(__file__).resolve().parents[2]


# class Settings(BaseSettings):
#     app_name: str = "Enterprise IT Support Agentic RAG Copilot"
#     app_env: str = "development"
#     groq_api_key: str = ""
#     tavily_api_key: str = ""
#     pinecone_api_key: str = ""
#     pinecone_index_name: str = "fde-it-support-rag"
#     pinecone_namespace: str = "company-it-kb"
#     embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
#     groq_model: str = "openai/gpt-oss-20b"
#     top_k: int = 4
#     max_retries: int = 1
#     admin_api_key: str = "change-me"
#     audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
#     upload_dir: str = str(BASE_DIR / "uploads")
#     sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

#     model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")


# @lru_cache
# def get_settings() -> Settings:
#     return Settings()