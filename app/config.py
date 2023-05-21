import pathlib
from functools import lru_cache

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    train_table: str = Field(..., env="TRAIN_TABLE")
    predict_table: str = Field(..., env="PREDICT_TABLE")
    result_table: str = Field(..., env="RESULT_TABLE")
    database: str = Field(..., env="DATABASE_NAME")
    pg_user: str = Field(..., env="POSTGRES_USER")
    pg_password: str = Field(..., env="POSTGRES_PASSWORD")
    db_url: str = Field(..., env="PG_URL")
    db_port: str = Field(..., env="PG_PORT")
    minio_user: str = Field(..., env="MINIO_ROOT_USER")
    minio_password: str = Field(..., env="MINIO_ROOT_PASSWORD")
    minio_url: str = Field(..., env="S3_ENDPOINT")
    minio_bucket: str = Field(..., env="S3_BUCKET")

    class Config:
        env_file = pathlib.Path(__file__).resolve().parent.parent / ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
