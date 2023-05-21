import io

import pandas as pd
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.database import Database

data_router = APIRouter()
settings = get_settings()


@data_router.post("/upload/train")
async def upload_train(file: UploadFile = File(...)) -> str:
    """
    Upload data for train
    """
    uploader = Database()
    df = pd.read_csv(file.file)
    file.file.close()
    status = uploader.save_data(data=df, table=settings.train_table)
    return status


@data_router.post("/upload/predict")
async def upload_predict(file: UploadFile = File(...)) -> str:
    """
    Upload data for predict
    """
    uploader = Database()
    df = pd.read_csv(file.file)
    file.file.close()
    status = uploader.save_data(data=df, table=settings.predict_table)
    return status


@data_router.get("/download/predict")
async def download_predict() -> StreamingResponse:
    """
    Download predictions
    """
    downloader = Database()
    query = f"SELECT * FROM {settings.result_table}"
    result = downloader.load_data(query)
    stream = io.StringIO()
    result["data"].to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    return response
