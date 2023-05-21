import asyncio
from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter

from app.predict.service import get_predictions

predict_router = APIRouter()


@predict_router.get("/predict")
async def predict() -> str:
    """
    Run prediction
    """
    with ProcessPoolExecutor() as executor:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, get_predictions)
        return result
