import uvicorn
from fastapi import FastAPI

from app.config import get_settings
from app.data.router import data_router
from app.exceptions import exception_handler
from app.predict.router import predict_router
from app.router import main_router

config = get_settings()

version = "v1"

app = FastAPI(
    title="Fast API ML",
    description="Data upload, predict and download results",
    version="0.0.1",
    contact={
        "name": "Pavel",
        "email": "test@example.com",
    })

app.include_router(main_router)
app.include_router(data_router, prefix=f"/api/{version}")
app.include_router(predict_router, prefix=f"/api/{version}")
app.exception_handler(exception_handler)

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8011, reload=True, log_level="debug")
