from fastapi import APIRouter
from fastapi.responses import RedirectResponse

main_router = APIRouter()


@main_router.get("/")
async def redirect():
    return RedirectResponse("/docs")
