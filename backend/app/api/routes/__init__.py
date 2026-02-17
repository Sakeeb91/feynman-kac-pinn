"""Route modules for API v1."""

from fastapi import APIRouter

from .problems import router as problems_router
from .results import router as results_router
from .simulations import router as simulations_router

api_router = APIRouter()
api_router.include_router(problems_router, prefix="/problems", tags=["problems"])
api_router.include_router(simulations_router, prefix="/simulations", tags=["simulations"])
api_router.include_router(results_router, prefix="/results", tags=["results"])

__all__ = ["api_router"]
