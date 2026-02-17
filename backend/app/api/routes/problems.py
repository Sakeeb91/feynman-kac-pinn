"""Problem catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.models.problem import ProblemListResponse, ProblemSummary
from ml.problems import available_problems, create_problem, default_problem_configs

router = APIRouter()


def _build_problem_summary(problem_id: str) -> ProblemSummary:
    problem = create_problem(problem_id, **default_problem_configs()[problem_id])
    return ProblemSummary(
        id=problem_id,
        name=problem.name,
        description=problem.description,
        dimension=problem.dimension,
    )


@router.get("", response_model=ProblemListResponse)
async def list_problems() -> ProblemListResponse:
    """List available PDE problems for simulation."""
    items = [_build_problem_summary(problem_id) for problem_id in available_problems()]
    return ProblemListResponse(items=items)
