"""Problem catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.models.problem import (
    ParameterSpec,
    ProblemDetail,
    ProblemDetailResponse,
    ProblemListResponse,
    ProblemSummary,
)
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


@router.get("/{problem_id}", response_model=ProblemDetailResponse)
async def get_problem(problem_id: str) -> ProblemDetailResponse:
    """Get details and parameters for a specific problem type."""
    defaults = default_problem_configs()
    if problem_id not in defaults:
        raise HTTPException(status_code=404, detail="Problem not found")

    problem = create_problem(problem_id, **defaults[problem_id])
    parameter_schema = [
        ParameterSpec(name=k, default=v, type=type(v).__name__)
        for k, v in defaults[problem_id].items()
    ]
    detail = ProblemDetail(
        id=problem_id,
        name=problem.name,
        description=problem.description,
        dimension=problem.dimension,
        parameters=problem.get_parameters(),
        parameter_schema=parameter_schema,
    )
    return ProblemDetailResponse(item=detail)
