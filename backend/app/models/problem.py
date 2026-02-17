"""Problem-related request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProblemSummary(BaseModel):
    id: str
    name: str
    description: str
    dimension: int


class ParameterSpec(BaseModel):
    name: str
    default: Any
    type: str = Field(description="Python type name of the default value")


class ProblemDetail(ProblemSummary):
    parameters: dict[str, Any]
    parameter_schema: list[ParameterSpec]


class ProblemListResponse(BaseModel):
    items: list[ProblemSummary]


class ProblemDetailResponse(BaseModel):
    item: ProblemDetail
