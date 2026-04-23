from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class SeniorityLevel(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


class CompanyInfo(BaseModel):
    name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    size: Optional[str] = Field(description="Company size: startup, scaleup, enterprise")
    stage: Optional[str] = Field(description="Funding stage if startup")


class JobAnalysis(BaseModel):
    job_title: str = Field(description="Exact job title from posting")
    company: CompanyInfo = Field(description="Company details")
    seniority: SeniorityLevel = Field(description="Seniority level of the role")
    required_skills: list[str] = Field(description="Must-have technical skills")
    nice_to_have: list[str] = Field(description="Optional or preferred skills")
    salary_min: Optional[int] = Field(description="Minimum salary in USD annually, null if not mentioned")
    salary_max: Optional[int] = Field(description="Maximum salary in USD annually, null if not mentioned")
    remote_ok: bool = Field(description="Whether remote work is allowed")
    match_score: int = Field(description="Match score 0-100 against an AI Engineer profile with Python, LangChain, RAG, Azure skills")
    match_reason: str = Field(description="One sentence explaining the match score")
    gap_skills: list[str] = Field(description="Skills required but missing from AI Engineer profile")
    summary: str = Field(description="2 sentence summary of the role")

    @field_validator("match_score")
    def validate_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Score must be between 0 and 100")
        return v
