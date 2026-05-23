# schemas.py - Created for Casting Defect Detection Project
from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
    image: bytes = Field(..., description="Image bytes for defect detection")
    user_id: str = Field(..., example="123456789")
    filename: Optional[str] = Field(None, description="Original filename")

class PredictionResponse(BaseModel):
    is_defect: bool
    confidence: float
    model_version: str
    processing_time_ms: float