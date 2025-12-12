from typing import List, Literal, Dict, Optional, Tuple
from pydantic import BaseModel, Field


class HearsayAnswer(BaseModel):
    answer: str = Field(..., description="Answer text (Yes or No)")
    explanation: int = Field(..., description="A 2-3 sentence explanation for the answer")

