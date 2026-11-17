from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from datetime import datetime, timezone
import uuid

class Memory(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    content: str
    memory_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance_score: float = 1.0
    tags: List[str] = Field(default_factory=list)
    related_ids: List[str] = Field(default_factory=list)
