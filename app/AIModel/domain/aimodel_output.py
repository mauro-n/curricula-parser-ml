from pydantic import BaseModel

class AiModelOutput(BaseModel):
    cv_class: int
    cv_prob: str