from pydantic import BaseModel

class Input(BaseModel):
    message: str