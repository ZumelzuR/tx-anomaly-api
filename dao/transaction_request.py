from pydantic import BaseModel
from datetime import datetime

class TransactionRequest(BaseModel):
    user_id: str
    amount: float
    location: str
    merchant_category: str
    timestamp: datetime
