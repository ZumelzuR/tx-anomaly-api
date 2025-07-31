from datetime import datetime
from models.models import LocationEnum, MerchantCategoryEnum, Transaction

def make_transaction(amount: float, location: str = LocationEnum.US, merchant_category: str = MerchantCategoryEnum.restaurant, timestamp = None):
    if timestamp is None:
        timestamp = datetime.now()
    return Transaction(
        user_id="user1test",
        amount=amount,
        location=location,
        merchant_category=merchant_category,
        timestamp=timestamp,
    )