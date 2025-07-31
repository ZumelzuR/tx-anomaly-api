import redis
import json

from typing import TypedDict, Dict

class UserCacheData(TypedDict, total=False):
    user_id: str
    avg_amount: float
    std_dev_amount: float
    last_location: str
    last_timestamp: str
    location_change_rate: float
    avg_time_between: float
    prev_timestamp_same_loc:str
    transaction_count: int
    location_count: int
    location_freq: Dict[str, float]
    merchant_category_freq: Dict[str, float]
    # Add any other fields as needed


class CacheService:
    def __init__(self, host: str, port: int, db: int):
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def __save(self, user_id, user_data):
        data = user_data.copy()
        for k,v in data.items():
            if isinstance(v, dict):
                data[k] = json.dumps(v)
        self.redis_client.hmset(f"user:{user_id}", data)


    def get_cache(self, user_id):
        # Fetch existing user stats from Redis
        user_key = f"user:{user_id}"
        data = self.redis_client.hgetall(user_key)
        if not data:
            return None
        
        # oarse keys avg_amount_user, std_dev_amount_user, location_count_user, avg_timebetweem
        for key in ['avg_amount', 'std_dev_amount', 'location_change_rate', 'avg_time_between']:
            if key in data:
                try:
                    data[key] = float(data[key])
                except Exception as e:
                    data[key] = 0.0

        for key in ['transaction_count', 'location_count']:
            if key in data:
                try:
                    data[key] = int(data[key])
                except Exception as e:
                    data[key] = 0
        
        for key in ["location_freq", "merchant_category_freq"]:
            if key in data:
                try:
                    data[key] = json.loads(data[key])
                except Exception as e:
                    data[key] = {}

        return data

    def _update(self, user_data: UserCacheData):
        user_key = f"user:{user_data['user_id']}"
        user_data = user_data.copy()
        self.__save(user_key, user_data)
        
        
        




