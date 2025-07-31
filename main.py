from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
import mongoengine as me

from fastapi import Request
from fastapi.responses import JSONResponse
import os
from apscheduler.schedulers.background import BackgroundScheduler

import uvicorn

from dao.transaction_request import TransactionRequest

from dotenv import load_dotenv

from models.models import LocationEnum, MerchantCategoryEnum, Transaction
from services.cache import CacheService
from services.fraud import FraudService
from services.job import update_cache_from_db
from services.seeder import TransactionSeeder

load_dotenv()   

MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "fraud_detection")
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

me.connect(db=MONGO_DB_NAME, host=MONGO_HOST, port=MONGO_PORT)

app = FastAPI()
fraud_service = FraudService()
transaction_seeder = TransactionSeeder()
cache_service = CacheService(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def job_update_cache_from_db():
    update_cache_from_db(cache_service)

def scheduler_update_cache():
    scheduler = BackgroundScheduler()
    scheduler.add_job(job_update_cache_from_db, 'interval', minutes=10)
    scheduler.start()

@app.post("/transactions")
async def submit_transaction(request: TransactionRequest):
    try:
        try: 
            location_enum = LocationEnum[request.location]
        except KeyError:
            location_enum = LocationEnum.OTHER
        try:
            merchant_category_enum = MerchantCategoryEnum[request.merchant_category]
        except KeyError:
            merchant_category_enum = MerchantCategoryEnum.other
        
        tx = Transaction(
            user_id=request.user_id,
            amount=request.amount,
            location=location_enum,
            merchant_category=merchant_category_enum,
            timestamp=request.timestamp
        )
        
        is_same_location_1h = True # for first users transctions
        last_transacrion = Transaction.objects(user_id=tx.user_id).order_by('-timestamp').first()
        # we will check is_same_location via db query
        if Transaction.objects(user_id=tx.user_id).count() > 0:
            one_hour_ago = tx.timestamp - timedelta(hours=1)
            is_same_location_1h = Transaction.objects(user_id=tx.user_id, location__ne=tx.location, timestamp__gte=one_hour_ago, timestamp__lte=tx.timestamp).order_by('-timestamp').count() > 0
        
        last_location_transaction = Transaction.objects(user_id=tx.user_id, location=tx.location).order_by('-timestamp').first()
        user_stats = cache_service.get_cache(tx.user_id)
        # predict
        result = fraud_service.hybrid_prediction(tx, last_transacrion, user_stats, last_location_transaction, is_same_location_1h)
        tx.is_flagged = result['risk_level'] == 'high'
        tx.save()
        # For now, just return an empty response
        return {
            'risk_level': result['risk_level'],
            'reasons': result['reasons'],
            'ml_score': result['ml_score']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transactions/flagged")
async def get_flagged_transactions(user_id: str, limit: int =100):
    try:
        txs = Transaction.objects(is_flagged=True, user_id=user_id).order_by('-timestamp').limit(limit)
        flagged_list = [
            {
                'user_id': tx.user_id,
                'amount': tx.amount,
                'location': tx.location,
                'merchant_category': tx.merchant_category,
                'timestamp': tx.timestamp,
                'is_flagged': tx.is_flagged
            }
            for tx in txs
        ]
        return {
            'flagged_transactions': flagged_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    transactions = transaction_seeder.generate_synthetic_transactions(1000)
    trasaction_rules = transaction_seeder.generate_rule_triggering_users()
    scheduler_update_cache()

    if fraud_service.scaler is None or fraud_service.model is None:
        print("Training model...")
        fraud_service.train_model(transactions)
        print("Model trained!")
    else:
        print("Model already trained!")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


