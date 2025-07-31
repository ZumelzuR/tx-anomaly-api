import numpy as np
import pandas as pd

from models.models import Transaction
from services.cache import CacheService

# create function to convert numpy types to built-in types for serialization
def convert_numpy_types(val):
    if isinstance(val, np.generic):
        return val.item()
    return val

def update_cache_from_db(cache_service: CacheService):
    user_ids = Transaction.objects().distinct('user_id')
    for user_id in user_ids:
        user_transactions = Transaction.objects(user_id=user_id).order_by('-timestamp')
        transaction_count = user_transactions.count()
        # we skipped users with no transactions
        if transaction_count == 0:
            continue
        data = []
        for tx in user_transactions:
            data.append({
                'amount': tx.amount,
                'timestamp': tx.timestamp,
                'location': tx.location,
                'merchant_category': tx.merchant_category,
            })
        df = pd.DataFrame(data)
        if df.empty:
            continue
        avg_amount = convert_numpy_types(df['amount'].mean())
        std_amount = convert_numpy_types(df['amount'].std())
        if len(df) > 1:
            location_changes = (df['location'] != df['location'].shift(1)).sum()
            location_change_rate = convert_numpy_types(location_changes / len(df))
        else:
            location_change_rate = 0
        if len(df) > 1:
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
            avg_time_between = convert_numpy_types(df['time_diff'].iloc[1:].mean())
        else:
            avg_time_between = 0
        location_count = int(df['location'].nunique())
        location_freq = {str(k): float (v) for k, v in df['location'].value_counts(normalize=True).items()}
        merchant_category_freq = {str(k): float (v) for k, v in df['merchant_category'].value_counts(normalize=True).items()}

        prev_timestamp_same_location = df.groupby('location')['timestamp'].shift(1)

        user_cache_data : UserCacheData = {
            "user_id": user_id,
            "avg_amount": float(avg_amount),
            "std_amount": float(std_amount),
            "last_location": df['location'].iloc[-1],
            "last_timestamp": df['timestamp'].iloc[-1],
            "location_change_rate": float(location_change_rate),
            "avg_time_between": float(avg_time_between) if not np.isnan(avg_time_between) else 1e-10,
            "prev_timestamp_same_location": str(prev_timestamp_same_location.iloc[-1]) if pd.notna(prev_timestamp_same_location.iloc[-1]) else "",
            "transaction_count": int(transaction_count),
            "location_count": int(location_count),
            "location_freq": location_freq,
            "merchant_category_freq": merchant_category_freq,
        }

        cache_service.update_cache(user_cache_data)
