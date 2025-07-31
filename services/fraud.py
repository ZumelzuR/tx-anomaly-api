import os
from typing import List
import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import IsolationForest

from models.models import Transaction
from services.cache import UserCacheData

from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    'high_amout_threshold': float(os.getenv('HIGH_AMOUNT_THRESHOLD', 5000)),
    'multiple_avg_threshold': float(os.getenv('MULTIPLE_AVG_THRESHOLD', 10)),
    'ml_high_threshold': float(os.getenv('ML_HIGH_THRESHOLD', -0.5)),
    'ml_medium_threshold': float(os.getenv('ML_MEDIUM_THRESHOLD', -0.2)),
}

class FraudService:
    def __init__(self):
        if not os.path.exists('ml'):
            os.makedirs('ml')
        self.model = IsolationForest(contamination=0.05)
        self.scaler = None
        self.columns = None
        self.load_model()

    def preprocess_data(self, transactions: List[Transaction]):
        data = [[t.user_id, t.amount, t.location, t.merchant_category, t.timestamp] for t in transactions]
        data = pd.DataFrame(data, columns=['user_id', 'amount', 'location', 'merchant_category', 'timestamp'])

        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.sort_values(by='timestamp', inplace=True)

        # prev_timestamp
        data['prev_timestamp'] = data.groupby('user_id')['timestamp'].shift(1)
        data['time_since_prev'] = (data['timestamp'] - data['prev_timestamp']).dt.total_seconds()
        data['avg_time_between'] = data.groupby('user_id')['time_since_prev'].transform('mean')

        data['interval_deviation'] = data['time_since_prev'] / data['avg_time_between'].replace(0, 1e-10)
        data['last_location_transaction'] = data.groupby('user_id')['location'].shift(1)
        data['location_changed'] = (data['location'] != data['last_location_transaction']).astype(int)

        data['location_change_rate'] = data.groupby('user_id')['location_changed'].transform('mean')
        data['amount'] = data['amount'].astype(float)
        data['location'] = data['location'].astype(str)
        data['merchant_category'] = data['merchant_category'].astype(str)
        data['user_id'] = data['user_id'].astype(str)
        # fillna
        # log amount 
        data['log_amount'] = np.log1p(np.maximum(data['amount'], 0.001))
        # avg amount per user
        data['avg_amount'] = data.groupby('user_id')['amount'].transform('mean')
        # std deviation
        data['std_dev_amount'] = data.groupby('user_id')['amount'].transform('std')
        # log time since prev 
        data['log_time_since_prev'] = np.log1p(np.maximum(data['time_since_prev'], 0.001))

        # transaction count per user
        data['transaction_count'] = data.groupby('user_id')['amount'].transform('count')
        # location count per user
        data['location_count'] = data.groupby('user_id')['location'].transform('nunique')
        
        data['prev_timestamp_same_location'] = data.groupby('user_id')['timestamp'].shift(1)
        # time since same location
        data['time_since_same_location'] = (data['timestamp'] - data['prev_timestamp_same_location']).dt.total_seconds()

        data['log_time_since_same_location'] = np.log1p(np.maximum(data['time_since_same_location'], 0.001))

        # weekday user ratio
        data['hour_of_day'] = data['timestamp'].dt.hour / 24
        data['day_of_week'] = data['timestamp'].dt.dayofweek / 7
        data['is_weekend'] = data['timestamp'].dt.dayofweek > 4

        # encode location and merchant category
        loc_freq = data['location'].value_counts(normalize=True)
        cat_freq = data['merchant_category'].value_counts(normalize=True)
        data['location_freq'] = data['location'].map(loc_freq)
        data['merchant_category_freq'] = data['merchant_category'].map(cat_freq)

        # drop prev_timestamp
        data = data.drop(columns=['user_id', 'timestamp', 'amount', 'location', 'merchant_category', 'time_since_prev','prev_timestamp', 'prev_timestamp_same_location', 'time_since_same_location'])

        # escalate
        # parse all columns to numeric before scaling
        data.dropna(inplace=True)
        data = data.select_dtypes(include=['number'])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
        joblib.dump(scaler, 'ml/scaler.pkl')
        joblib.dump(scaled_data, 'ml/scaled_data.pkl')
        return scaled_data
    
    def train_model(self, transactions: List[Transaction]):
        scaled_data = self.preprocess_data(transactions)
        self.model.fit(scaled_data)
        self.columns = scaled_data.columns

        joblib.dump(self.model, 'ml/model.pkl')
        joblib.dump(self.columns, 'ml/columns.pkl')
        return self.model
    
    # need transaction, last transaction, user_stats from redis, last_loc_transaction
    def process_input(self, transaction: Transaction, last_transaction: Transaction, user_stats: UserCacheData, last_loc_transaction: Transaction):
        data = pd.DataFrame([[transaction.user_id, transaction.amount, transaction.location, transaction.merchant_category, transaction.timestamp]], columns=['user_id', 'amount', 'location', 'merchant_category', 'timestamp'])
        if not hasattr(self, 'columns') or self.columns is None:
            raise Exception("Model not trained")
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # validate if is not first transaction
        if last_transaction:
            data['prev_timestamp'] = pd.to_datetime(last_transaction.timestamp)
            data['time_since_prev'] = (data['timestamp'] - data['prev_timestamp']).dt.total_seconds()
            data['avg_time_between'] = user_stats['avg_time_between'] if user_stats else 3600 # 1hr

            data['interval_deviation'] = float(data['time_since_prev'].iloc[0]) / float(data['avg_time_between'].iloc[0] if data['avg_time_between'].iloc[0] != 0 else 1e-10)
            data['last_location_transaction'] = last_transaction.location
            data['location_changed'] = (data['location'] != data['last_location_transaction']).astype(int)
        else:
            data['time_since_prev'] = pd.NaT
            data['avg_time_between'] = user_stats['avg_time_between'] if user_stats else 3600 # 1hr
            data['interval_deviation'] = 0
            data['last_location_transaction'] = transaction.location
            data['location_changed'] = 1

        if last_loc_transaction:
            data['prev_timestamp_same_location'] = pd.to_datetime(last_loc_transaction.timestamp)
            data['time_since_same_location'] = (data['timestamp'] - data['prev_timestamp_same_location']).dt.total_seconds()
            data['log_time_since_same_location'] = np.log1p(np.maximum(data['time_since_same_location'], 0.001))
        else:
            data['prev_timestamp_same_location'] = pd.NaT
            data['time_since_same_location'] = pd.NaT
            data['log_time_since_same_location'] = 0
        
        data['location_change_rate'] = user_stats['location_change_rate'] if user_stats else 0.1
        # log amount
        data['log_amount'] = np.log1p(np.maximum(data['amount'], 0.001))
        # avg amount
        data['avg_amount'] = user_stats['avg_amount'] if user_stats else 10
        # std deviation
        data['std_dev_amount'] = user_stats['std_dev_amount'] if user_stats else 5
        # log time since prev
        data['log_time_since_prev'] = np.log1p(np.maximum(data['time_since_prev'], 0.0001))

        # transaction count
        data['transaction_count'] = user_stats['transaction_count'] if user_stats else 1
        # location count
        data['location_count'] = user_stats['location_count'] if user_stats else 1
        
        # weekday user ratio
        data['hour_of_day'] = data['timestamp'].dt.hour / 24
        data['day_of_week'] = data['timestamp'].dt.dayofweek / 7
        data['is_weekend'] = data['timestamp'].dt.dayofweek > 4

        if user_stats:
            data['location_freq'] = data['location'].map(user_stats['location_freq'])
            data['merchant_category_freq'] = data['merchant_category'].map(user_stats['merchant_category_freq'])
        else:
            data['location_freq'] = {}
            data['merchant_category_freq'] = {}
            
        data = data[self.columns]
        data = data.fillna(0)
        data_scaled = self.scaler.transform(data)
        # parse to df
        data_scaled = pd.DataFrame(data_scaled, columns=self.columns)
        return data_scaled
    
    
    def predict(self, transaction: Transaction, last_transaction: Transaction, user_stats: UserCacheData, last_loc_transaction: Transaction):
        if not user_stats or not last_transaction:
            return -0.01
        scaled_data = self.process_input(transaction, last_transaction, user_stats, last_loc_transaction)
        prediction = self.model.score_samples(scaled_data)
        return prediction[0]
    
    def load_model(self):
        if os.path.exists('ml/model.pkl'):
            self.model = joblib.load('ml/model.pkl')
            self.scaler = joblib.load('ml/scaler.pkl')
            self.columns = joblib.load('ml/columns.pkl')
    
    # from fb  
    def _rule_based_prediction(self, transaction: Transaction, is_same_location_1h: bool, user_stats: UserCacheData):
        flags = []
        reasons = []

        # if we have user statis we apply the mult avg amount threshold check
        if user_stats and user_stats.get('avg_amount', 0) > 0:
            if transaction.amount > user_stats['avg_amount'] * CONFIG['multiple_avg_threshold']:
                flags.append('medium')
                reasons.append('Transcation amount is multiple times greater than average amount')
        
        # rule amount threshold
        if transaction.amount > CONFIG['high_amout_threshold']:
            flags.append('high')
            reasons.append('Transaction amount is greater than high amount threshold')
        
        # rule location change rate
        if not is_same_location_1h:
            flags.append('high')
            reasons.append('Transaction is not in the same location of the last transactions during the last hour window')
        
        return flags, reasons
    
    def hybrid_prediction(self, transaction: Transaction, last_transaction: Transaction, user_stats: UserCacheData, last_loc_transaction: Transaction, is_same_location_1h: bool):
        flags, reasons = self._rule_based_prediction(transaction, is_same_location_1h, user_stats)

        ml_score = -0.01 # in case we dont have history
        if user_stats:
            ml_score = self.predict(transaction, last_transaction, user_stats, last_loc_transaction)
        
            # if ml score is low, we apply the rule based prediction
            if ml_score < CONFIG['ml_low_threshold']:
                flags, reasons = self._rule_based_prediction(transaction, is_same_location_1h, user_stats)
                if ml_score < CONFIG['ml_high_threshold']:
                    flags.append('high')
                    reasons.append('Ml model flagged transaction as high anomaly')
                elif ml_score > CONFIG['ml_medium_threshold']:
                    flags.append('medium')
                    reasons.append('Ml model flagged transaction as medium anomaly')

        if 'high' in flags:
            risk_level = 'high'
        elif 'medium' in flags:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'reasons': reasons,
            'ml_score': ml_score
        }







        
