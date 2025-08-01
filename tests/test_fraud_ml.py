from datetime import datetime, timedelta
import os
import pickle

import pytest

from models.models import LocationEnum, MerchantCategoryEnum
from services.fraud import FraudService
from tests.utils.transaction import make_transaction

ML_MODEL_PATH = os.path.join('ml', 'model.pkl')

config = {
    'high_amout_threshold': 5000,
    'multiple_avg_threshold': 10,
    'ml_high_threshold': -0.52,
    'ml_medium_threshold': -0.2,
}

@pytest.fixture
def ml_model():
    with open(ML_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

@pytest.fixture
def fraud_service():
    return FraudService()

class TestFraudML:
    def test_ml_model_normal_behavior(self, fraud_service: FraudService):
        """
        Normal user: several transactions with similar amounts, locations, and categories.
        ML score should NOT be anomaly (score >= -0.5).
        """
        user_id = "user_normal"
        now = datetime.now()
        # Simulate 10 normal transactions
        transactions = [
            make_transaction(
                amount=100 + i,  # small variation
                location=LocationEnum.US,
                merchant_category=MerchantCategoryEnum.restaurant,
                timestamp=now - timedelta(days=10 - i)
            )
            for i in range(10)
        ]
        user_stats = {
            'avg_amount': 105,
            'std_dev_amount': 3,
            'avg_time_between': 3600 * 24,  # 1 day
            'location_change_rate': 0.0,
            'transaction_count': 10,
            'location_count': 1,
            'location_freq': {LocationEnum.US: 10},
            'merchant_category_freq': {MerchantCategoryEnum.restaurant: 10},
        }
        # Test a new normal transaction
        transaction = make_transaction(
            amount=104,
            location=LocationEnum.US,
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=now
        )
        last_transaction = transactions[-1]
        last_loc_transaction = transactions[-1]
        score = fraud_service.predict(transaction, last_transaction, user_stats, last_loc_transaction)
        assert score >= config['ml_high_threshold'], f"Normal case: Score {score} should NOT be anomaly (>= {config['ml_high_threshold']})"

    def test_ml_model_high_amount_anomaly(self, fraud_service: FraudService):
        """
        Anomalous case: sudden huge amount compared to user's average.
        ML score should be anomaly (score < -0.5).
        """
        now = datetime.now()
        # Simulate 10 normal transactions
        transactions = [
            make_transaction(
                amount=100 + i,
                location=LocationEnum.US,
                merchant_category=MerchantCategoryEnum.grocery,
                timestamp=now - timedelta(days=10 - i)
            )
            for i in range(10)
        ]
        user_stats = {
            'avg_amount': 105,
            'std_dev_amount': 3,
            'avg_time_between': 3600 * 24,
            'location_change_rate': 0.0,
            'transaction_count': 10,
            'location_count': 1,
            'location_freq': {LocationEnum.US: 10},
            'merchant_category_freq': {MerchantCategoryEnum.grocery: 10},
        }
        # Anomalous transaction: huge amount
        transaction = make_transaction(
            amount=10000,
            location=LocationEnum.US,
            merchant_category=MerchantCategoryEnum.grocery,
            timestamp=now
        )
        last_transaction = transactions[-1]
        last_loc_transaction = transactions[-1]
        score = fraud_service.predict(transaction, last_transaction, user_stats, last_loc_transaction)
        assert score < config['ml_high_threshold'], f"High amount anomaly: Score {score} should be anomaly (< {config['ml_high_threshold']})"

    def test_ml_model_location_change_anomaly(self, fraud_service: FraudService):
        """
        Anomalous case: sudden location change for the user.
        ML score should be anomaly (score < -0.5).
        """
        user_id = "user_location_change"
        now = datetime.now()
        # Simulate 10 normal transactions in US
        transactions = [
            make_transaction(
                amount=80 + i,
                location=LocationEnum.US,
                merchant_category=MerchantCategoryEnum.clothing,
                timestamp=now - timedelta(days=10 - i)
            )
            for i in range(10)
        ]
        user_stats = {
            'avg_amount': 100,
            'std_dev_amount': 5,
            'avg_time_between': 36,
            'location_change_rate': 0.0,
            'transaction_count': 50,
            'location_count': 1,
            'location_freq': {LocationEnum.US: 10},
            'merchant_category_freq': {MerchantCategoryEnum.clothing: 1},
        }
        # Anomalous transaction: different country
        transaction = make_transaction(
            amount=9110,
            location=LocationEnum.JP,
            merchant_category=MerchantCategoryEnum.clothing,
            timestamp=now
        )
        last_transaction = make_transaction(
            amount=10,
            location=LocationEnum.JP,
            merchant_category=MerchantCategoryEnum.clothing,
            timestamp=now - timedelta(hours=30),
        )
        last_loc_transaction = make_transaction(
            amount=10,
            location=LocationEnum.JP,
            merchant_category=MerchantCategoryEnum.clothing,
            timestamp=now - timedelta(hours=30),
        )
        score = fraud_service.predict(transaction, last_transaction, user_stats, last_loc_transaction)
        assert score < config['ml_high_threshold'], f"Location anomaly: Score {score} should be anomaly (< {config['ml_high_threshold']})"

    def test_ml_model_merchant_category_anomaly(self, fraud_service: FraudService):
        """
        Anomalous case: sudden new merchant category for the user.
        ML score should be anomaly (score < -0.5).
        """
        user_id = "user_category_change"
        now = datetime.now()
        # Simulate 10 normal transactions in 'restaurant'
        transactions = [
            make_transaction(
                amount=60 + i,
                location=LocationEnum.FR,
                merchant_category=MerchantCategoryEnum.restaurant,
                timestamp=now - timedelta(days=10 - i)
            )
            for i in range(10)
        ]
        user_stats = {
            'avg_amount': 65,
            'std_dev_amount': 2,
            'avg_time_between': 3600 * 24,
            'location_change_rate': 0.0,
            'transaction_count': 10,
            'location_count': 1,
            'location_freq': {LocationEnum.FR: 10},
            'merchant_category_freq': {MerchantCategoryEnum.restaurant: 10},
        }
        # Anomalous transaction: new merchant category
        transaction = make_transaction(
            amount=70,
            location=LocationEnum.FR,
            merchant_category=MerchantCategoryEnum.electronics,
            timestamp=now
        )
        last_transaction = transactions[-1]
        last_loc_transaction = transactions[-1]
        score = fraud_service.predict(transaction, last_transaction, user_stats, last_loc_transaction)
        assert score < config['ml_high_threshold'], f"Merchant category anomaly: Score {score} should be anomaly (< {config['ml_high_threshold']})"

    def test_ml_model_multiple_anomalies(self, fraud_service: FraudService):
        """
        Anomalous case: huge amount, new location, and new merchant category.
        ML score should be anomaly (score < -0.5).
        """
        user_id = "user_multi_anomaly"
        now = datetime.now()
        # Simulate 10 normal transactions
        transactions = [
            make_transaction(
                amount=120 + i,
                location=LocationEnum.DE,
                merchant_category=MerchantCategoryEnum.gas,
                timestamp=now - timedelta(days=10 - i)
            )
            for i in range(10)
        ]
        user_stats = {
            'avg_amount': 125,
            'std_dev_amount': 2,
            'avg_time_between': 3600 * 24,
            'location_change_rate': 0.0,
            'transaction_count': 10,
            'location_count': 1,
            'location_freq': {LocationEnum.DE: 10},
            'merchant_category_freq': {MerchantCategoryEnum.gas: 10},
        }
        # Anomalous transaction: new location, new category, huge amount
        transaction = make_transaction(
            amount=10000,
            location=LocationEnum.IN,
            merchant_category=MerchantCategoryEnum.travel,
            timestamp=now
        )
        last_transaction = transactions[-1]
        last_loc_transaction = transactions[-1]
        score = fraud_service.predict(transaction, last_transaction, user_stats, last_loc_transaction)
        assert score < config['ml_high_threshold'], f"Multiple anomaly: Score {score} should be anomaly (< {config['ml_high_threshold']})"
