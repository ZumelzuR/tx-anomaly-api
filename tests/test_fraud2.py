import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from models.models import LocationEnum
from services.fraud import FraudService
from tests.utils.transaction import make_transaction

@pytest.fixture
def fraud_service():
    service = FraudService()
    # service.normalize_ml_score = lambda x: 42 if x is not None else None
    return service

class TestFraudRules:
    def test_rule_based_high_amount(self, fraud_service: FraudService):
        tx = make_transaction(amount=6000)
        last_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(hours=1))
        last_location_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(hours=1))
        user_stats = {
            "avg_amount": 100,
            "std_dev_amount": 10,
            "avg_time_between": 3600,
            "location_change_rate": 0.1,
            "transaction_count": 10,
            "location_count": 2,
            "location_freq": {"US": 1.0},
            "merchant_category_freq": {"retail": 1.0}
        }
        result = fraud_service.hybrid_prediction(tx, last_tx, user_stats, last_location_tx, True)
        assert result["risk_level"] == "high"
        assert "Transaction amount is greater than" in " ".join(result["reasons"])

    def test_rule_based_high_amount_no_last_transaction(self, fraud_service):
        tx = make_transaction(amount=6000)
        last_tx = None
        last_location_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(hours=1))
        user_stats = {
            "avg_amount": 100,
            "std_dev_amount": 10,
            "avg_time_between": 3600,
            "location_change_rate": 0.1,
            "transaction_count": 10,
            "location_count": 2,
            "location_freq": {"US": 1.0},
            "merchant_category_freq": {"retail": 1.0}
        }
        result = fraud_service.hybrid_prediction(tx, last_tx, user_stats, last_location_tx, True)
        assert result["risk_level"] == "high"
        assert "Transaction amount is greater than" in " ".join(result["reasons"])

    def test_rule_based_high_amount_no_user_stats(self, fraud_service):
        tx = make_transaction(amount=6000)
        last_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(hours=1))
        last_location_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(hours=1))
        user_stats = None
        result = fraud_service.hybrid_prediction(tx, last_tx, user_stats, last_location_tx, True)
        assert result["risk_level"] == "high"
        assert "Transaction amount is greater than" in " ".join(result["reasons"])

    def test_rule_based_high_amount_no_last_transaction_no_user_stats(self, fraud_service):
        tx = make_transaction(amount=6000)
        last_tx = None
        last_location_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(hours=1))
        user_stats = None
        result = fraud_service.hybrid_prediction(tx, last_tx, user_stats, last_location_tx, True)
        assert result["risk_level"] == "high"
        assert "Transaction amount is greater than" in " ".join(result["reasons"])

    # test rule when user have low amount and no transactions
    def test_rule_based_low_amount_no_transactions(self, fraud_service):
        tx = make_transaction(amount=100)
        last_tx = None
        last_location_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(hours=1))
        user_stats = None
        result = fraud_service.hybrid_prediction(tx, last_tx, user_stats, True, last_location_tx)
        assert result["risk_level"] == "low"

    def test_rule_based_low_amount_no_transactions_no_user_stats(self, fraud_service):
        tx = make_transaction(amount=100)
        last_tx = None
        last_location_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(hours=1))
        user_stats = None
        result = fraud_service.hybrid_prediction(tx, last_tx, user_stats, last_location_tx, True)
        assert result["risk_level"] == "low"

    def test_rule_based_not_same_location_1hr(self, fraud_service):
        tx = make_transaction(amount=100, location=LocationEnum.US)
        last_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(minutes=50), location=LocationEnum.FR)
        last_location_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(hours=1))
        user_stats = None
        result = fraud_service.hybrid_prediction(tx, last_tx, user_stats, last_location_tx, False)
        assert result["risk_level"] == "high"

    def test_rule_based_transaction_amount_multiple_avg_amount_threshold(self, fraud_service):
        tx = make_transaction(amount=1500)  # 15x average (100), but below high threshold (5000)
        last_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(hours=1))
        last_location_tx = make_transaction(amount=100, timestamp=datetime.now() - timedelta(hours=1))
        user_stats = {
            "avg_amount": 100,
            "std_dev_amount": 10,
            "avg_time_between": 3600,
            "location_change_rate": 0.1,
            "transaction_count": 10,
            "location_count": 2,
            "location_freq": {"US": 1.0},
            "merchant_category_freq": {"retail": 1.0}
        }
        result = fraud_service.hybrid_prediction(tx, last_tx, user_stats, last_location_tx, True)
        assert result["risk_level"] == "high"