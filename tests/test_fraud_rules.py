from datetime import datetime, timedelta
from unittest.mock import MagicMock
import pytest

from services.fraud import FraudService
from tests.utils.transaction import make_transaction

@pytest.fixture
def fraud_service():
    service = FraudService()
    # we mock ml model prediction to get normal scores
    service.model = MagicMock()
    service.model.score_samples = MagicMock(return_value=[0.1])
    return service

class TestFraudRules:
    def test_rule_high_amount(self, fraud_service: FraudService):
        transaction = make_transaction(amount=10000)
        user_stats = {
            'avg_amount': 10000,
            'std_amount': 10,
            'avg_time_between': 3600,
            'location_change_rate': 0.5,
            "transaction_count": 10,
            "location_count": 2,
            "location_freq": {"US": 1},
            "merchant_category_freq": {"restaurant": 1},
        }
        flags, reasons = fraud_service._rule_based_prediction(transaction,True , user_stats)
        print(flags, reasons)
        assert flags == ['high']
        assert reasons == ['Transaction amount is greater than high amount threshold']

    def test_rule_multiple_avg_amount(self, fraud_service: FraudService):
        # Transaction amount is multiple times greater than average amount
        transaction = make_transaction(amount=10000)
        user_stats = {
            'avg_amount': 100,
            'std_amount': 10,
            'avg_time_between': 3600,
            'location_change_rate': 0.5,
            "transaction_count": 10,
            "location_count": 2,
            "location_freq": {"US": 1},
            "merchant_category_freq": {"restaurant": 1},
        }
        flags, reasons = fraud_service._rule_based_prediction(transaction, True, user_stats)
        # Should flag as medium for multiple avg, not high (unless high threshold is also crossed)
        assert 'medium' in flags
        assert any("multiple times greater than average amount" in r for r in reasons)

    def test_rule_not_same_location_1h(self, fraud_service: FraudService):
        # Transaction is not in the same location as the last transaction
        transaction = make_transaction(amount=50)
        user_stats = {
            'avg_amount': 40,
            'std_amount': 5,
            'avg_time_between': 3600,
            'location_change_rate': 0.5,
            "transaction_count": 10,
            "location_count": 2,
            "location_freq": {"US": 1},
            "merchant_category_freq": {"restaurant": 1},
        }
        # is_same_location_1h = False triggers both medium and high flags for location
        flags, reasons = fraud_service._rule_based_prediction(transaction, False, user_stats)
        assert 'high' in flags
        assert any("not in the same location of the last transactions during the last hour window" in r for r in reasons)

    def test_rule_no_user_stats(self, fraud_service: FraudService):
        # If user_stats is None, only high amount and location rules apply
        transaction = make_transaction(amount=10000)
        user_stats = None
        flags, reasons = fraud_service._rule_based_prediction(transaction, False, user_stats)
        # Should flag high for amount, and both medium/high for location
        assert 'high' in flags
        assert any("greater than high amount threshold" in r for r in reasons)
    
