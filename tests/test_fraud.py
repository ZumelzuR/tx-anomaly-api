import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pytest

from models.models import LocationEnum, MerchantCategoryEnum
from services.fraud import FraudService, CONFIG
from services.cache import UserCacheData
from tests.utils.transaction import make_transaction


class TestFraudHybrid:
    """Test hybrid prediction with all combinations of rules and ML scores"""
    
    @pytest.fixture
    def fraud_service(self):
        """Create fraud service with mocked ML model"""
        # Configure before initialization
        os.environ['HIGH_AMOUNT_THRESHOLD'] = '5000'
        os.environ['MULTIPLE_AVG_THRESHOLD'] = '10'
        os.environ['ML_HIGH_THRESHOLD'] = '-0.5'
        os.environ['ML_MEDIUM_THRESHOLD'] = '-0.2'
        os.environ['ML_LOW_THRESHOLD'] = '-0.2'  # Add missing threshold
        
        service = FraudService()
        # Mock the predict method directly to avoid complex ML processing
        service.predict = MagicMock()
        return service
    
    @pytest.fixture
    def base_user_stats(self):
        """Base user statistics for testing"""
        return {
            'avg_amount': 100.0,
            'std_dev_amount': 10.0,
            'avg_time_between': 3600.0,
            'location_change_rate': 0.1,
            'transaction_count': 10,
            'location_count': 2,
            'location_freq': {LocationEnum.US: 0.8, LocationEnum.CA: 0.2},
            'merchant_category_freq': {MerchantCategoryEnum.restaurant: 0.6, MerchantCategoryEnum.grocery: 0.4},
        }
    
    @pytest.fixture
    def base_transaction(self):
        """Base transaction for testing"""
        return make_transaction(
            amount=50.0,
            location=LocationEnum.US,
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def base_last_transaction(self):
        """Base last transaction for testing"""
        return make_transaction(
            amount=45.0,
            location=LocationEnum.US,
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=datetime.now() - timedelta(hours=1)
        )
    
    @pytest.fixture
    def base_last_loc_transaction(self):
        """Base last location transaction for testing"""
        return make_transaction(
            amount=40.0,
            location=LocationEnum.US,
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=datetime.now() - timedelta(hours=2)
        )

    def test_hybrid_no_user_stats_no_history(self, fraud_service, base_transaction):
        """Test hybrid prediction when no user stats and no history"""
        result = fraud_service.hybrid_prediction(
            transaction=base_transaction,
            last_transaction=None,
            user_stats=None,
            last_loc_transaction=None,
            is_same_location_1h=True
        )
        
        # Should only apply rule-based prediction
        assert result['risk_level'] in ['low', 'medium', 'high']
        assert isinstance(result['reasons'], list)
        assert result['ml_score'] == -0.01  # Default when no history

    def test_hybrid_normal_ml_normal_rules(self, fraud_service, base_transaction, base_last_transaction, 
                                          base_user_stats, base_last_loc_transaction):
        """Test hybrid: normal ML score + normal rules"""
        # Mock ML to return normal score (> -0.2)
        fraud_service.predict.return_value = -0.1
        
        result = fraud_service.hybrid_prediction(
            transaction=base_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=True
        )
        
        assert result['risk_level'] == 'low'
        assert len(result['reasons']) == 0  # No rule violations
        assert result['ml_score'] == -0.1

    def test_hybrid_medium_ml_normal_rules(self, fraud_service, base_transaction, base_last_transaction, 
                                          base_user_stats, base_last_loc_transaction):
        """Test hybrid: medium ML score + normal rules"""
        # Mock ML to return medium anomaly score (-0.5 to -0.2)
        fraud_service.predict.return_value = -0.3
        
        result = fraud_service.hybrid_prediction(
            transaction=base_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=True
        )
        
        assert result['risk_level'] == 'medium'
        assert any('Ml model flagged transaction as medium anomaly' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.3

    def test_hybrid_high_ml_normal_rules(self, fraud_service, base_transaction, base_last_transaction, 
                                        base_user_stats, base_last_loc_transaction):
        """Test hybrid: high ML score + normal rules"""
        # Mock ML to return high anomaly score (< -0.5)
        fraud_service.predict.return_value = -0.8
        
        result = fraud_service.hybrid_prediction(
            transaction=base_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=True
        )
        
        assert result['risk_level'] == 'high'
        assert any('Ml model flagged transaction as high anomaly' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.8

    def test_hybrid_normal_ml_high_amount_rule(self, fraud_service, base_last_transaction, 
                                              base_user_stats, base_last_loc_transaction):
        """Test hybrid: normal ML + high amount rule violation"""
        # Mock ML to return normal score
        fraud_service.predict.return_value = -0.1
        
        # Transaction with high amount
        high_amount_transaction = make_transaction(
            amount=6000.0,  # Above threshold
            location=LocationEnum.US,
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=datetime.now()
        )
        
        result = fraud_service.hybrid_prediction(
            transaction=high_amount_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=True
        )
        
        assert result['risk_level'] == 'high'
        assert any('Transaction amount is greater than high amount threshold' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.1

    def test_hybrid_normal_ml_multiple_avg_rule(self, fraud_service, base_last_transaction, 
                                               base_user_stats, base_last_loc_transaction):
        """Test hybrid: normal ML + multiple avg amount rule violation"""
        # Mock ML to return normal score
        fraud_service.predict.return_value = -0.1
        
        # Transaction with amount multiple times greater than average
        high_amount_transaction = make_transaction(
            amount=1500.0,  # 15x average (100)
            location=LocationEnum.US,
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=datetime.now()
        )
        
        result = fraud_service.hybrid_prediction(
            transaction=high_amount_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=True
        )
        
        assert result['risk_level'] == 'medium'
        assert any('multiple times greater than average amount' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.1

    def test_hybrid_normal_ml_location_change_rule(self, fraud_service, base_transaction, base_last_transaction, 
                                                  base_user_stats, base_last_loc_transaction):
        """Test hybrid: normal ML + location change rule violation"""
        # Mock ML to return normal score
        fraud_service.predict.return_value = -0.1
        
        result = fraud_service.hybrid_prediction(
            transaction=base_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=False  # Location change violation
        )
        
        assert result['risk_level'] == 'high'
        assert any('not in the same location of the last transactions during the last hour window' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.1

    def test_hybrid_high_ml_high_amount_rule(self, fraud_service, base_last_transaction, 
                                            base_user_stats, base_last_loc_transaction):
        """Test hybrid: high ML + high amount rule violation"""
        # Mock ML to return high anomaly score
        fraud_service.predict.return_value = -0.8
        
        # Transaction with high amount
        high_amount_transaction = make_transaction(
            amount=6000.0,  # Above threshold
            location=LocationEnum.US,
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=datetime.now()
        )
        
        result = fraud_service.hybrid_prediction(
            transaction=high_amount_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=True
        )
        
        assert result['risk_level'] == 'high'
        assert any('Transaction amount is greater than high amount threshold' in reason for reason in result['reasons'])
        assert any('Ml model flagged transaction as high anomaly' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.8

    def test_hybrid_medium_ml_multiple_avg_rule(self, fraud_service, base_last_transaction, 
                                               base_user_stats, base_last_loc_transaction):
        """Test hybrid: medium ML + multiple avg amount rule violation"""
        # Mock ML to return medium anomaly score
        fraud_service.predict.return_value = -0.3
        
        # Transaction with amount multiple times greater than average
        high_amount_transaction = make_transaction(
            amount=1500.0,  # 15x average (100)
            location=LocationEnum.US,
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=datetime.now()
        )
        
        result = fraud_service.hybrid_prediction(
            transaction=high_amount_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=True
        )
        
        assert result['risk_level'] == 'medium'
        assert any('multiple times greater than average amount' in reason for reason in result['reasons'])
        assert any('Ml model flagged transaction as medium anomaly' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.3

    def test_hybrid_high_ml_location_change_rule(self, fraud_service, base_transaction, base_last_transaction, 
                                                base_user_stats, base_last_loc_transaction):
        """Test hybrid: high ML + location change rule violation"""
        # Mock ML to return high anomaly score
        fraud_service.predict.return_value = -0.8
        
        result = fraud_service.hybrid_prediction(
            transaction=base_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=False  # Location change violation
        )
        
        assert result['risk_level'] == 'high'
        assert any('not in the same location of the last transactions during the last hour window' in reason for reason in result['reasons'])
        assert any('Ml model flagged transaction as high anomaly' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.8

    def test_hybrid_all_rules_violated(self, fraud_service, base_last_transaction, 
                                      base_user_stats, base_last_loc_transaction):
        """Test hybrid: all rules violated (high amount, multiple avg, location change)"""
        # Mock ML to return normal score
        fraud_service.predict.return_value = -0.1
        
        # Transaction violating all rules
        problematic_transaction = make_transaction(
            amount=8000.0,  # Above high threshold and multiple avg
            location=LocationEnum.CA,  # Different location
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=datetime.now()
        )
        
        result = fraud_service.hybrid_prediction(
            transaction=problematic_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=False  # Location change violation
        )
        
        assert result['risk_level'] == 'high'  # Should be high due to high amount and location change
        assert any('Transaction amount is greater than high amount threshold' in reason for reason in result['reasons'])
        assert any('not in the same location of the last transactions during the last hour window' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.1

    def test_hybrid_high_ml_all_rules_violated(self, fraud_service, base_last_transaction, 
                                              base_user_stats, base_last_loc_transaction):
        """Test hybrid: high ML + all rules violated"""
        # Mock ML to return high anomaly score
        fraud_service.predict.return_value = -0.8
        
        # Transaction violating all rules
        problematic_transaction = make_transaction(
            amount=8000.0,  # Above high threshold and multiple avg
            location=LocationEnum.CA,  # Different location
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=datetime.now()
        )
        
        result = fraud_service.hybrid_prediction(
            transaction=problematic_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=False  # Location change violation
        )
        
        assert result['risk_level'] == 'high'
        assert any('Transaction amount is greater than high amount threshold' in reason for reason in result['reasons'])
        assert any('not in the same location of the last transactions during the last hour window' in reason for reason in result['reasons'])
        assert any('Ml model flagged transaction as high anomaly' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.8

    def test_hybrid_no_user_stats_with_rules(self, fraud_service, base_transaction):
        """Test hybrid: no user stats but with rule violations"""
        # Transaction with high amount (no user stats for multiple avg check)
        high_amount_transaction = make_transaction(
            amount=6000.0,  # Above threshold
            location=LocationEnum.US,
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=datetime.now()
        )
        
        result = fraud_service.hybrid_prediction(
            transaction=high_amount_transaction,
            last_transaction=None,
            user_stats=None,
            last_loc_transaction=None,
            is_same_location_1h=True
        )
        
        assert result['risk_level'] == 'high'
        assert any('Transaction amount is greater than high amount threshold' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.01

    def test_hybrid_edge_case_ml_thresholds(self, fraud_service, base_transaction, base_last_transaction, 
                                           base_user_stats, base_last_loc_transaction):
        """Test hybrid: edge cases around ML thresholds"""
        # Test exactly at ML high threshold (-0.5) - should trigger medium flag
        fraud_service.predict.return_value = -0.5
        
        result = fraud_service.hybrid_prediction(
            transaction=base_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=True
        )
        
        assert result['risk_level'] == 'medium'  # Should trigger medium flag
        assert any('Ml model flagged transaction as medium anomaly' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.5

        # Test exactly at ML medium threshold (-0.2) - should not trigger any ML flag
        fraud_service.predict.return_value = -0.2
        
        result = fraud_service.hybrid_prediction(
            transaction=base_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=True
        )
        
        assert result['risk_level'] == 'low'  # Should not trigger ML flag
        assert result['ml_score'] == -0.2

    def test_hybrid_risk_level_priority(self, fraud_service, base_transaction, base_last_transaction, 
                                       base_user_stats, base_last_loc_transaction):
        """Test that high risk level takes priority over medium"""
        # Mock ML to return medium anomaly score
        fraud_service.predict.return_value = -0.3
        
        # Transaction with high amount (high risk) but medium ML
        high_amount_transaction = make_transaction(
            amount=6000.0,  # Above threshold
            location=LocationEnum.US,
            merchant_category=MerchantCategoryEnum.restaurant,
            timestamp=datetime.now()
        )
        
        result = fraud_service.hybrid_prediction(
            transaction=high_amount_transaction,
            last_transaction=base_last_transaction,
            user_stats=base_user_stats,
            last_loc_transaction=base_last_loc_transaction,
            is_same_location_1h=True
        )
        
        # Should be high due to high amount rule, not medium from ML
        assert result['risk_level'] == 'high'
        assert any('Transaction amount is greater than high amount threshold' in reason for reason in result['reasons'])
        assert any('Ml model flagged transaction as medium anomaly' in reason for reason in result['reasons'])
        assert result['ml_score'] == -0.3
