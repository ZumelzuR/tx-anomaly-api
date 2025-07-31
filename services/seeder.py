import random
import datetime
import numpy as np
from typing import List, Dict, Tuple

from models.models import LocationEnum, MerchantCategoryEnum, Transaction


class TransactionSeeder:
    def __init__(self):
        self.merchant_categories = list(MerchantCategoryEnum)
        self.locations = [
            ("US", "New York"), ("US", "Los Angeles"), ("US", "Chicago"),
            ("US", "Houston"), ("US", "Phoenix"), ("CA", "Toronto"),
            ("GB", "London"), ("AU", "Sydney"), ("DE", "Berlin"),
            ("FR", "Paris"), ("JP", "Tokyo"), ("IN", "Mumbai"),
            ("CN", "Beijing"), ("BR", "Sao Paulo")
        ]
        
        # Define realistic merchant category patterns
        self.category_patterns = {
            MerchantCategoryEnum.grocery: {'base_amount': 15, 'std_amount': 8, 'frequency': 0.3},
            MerchantCategoryEnum.restaurant: {'base_amount': 45, 'std_amount': 25, 'frequency': 0.25},
            MerchantCategoryEnum.gas: {'base_amount': 25, 'std_amount': 15, 'frequency': 0.15},
            MerchantCategoryEnum.clothing: {'base_amount': 60, 'std_amount': 40, 'frequency': 0.1},
            MerchantCategoryEnum.electronics: {'base_amount': 80, 'std_amount': 50, 'frequency': 0.05},
            MerchantCategoryEnum.utilities: {'base_amount': 120, 'std_amount': 30, 'frequency': 0.05},
            MerchantCategoryEnum.electronics: {'base_amount': 200, 'std_amount': 150, 'frequency': 0.05},
            MerchantCategoryEnum.other: {'base_amount': 35, 'std_amount': 20, 'frequency': 0.05}
        }
        
        # Define location patterns (home vs travel)
        self.location_patterns = {
            'home_location': 0.7,  # 70% of transactions in home location
            'travel_frequency': 0.1,  # 10% chance of travel transaction
            'nearby_locations': 0.2   # 20% chance of nearby location
        }

    def _create_user_profile(self, user_id: str) -> Dict:
        """Create a realistic user profile with consistent patterns."""
        user_seed = hash(user_id) % 1000000
        random.seed(user_seed)
        
        # Home location (primary location for user)
        home_country, home_city = random.choice(self.locations)
        try:
            home_location = LocationEnum[home_country]
        except KeyError:
            home_location = LocationEnum.OTHER
            
        # Nearby locations (same country or neighboring countries)
        nearby_locations = [loc for loc in self.locations if loc[0] == home_country]
        if len(nearby_locations) < 3:
            nearby_locations.extend([loc for loc in self.locations if loc[0] != home_country][:2])
        
        # Primary merchant categories for this user
        primary_categories = random.sample(self.merchant_categories, 3)
        
        # User's spending profile
        spending_profile = {
            'base_amount': random.uniform(30, 150),
            'amount_variability': random.uniform(0.2, 0.6),
            'transaction_frequency_days': random.uniform(1.5, 4.0),
            'weekend_activity': random.uniform(0.3, 0.7),  # Higher activity on weekends
            'evening_activity': random.uniform(0.2, 0.5)   # Evening transaction probability
        }
        
        # Time patterns
        time_patterns = {
            'preferred_hours': random.sample(range(8, 22), 3),  # 3 preferred hours
            'weekend_hours': random.sample(range(10, 23), 4),   # More hours on weekends
            'lunch_hours': [11, 12, 13],  # Lunch time transactions
            'dinner_hours': [18, 19, 20, 21]  # Dinner time transactions
        }
        
        return {
            'home_location': home_location,
            'nearby_locations': nearby_locations,
            'primary_categories': primary_categories,
            'spending_profile': spending_profile,
            'time_patterns': time_patterns,
            'user_seed': user_seed
        }

    def _generate_realistic_amount(self, category: MerchantCategoryEnum, user_profile: Dict) -> float:
        """Generate realistic transaction amounts based on category and user profile."""
        category_name = category.name
        pattern = self.category_patterns.get(category_name, self.category_patterns['other'])
        
        # Base amount from category pattern
        base_amount = pattern['base_amount']
        
        # Adjust based on user's spending profile
        user_multiplier = user_profile['spending_profile']['base_amount'] / 75  # Normalize around 75
        adjusted_base = base_amount * user_multiplier
        
        # Add realistic variability
        variability = pattern['std_amount'] * user_profile['spending_profile']['amount_variability']
        amount = np.random.normal(adjusted_base, variability)
        
        # Ensure reasonable bounds
        amount = max(1.0, min(amount, 2000.0))
        return round(amount, 2)

    def _generate_realistic_location(self, user_profile: Dict, is_travel: bool = False) -> LocationEnum:
        """Generate realistic location based on user's home and travel patterns."""
        random.seed(user_profile['user_seed'] + int(datetime.datetime.now().timestamp() / 86400))
        
        if is_travel:
            # Travel to different country
            travel_locations = [loc for loc in self.locations if loc[0] != user_profile['home_location'].name]
            if travel_locations:
                country_code, _ = random.choice(travel_locations)
                try:
                    return LocationEnum[country_code]
                except KeyError:
                    return LocationEnum.OTHER
        else:
            # Normal location pattern
            rand = random.random()
            if rand < self.location_patterns['home_location']:
                return user_profile['home_location']
            elif rand < self.location_patterns['home_location'] + self.location_patterns['nearby_locations']:
                # Nearby location
                nearby_country, _ = random.choice(user_profile['nearby_locations'])
                try:
                    return LocationEnum[nearby_country]
                except KeyError:
                    return LocationEnum.OTHER
            else:
                # Occasional travel
                travel_locations = [loc for loc in self.locations if loc[0] != user_profile['home_location'].name]
                if travel_locations:
                    country_code, _ = random.choice(travel_locations)
                    try:
                        return LocationEnum[country_code]
                    except KeyError:
                        return LocationEnum.OTHER
        
        return user_profile['home_location']

    def _generate_realistic_timestamp(self, user_profile: Dict, last_timestamp: datetime.datetime = None) -> datetime.datetime:
        """Generate realistic timestamp based on user's time patterns."""
        if last_timestamp is None:
            # Start within the last 6 months
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=180)
            base_date = start_date + datetime.timedelta(days=random.randint(0, 30))
        else:
            # Next transaction follows user's frequency pattern
            avg_days = user_profile['spending_profile']['transaction_frequency_days']
            days_since_last = np.random.exponential(avg_days)
            days_since_last = max(0.1, min(days_since_last, 14))  # Between 2 hours and 2 weeks
            base_date = last_timestamp + datetime.timedelta(days=days_since_last)
        
        # Determine time of day based on patterns
        is_weekend = base_date.weekday() >= 5
        hour_choice = random.random()
        
        if is_weekend:
            # Weekend patterns - more activity throughout the day
            if hour_choice < 0.3:
                hour = random.choice(user_profile['time_patterns']['weekend_hours'])
            else:
                hour = random.randint(10, 22)
        else:
            # Weekday patterns
            if hour_choice < 0.2:
                # Lunch time
                hour = random.choice(user_profile['time_patterns']['lunch_hours'])
            elif hour_choice < 0.4:
                # Dinner time
                hour = random.choice(user_profile['time_patterns']['dinner_hours'])
            elif hour_choice < 0.7:
                # Preferred hours
                hour = random.choice(user_profile['time_patterns']['preferred_hours'])
            else:
                # Other hours
                hour = random.randint(8, 21)
        
        # Add some randomness to minutes and seconds
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return base_date.replace(
            hour=hour,
            minute=minute,
            second=second,
            microsecond=0
        )

    def _select_merchant_category(self, user_profile: Dict, location: LocationEnum) -> MerchantCategoryEnum:
        """Select merchant category based on user preferences and location."""
        # Higher probability for primary categories
        if random.random() < 0.6:
            return random.choice(user_profile['primary_categories'])
        else:
            # Occasional other categories
            return random.choice(self.merchant_categories)

    def generate_normal_transaction(self, user_id: str, last_timestamp: datetime.datetime = None) -> Transaction:
        """
        Generate a transaction that is 'normal' for a given user with realistic patterns.
        """
        # Get or create user profile
        user_profile = self._create_user_profile(user_id)
        
        # Determine if this is a travel transaction (rare)
        is_travel = random.random() < self.location_patterns['travel_frequency']
        
        # Generate location
        location = self._generate_realistic_location(user_profile, is_travel)
        
        # Select merchant category
        merchant_category = self._select_merchant_category(user_profile, location)
        
        # Generate realistic amount
        amount = self._generate_realistic_amount(merchant_category, user_profile)
        
        # Generate timestamp
        timestamp = self._generate_realistic_timestamp(user_profile, last_timestamp)
        
        transaction = Transaction(
            user_id=user_id,
            amount=amount,
            location=location,
            timestamp=timestamp,
            merchant_category=merchant_category
        )
        return transaction

    def generate_synthetic_transactions(self, count: int = 1000) -> List[Transaction]:
        """
        Generate and save multiple 'normal' Transaction documents to MongoDB.
        These transactions are designed to teach the Isolation Forest what is 'normal'.
        """
        transactions = []
        print(f"Generating and saving {count} realistic normal transactions to MongoDB...")

        # Simulate 100 users, each with a normal pattern
        num_users = 100
        user_ids = [str(i + 1) for i in range(num_users)]
        user_last_timestamp = {uid: None for uid in user_ids}
        tx_per_user = count // num_users
        tx_counter = 0

        # For each user, generate their transactions in order
        for uid in user_ids:
            last_ts = None
            for _ in range(tx_per_user):
                tx = self.generate_normal_transaction(uid, last_ts)
                transactions.append(tx)
                last_ts = tx.timestamp
                tx_counter += 1
                if tx_counter % 100 == 0:
                    print(f"Generated {tx_counter} transactions...")

        # If count is not a multiple of num_users, add a few more
        while len(transactions) < count:
            uid = random.choice(user_ids)
            last_ts = user_last_timestamp.get(uid, None)
            tx = self.generate_normal_transaction(uid, last_ts)
            transactions.append(tx)
            user_last_timestamp[uid] = tx.timestamp
            tx_counter += 1
            if tx_counter % 100 == 0:
                print(f"Generated {tx_counter} transactions...")

        # Use bulk insert for efficiency
        Transaction.objects.insert(transactions, load_bulk=True)

        print(f"Successfully saved {len(transactions)} realistic transactions to MongoDB!")
        return transactions

    def generate_rule_triggering_users(self) -> List[Transaction]:
        """
        Generate 5 users that trigger specific fraud detection rules:
        1. User with transaction > $5000
        2. User with transactions in different countries in same hour
        3. User with normal avg amount ~$100 but triggers with $1000 transaction
        4. User with normal avg amount ~$100 but triggers with $1000 transaction
        5. User with normal avg amount ~$100 but triggers with $1000 transaction
        """
        rule_transactions = []
        
        # User 1: High amount transaction (>$5000)
        print("Generating user with high amount transaction (>$5000)...")
        user1_id = "fraud_user_1"
        user1_profile = self._create_user_profile(user1_id)
        
        # Generate normal transactions first
        last_ts = None
        for i in range(10):
            tx = self.generate_normal_transaction(user1_id, last_ts)
            rule_transactions.append(tx)
            last_ts = tx.timestamp
        
        # Add the high amount transaction
        high_amount_tx = Transaction(
            user_id=user1_id,
            amount=6000.0,  # > $5000 threshold
            location=user1_profile['home_location'],
            timestamp=last_ts + datetime.timedelta(hours=2),
            merchant_category=MerchantCategoryEnum.electronics
        )
        rule_transactions.append(high_amount_tx)
        
        # User 2: Different countries in same hour
        print("Generating user with transactions in different countries in same hour...")
        user2_id = "fraud_user_2"
        user2_profile = self._create_user_profile(user2_id)
        
        # Generate normal transactions first
        last_ts = None
        for i in range(8):
            tx = self.generate_normal_transaction(user2_id, last_ts)
            rule_transactions.append(tx)
            last_ts = tx.timestamp
        
        # Add transactions in different countries within 1 hour
        base_time = last_ts + datetime.timedelta(hours=1)
        
        # First transaction in home country
        tx1 = Transaction(
            user_id=user2_id,
            amount=150.0,
            location=user2_profile['home_location'],
            timestamp=base_time,
            merchant_category=MerchantCategoryEnum.electronics
        )
        rule_transactions.append(tx1)
        
        # Second transaction in different country (30 minutes later)
        travel_locations = [loc for loc in self.locations if loc[0] != user2_profile['home_location'].name]
        if travel_locations:
            country_code, _ = random.choice(travel_locations)
            try:
                travel_location = LocationEnum[country_code]
            except KeyError:
                travel_location = LocationEnum.OTHER
        else:
            travel_location = LocationEnum.OTHER
            
        tx2 = Transaction(
            user_id=user2_id,
            amount=200.0,
            location=travel_location,
            timestamp=base_time + datetime.timedelta(minutes=30),
            merchant_category=MerchantCategoryEnum.electronics
        )
        rule_transactions.append(tx2)
        
        # Users 3, 4, 5: Normal avg ~$100 but trigger with $1000 transactions
        for user_num in range(3, 6):
            print(f"Generating user {user_num} with normal avg ~$100 but $1000 transaction...")
            user_id = f"fraud_user_{user_num}"
            user_profile = self._create_user_profile(user_id)
            
            # Generate normal transactions with amounts around $100
            last_ts = None
            for i in range(12):
                # Override the amount generation to keep it around $100
                tx = self.generate_normal_transaction(user_id, last_ts)
                # Modify amount to be around $100
                tx.amount = random.uniform(80, 120)
                rule_transactions.append(tx)
                last_ts = tx.timestamp
            
            # Add the triggering transaction ($1000)
            trigger_tx = Transaction(
                user_id=user_id,
                amount=1000.0,  # 10x the average of ~$100
                location=user_profile['home_location'],
                timestamp=last_ts + datetime.timedelta(hours=3),
                merchant_category=MerchantCategoryEnum.electronics
            )
            rule_transactions.append(trigger_tx)
        
        # Save all rule-triggering transactions
        Transaction.objects.insert(rule_transactions, load_bulk=True)
        print(f"Successfully saved {len(rule_transactions)} rule-triggering transactions to MongoDB!")
        return rule_transactions