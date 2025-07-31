import mongoengine as me

from enum import Enum

class LocationEnum(str, Enum):
    US = "US"
    CA = "CA"
    GB = "GB"
    DE = "DE"
    FR = "FR"
    IT = "IT"
    ES = "ES"
    AU = "AU"
    JP = "JP"
    IN = "IN"
    OTHER = "OTHER"

class MerchantCategoryEnum(str, Enum):
    grocery = "grocery"
    restaurant = "restaurant"
    gas = "gas"
    clothing = "clothing"
    electronics = "electronics"
    travel = "travel"
    entertainment = "entertainment"
    health = "health"
    utilities = "utilities"
    other = "other"


class Transaction(me.Document):
    user_id = me.StringField(required=True)
    amount = me.FloatField(required=True)
    location = me.StringField(required=True)
    timestamp = me.DateTimeField(required=True)
    merchant_category = me.StringField(required=True)
    is_flagged = me.BooleanField(required=False)

    meta = {
        'collection': 'transactions'
    }
