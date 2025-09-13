from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

mongo_client = MongoClient(os.getenv('FYLLO_MONGO_URI'))
db = mongo_client["database"]
device_collection = db["device"]
field_data_collection = db["FinalFieldData"]

def get_plot_ids():
    filter_query = {
        "deviceType": "NERO_INFINITY_UNIT",
        "installationDate": {"$gte": datetime(2025, 7, 10)},
        "isAssigned": True
    }
    return [
        doc["plotId"] for doc in device_collection.find(filter_query, {"plotId": 1, "_id": 0})
    ]

def get_field_data(plot_id):
    filter_query = {
        "plotId": plot_id,
        "timestamp": {"$gte": datetime(2025, 7, 10)}
    }
    projection = {
        "_id": 1,
        "deviceId": 1,
        "plotId": 1,
        "farmUserId": 1,
        "timestamp": 1,
        "moisture1": 1,
        "moisture2": 1,
        "I1": 1,
        "I2": 1
    }
    docs = list(field_data_collection.find(filter_query, projection))
    return docs