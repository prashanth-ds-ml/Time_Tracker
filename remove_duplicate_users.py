from pymongo import MongoClient
from datetime import datetime
import streamlit as st

MONGO_URI = st.secrets["mongo_uri"]
DB_NAME = "time_tracker_db"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db["users"]

def remove_duplicate_prashanth():
    users = list(users_collection.find({"username": "prashanth"}))
    if len(users) > 1:
        # Keep the first, remove the rest
        for user in users[1:]:
            users_collection.delete_one({"_id": user["_id"]})
        print(f"Removed {len(users)-1} duplicate 'prashanth' users.")
    else:
        print("No duplicate 'prashanth' users found.")

if __name__ == "__main__":
    remove_duplicate_prashanth()
