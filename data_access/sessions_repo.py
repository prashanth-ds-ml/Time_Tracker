# data_access/sessions_repo.py
from typing import Any, Dict, List, Optional
import streamlit as st
from core.db import get_db
from core.time_utils import week_key_from_datestr

db = get_db()

@st.cache_data(ttl=5, show_spinner=False)
def total_day_pe(uid: str, date_ist: str) -> float:
    pipeline = [
        {"$match": {"user": uid, "date_ist": date_ist, "t": "W"}},
        {"$group": {"_id": None, "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
    ]
    doc = next(iter(db.sessions.aggregate(pipeline)), None)
    return float(doc["pe"]) if doc else 0.0

@st.cache_data(ttl=5, show_spinner=False)
def aggregate_pe_by_goal_bucket(uid: str, week_key: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    pipeline = [
        {"$match": {"user": uid, "week_key": week_key, "t": "W", "goal_id": {"$exists": True, "$ne": None}}},
        {"$group": {
            "_id": {"goal_id": "$goal_id", "alloc_bucket": "$alloc_bucket"},
            "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
    ]
    for row in db.sessions.aggregate(pipeline):
        gid = row["_id"]["goal_id"]
        bucket = row["_id"]["alloc_bucket"] or "current"
        pe = float(row["pe"])
        if gid not in out:
            out[gid] = {"current": 0.0, "backlog": 0.0}
        if bucket not in out[gid]:
            out[gid][bucket] = 0.0
        out[gid][bucket] += pe
    return out

@st.cache_data(ttl=5, show_spinner=False)
def list_today_sessions(uid: str, date_ist: str) -> List[Dict[str, Any]]:
    return list(db.sessions.find({"user": uid, "date_ist": date_ist}).sort("started_at_ist", 1))

def delete_last_today_session(uid: str, date_ist: str) -> Optional[str]:
    last = db.sessions.find({"user": uid, "date_ist": date_ist}).sort("started_at_ist", -1).limit(1)
    last = next(iter(last), None)
    if last:
        db.sessions.delete_one({"_id": last["_id"]})
        st.cache_data.clear()
        return last["_id"]
    return None
