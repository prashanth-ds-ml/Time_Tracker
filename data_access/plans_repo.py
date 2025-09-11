# data_access/plans_repo.py
from typing import Any, Dict, List, Optional
import streamlit as st
from core.db import get_db
from core.time_utils import utc_now_naive

db = get_db()

@st.cache_data(ttl=5, show_spinner=False)
def get_week_plan(uid: str, week_key: str) -> Optional[Dict[str, Any]]:
    return db.weekly_plans.find_one({"user": uid, "week_key": week_key})

def upsert_week_plan(uid: str, week_key: str, week_start: str, week_end: str,
                     capacity: Dict[str, int], items: List[Dict[str, Any]]):
    _id = f"{uid}|{week_key}"
    now = utc_now_naive()
    db.weekly_plans.update_one(
        {"_id": _id},
        {"$setOnInsert": {"_id": _id, "user": uid, "created_at": now, "schema_version": 1},
         "$set": {"week_key": week_key, "week_start": week_start, "week_end": week_end,
                  "capacity": capacity, "items": items, "updated_at": now}},
        upsert=True
    )
    st.cache_data.clear()
