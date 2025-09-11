# data_access/goals_repo.py
from typing import Any, Dict, List, Optional
import streamlit as st
from core.db import get_db, USER_ID
from core.time_utils import utc_now_naive

db = get_db()

@st.cache_data(ttl=5, show_spinner=False)
def get_goals(uid: str) -> List[Dict[str, Any]]:
    return list(db.goals.find({"user": uid}).sort("updated_at", -1))

@st.cache_data(ttl=5, show_spinner=False)
def get_goals_map(uid: str) -> Dict[str, Dict[str, Any]]:
    return {g["_id"]: g for g in get_goals(uid)}

def create_goal(user_id: str, title: str, category: str, status: str = "In Progress",
                priority: int = 3, tags: Optional[List[str]] = None) -> str:
    import uuid
    gid = uuid.uuid4().hex[:12]
    now = utc_now_naive()
    doc = {
        "_id": gid, "user": user_id, "title": title.strip(),
        "category": category.strip() or "Other",
        "status": status, "priority": int(priority),
        "is_primary": False,
        "tags": [t.strip() for t in (tags or []) if t.strip()],
        "target_poms": None,
        "created_at": now, "updated_at": now, "schema_version": 1
    }
    db.goals.insert_one(doc)
    st.cache_data.clear()
    return gid

def update_goal(goal_id: str, updates: Dict[str, Any]):
    updates["updated_at"] = utc_now_naive()
    db.goals.update_one({"_id": goal_id, "user": USER_ID}, {"$set": updates})
    st.cache_data.clear()

def delete_goal(goal_id: str) -> bool:
    has_sessions = db.sessions.count_documents({"user": USER_ID, "goal_id": goal_id}) > 0
    if has_sessions:
        return False
    db.weekly_plans.update_many({"user": USER_ID}, {"$pull": {"items": {"goal_id": goal_id}}})
    db.goals.delete_one({"_id": goal_id, "user": USER_ID})
    st.cache_data.clear()
    return True
