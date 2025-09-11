# data_access/users_repo.py
from typing import Any, Dict, Optional
import streamlit as st
from core.db import get_db

db = get_db()

@st.cache_data(ttl=5, show_spinner=False)
def get_user(uid: str) -> Optional[Dict[str, Any]]:
    return db.users.find_one({"_id": uid})
