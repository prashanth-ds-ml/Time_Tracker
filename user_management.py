# user_management.py
import streamlit as st
import pandas as pd
from db import (
    list_users, add_user, ensure_indexes,
    get_sessions_df, registry_defaults, update_registry_defaults
)
from datetime import datetime

def export_sessions_csv_ui(user: str):
    df = get_sessions_df(user)
    if df.empty:
        st.info("No sessions to export.")
        return
    out = df.sort_values("date")
    st.download_button(
        "⬇️ Export Sessions (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"{user}_sessions.csv",
        mime="text/csv"
    )

def sidebar_admin(user: str):
    st.sidebar.markdown("### ⚙️ Admin")
    if st.sidebar.button("Ensure Mongo Indexes"):
        ensure_indexes()
        st.sidebar.success("Indexes ensured.")
    export_sessions_csv_ui(user)

def pick_user() -> str:
    users = list_users()
    if not users:
        add_user("prashanth")
        users = list_users()
    idx = 0
    if "user" in st.session_state and st.session_state.user in users:
        idx = users.index(st.session_state.user)
    sel = st.selectbox("👤 User", users, index=idx, key="user_select_header")
    return sel

def defaults_block(user: str):
    st.caption("Defaults drive weekly capacity.")
    defs = registry_defaults(user)
    c1, c2 = st.columns(2)
    with c1:
        wp = st.number_input("Weekday avg (poms)", 0, 12, value=int(defs.get("weekday_poms", 3)))
    with c2:
        we = st.number_input("Weekend avg (poms)", 0, 12, value=int(defs.get("weekend_poms", 5)))
    if st.button("💾 Save Defaults"):
        update_registry_defaults(user, wp, we)
        st.success("Defaults updated.")
        st.experimental_rerun()
