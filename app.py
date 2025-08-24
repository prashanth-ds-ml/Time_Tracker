# app.py
import streamlit as st
from datetime import date

from user_management import (
    get_all_users, add_user, ensure_indexes, export_sessions_csv
)
from focus import render_focus_timer
from weekly_planner import render_weekly_planner
from analytics import render_analytics_review
from journal import render_journal
from user_management import now_ist  # for default dates

# =========================
# CONFIG (entry point)
# =========================
st.set_page_config(
    page_title="Focus Timer • Weekly Priorities",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'About': "Focus Timer - Dynamic Weekly Priority & Pomodoro Management"}
)

# =========================
# Session state helpers
# =========================
def init_session_state():
    defaults = {
        "start_time": None,
        "is_break": False,
        "task": "",
        "user": None,
        "page": "🎯 Focus Timer",
        "planning_week_date": now_ist().date(),
        "review_week_date": now_ist().date(),
        "active_goal_id": None,
        "active_goal_title": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_runtime_state_for_user():
    st.session_state.start_time = None
    st.session_state.is_break = False
    st.session_state.task = ""
    st.session_state.active_goal_id = None
    st.session_state.active_goal_title = ""

init_session_state()

# =========================
# Router
# =========================
def main_header_and_router():
    # Establish user before any use
    users = get_all_users()
    if not users:
        add_user("prashanth")
        users = get_all_users()
    if st.session_state.user is None or st.session_state.user not in users:
        st.session_state.user = users[0]

    # Sidebar admin
    st.sidebar.markdown("### ⚙️ Admin")
    if st.sidebar.button("Initialize Mongo Indexes"):
        ensure_indexes()
    export_sessions_csv(st.session_state.user)

    # Top controls
    c1, c2, c3 = st.columns([2,3,2])
    with c1:
        try:
            idx = users.index(st.session_state.user) if st.session_state.user in users else 0
        except Exception:
            idx = 0
        sel = st.selectbox("👤 User", users, index=idx, key="user_select_header")
        if sel != st.session_state.user:
            st.session_state.user = sel
            reset_runtime_state_for_user()
            # Clear data caches on switch
            from user_management import get_user_sessions, get_user_settings, fetch_goals
            get_user_sessions.clear(); get_user_settings.clear(); fetch_goals.clear()
            st.rerun()
    with c2:
        pages = ["🎯 Focus Timer", "📅 Weekly Planner", "📊 Analytics & Review", "🧾 Journal"]
        current = st.session_state.get("page", pages[0])
        st.session_state.page = st.selectbox("📍 Navigate", pages, index=pages.index(current) if current in pages else 0)
    with c3:
        with st.expander("➕ Add User"):
            u = st.text_input("Username", key="new_user_input")
            if st.button("Add", key="add_user_btn") and u:
                if add_user(u.strip()):
                    st.session_state.user = u.strip()
                    reset_runtime_state_for_user()
                    st.success("✅ User added!")
                    st.rerun()
                else:
                    st.warning("User already exists!")

    st.divider()
    page = st.session_state.page
    if page == "🎯 Focus Timer":
        render_focus_timer(st.session_state.user)
    elif page == "📅 Weekly Planner":
        render_weekly_planner(st.session_state.user, st.session_state.planning_week_date)
    elif page == "📊 Analytics & Review":
        render_analytics_review(st.session_state.user, st.session_state.review_week_date)
    elif page == "🧾 Journal":
        render_journal(st.session_state.user)

if __name__ == "__main__":
    main_header_and_router()
