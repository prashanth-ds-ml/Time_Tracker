# app.py
import streamlit as st
from datetime import date
from db import ensure_indexes
from user_management import sidebar_admin, pick_user
from focus import render_focus_timer
from weekly_planner import render_weekly_planner
from analytics import render_analytics
from journal import render_journal

st.set_page_config(
    page_title="Focus Timer • Weekly Priorities",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'About': "Focus Timer - Dynamic Weekly Priority & Pomodoro Management"}
)

def _init_state():
    defaults = {
        "user": None,
        "page": "🎯 Focus Timer",
        "planning_week_date": date.today(),
        "start_time": None,
        "is_break": False,
        "task": "",
        "active_goal_id": None,
        "active_goal_title": "",
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def main_header_and_router():
    _init_state()
    ensure_indexes()

    # Top bar
    c1, c2, c3 = st.columns([2,3,2])
    with c1:
        sel = pick_user()
        if sel != st.session_state.user:
            st.session_state.user = sel
            # clear volatile state
            st.session_state.start_time = None
            st.session_state.is_break = False
            st.session_state.task = ""
            st.session_state.active_goal_id = None
            st.session_state.active_goal_title = ""
            st.experimental_rerun()
    with c2:
        pages = ["🎯 Focus Timer","📅 Weekly Planner","📊 Analytics & Review","🧾 Journal"]
        st.session_state.page = st.selectbox("📍 Navigate", pages,
                                             index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
    with c3:
        with st.expander("➕ Add User"):
            u = st.text_input("Username", key="new_user_input")
            if st.button("Add", key="add_user_btn") and u:
                from db import add_user
                add_user(u.strip())
                st.success("✅ User added!")
                st.rerun()

    st.divider()

    # Sidebar admin (after we know user)
    sidebar_admin(st.session_state.user)

    page = st.session_state.page
    if page == "🎯 Focus Timer":
        render_focus_timer(st.session_state.user)
    elif page == "📅 Weekly Planner":
        render_weekly_planner(st.session_state.user, st.session_state.planning_week_date)
    elif page == "📊 Analytics & Review":
        render_analytics(st.session_state.user)
    elif page == "🧾 Journal":
        render_journal(st.session_state.user)

if __name__ == "__main__":
    main_header_and_router()
