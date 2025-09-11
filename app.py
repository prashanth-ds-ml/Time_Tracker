# app.py
import streamlit as st

from core.config import APP_TITLE, PAGE_ICON
from core.db import get_db, USER_ID
from core.time_utils import today_iso, week_key_from_datestr
from data_access.goals_repo import get_goals_map
from data_access.plans_repo import get_week_plan
from services.planner_service import derive_auto_plan_from_active, week_dates_list
from ui.tabs.timer_tab import render_timer_tab
from ui.tabs.planner_tab import render_planner_tab
from ui.tabs.analytics_tab import render_analytics_tab

st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout="wide")

db = get_db()
today = today_iso()
default_week_key = week_key_from_datestr(today)

goals_map = get_goals_map(USER_ID)
default_plan = get_week_plan(USER_ID, default_week_key)
if not default_plan:
    cap_auto, items_auto = derive_auto_plan_from_active(USER_ID, default_week_key)
    default_plan = {
        "_id": f"{USER_ID}|{default_week_key}",
        "user": USER_ID,
        "week_key": default_week_key,
        "week_start": week_dates_list(default_week_key)[0],
        "week_end": week_dates_list(default_week_key)[-1],
        "capacity": cap_auto,
        "items": items_auto,
        "derived": True
    }

# Sidebar
st.sidebar.header("⚙️ Connection")
st.sidebar.write(f"**DB:** `{db.name}`")
st.sidebar.write(f"**User:** `{USER_ID}`")

with st.sidebar.expander("🔍 Diagnostics", expanded=False):
    try:
        info = db.command("buildInfo")
        st.write("Connected:", True)
        st.write("Mongo Version:", info.get("version"))
        st.write("Collections:", sorted(db.list_collection_names()))
    except Exception as e:
        st.error(f"Diagnostics failed: {e}")

st.sidebar.subheader(f"📅 Week {default_week_key}")
if default_plan and default_plan.get("items"):
    st.sidebar.caption(f"{default_plan.get('week_start')} → {default_plan.get('week_end')}")
    cap = default_plan.get("capacity", {})
    st.sidebar.write(f"Capacity: **{cap.get('total', 0)}** poms")
else:
    st.sidebar.info("No weekly plan (and no active goals) for this ISO week yet.")

# Tabs
tab1, tab2, tab3 = st.tabs(["⏱️ Focus Lab", "🗂️ Traction Plan", "📊 Consistency Insights"])

with tab1:
    render_timer_tab(USER_ID, default_week_key, goals_map, default_plan)

with tab2:
    render_planner_tab(USER_ID)

with tab3:
    render_analytics_tab(USER_ID)
