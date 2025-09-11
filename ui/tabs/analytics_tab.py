# ui/tabs/analytics_tab.py
import pandas as pd
import streamlit as st
from core.db import get_db
from core.time_utils import week_dates_list
from data_access.plans_repo import get_week_plan
from data_access.sessions_repo import aggregate_pe_by_goal_bucket
from data_access.sessions_repo import list_today_sessions

db = get_db()

def render_analytics_tab(USER_ID: str):
    st.header("📊 Consistency Insights")

    weeks_sessions = sorted(db.sessions.distinct("week_key", {"user": USER_ID, "t": "W"}))
    weeks_plans    = sorted(db.weekly_plans.distinct("week_key", {"user": USER_ID}))
    all_weeks      = sorted(set(weeks_sessions) | set(weeks_plans))
    count_weeks    = len(all_weeks)

    if count_weeks == 0:
        st.info("No data yet. Log some sessions or save a weekly plan to see analytics.")
        return

    # Keep your existing weekly summary table & charts
    last_n = st.slider("Show last N weeks", min_value=1, max_value=min(12, count_weeks), value=min(6, count_weeks))
    weeks_view = all_weeks[-last_n:]

    rows = []
    for W in weeks_view:
        planW = get_week_plan(USER_ID, W)
        planned = sum(int(it.get("planned_current", 0)) for it in (planW.get("items", []) if planW else []))
        pe_doc = next(iter(db.sessions.aggregate([
            {"$match": {"user": USER_ID, "week_key": W, "t": "W"}},
            {"$group": {"_id": None, "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
        ])), None)
        actual_pe = float(pe_doc["pe"]) if pe_doc else 0.0
        focus_total = db.sessions.count_documents({"user": USER_ID, "week_key": W, "t": "W", "kind": {"$ne": "activity"}})
        deep = db.sessions.count_documents({"user": USER_ID, "week_key": W, "t": "W", "kind": {"$ne": "activity"}, "deep_work": True})
        deep_pct = (deep / focus_total * 100.0) if focus_total else 0.0
        valid_breaks = db.sessions.count_documents({
            "user": USER_ID, "week_key": W, "t": "B", "dur_min": {"$gte": 4},
            "$or": [{"skipped": {"$exists": False}}, {"skipped": {"$ne": True}}]
        })
        break_pct = (min(valid_breaks, focus_total) / focus_total * 100.0) if focus_total else 0.0
        pe_by_mode = {row["_id"]: row["pe"] for row in db.sessions.aggregate([
            {"$match": {"user": USER_ID, "week_key": W, "t": "W"}},
            {"$group": {"_id": "$goal_mode", "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
        ])}
        unplan_pct = ((float(pe_by_mode.get("custom", 0.0)) / actual_pe) * 100.0) if actual_pe else 0.0
        adh = ((min(actual_pe, planned) / planned) * 100.0) if planned else 0.0
        rows.append({
            "week": W, "planned": planned, "actual_pe": round(actual_pe, 1),
            "adherence_pct": round(adh, 1), "deep_pct": round(deep_pct, 1),
            "break_pct": round(break_pct, 1), "unplanned_pct": round(unplan_pct, 1)
        })

    dfw = pd.DataFrame(rows)
    st.dataframe(dfw, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Adherence %")
        st.line_chart(dfw.set_index("week")["adherence_pct"])
    with c2:
        st.subheader("Deep Work %")
        st.bar_chart(dfw.set_index("week")["deep_pct"])

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Unplanned %")
        st.bar_chart(dfw.set_index("week")["unplanned_pct"])
    with c4:
        st.subheader("Actual PE")
        st.bar_chart(dfw.set_index("week")["actual_pe"])

    st.divider()

    # Daily minutes for selected week (minutes only)
    idx_last = max(0, len(weeks_view) - 1)
    sel_week = st.selectbox("Pick a week for daily minutes", weeks_view, index=idx_last)
    st.subheader("📆 Daily Minutes (selected week)")
    days = week_dates_list(sel_week)
    daily_rows = []
    for d in days:
        mins_doc = next(iter(db.sessions.aggregate([
            {"$match": {"user": USER_ID, "date_ist": d, "t":"W"}},
            {"$group": {"_id": None, "mins": {"$sum": "$dur_min"}}}
        ])), None)
        daily_rows.append({"date": d, "minutes": int(mins_doc["mins"]) if mins_doc else 0})
    dfd = pd.DataFrame(daily_rows).set_index("date")
    st.bar_chart(dfd["minutes"])
