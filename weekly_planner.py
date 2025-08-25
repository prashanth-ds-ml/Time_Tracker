# weekly_planner.py
import streamlit as st
import pandas as pd
from datetime import timedelta
from typing import Dict
from db import (
    now_ist, week_bounds,
    registry_defaults, update_registry_defaults,
    list_registry_goals, upsert_registry_goal, update_registry_goal_status,
    get_or_create_week_plan, save_week_plan, get_sessions_df
)

def _proportional_allocation(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    if total <= 0 or not weights:
        return {gid: 0 for gid in weights}
    total_w = sum(max(1, int(w)) for w in weights.values())
    raw = {gid: (max(1, int(w))/total_w)*total for gid, w in weights.items()}
    base = {gid: int(v) for gid, v in raw.items()}
    diff = total - sum(base.values())
    if diff != 0:
        fracs = sorted(((gid, raw[gid]-int(raw[gid])) for gid in raw), key=lambda x: x[1], reverse=True)
        i = 0
        while diff != 0 and fracs:
            g = fracs[i % len(fracs)][0]
            base[g] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            i += 1
    return base

def _capacity_for_week(user: str, week_start):
    defs = registry_defaults(user)
    weekday_poms = int(defs.get("weekday_poms", 3))
    weekend_poms = int(defs.get("weekend_poms", 5))
    wd = sum(1 for i in range(7) if (week_start + timedelta(days=i)).weekday() < 5)
    we = 7 - wd
    return weekday_poms * wd + weekend_poms * we

def _last_week_table(user: str, week_start):
    prev_ws, prev_we = week_start - timedelta(days=7), week_start - timedelta(days=1)
    prev_plan = get_or_create_week_plan(user, prev_ws)
    alloc = prev_plan.get("allocations_by_goal", {}) or {}
    if not alloc:
        st.info("No plan existed last week.")
        return
    df = get_sessions_df(user)
    if df.empty:
        df = pd.DataFrame(columns=["date","time","pomodoro_type","duration","user","goal_id","task","category"])
        df["date"] = pd.to_datetime(df["date"])
    df["date_only"] = df["date"].dt.date
    mask = (df["date_only"]>=prev_ws) & (df["date_only"]<=prev_we) & (df["pomodoro_type"]=="Work")
    actual = df[mask].groupby("goal_id").size().to_dict()
    rows = []
    titles = {g["goal_id"]: g.get("title","(goal)") for g in prev_plan.get("goals_embedded", [])}
    for gid, planned in alloc.items():
        rows.append({
            "Goal": titles.get(gid, "(missing)"),
            "Planned": int(planned),
            "Actual": int(actual.get(gid, 0)),
            "Carry": max(0, int(planned) - int(actual.get(gid, 0))),
        })
    view = pd.DataFrame(rows).sort_values("Goal")
    st.dataframe(view, use_container_width=True, hide_index=True)

def render_weekly_planner(user: str, picked_date):
    st.header("📅 Weekly Planner")

    # Date
    week_start, week_end = week_bounds(picked_date)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**Week:** {week_start} → {week_end}")
    with c2:
        defs = registry_defaults(user)
        wp = st.number_input("Weekday avg", 0, 12, value=int(defs.get("weekday_poms",3)))
    with c3:
        we = st.number_input("Weekend avg", 0, 12, value=int(defs.get("weekend_poms",5)))
    if st.button("💾 Save Capacity Defaults"):
        update_registry_defaults(user, wp, we)
        st.success("Defaults updated")
        st.rerun()

    st.divider()

    # Goals & Priority Weights (editable any time)
    st.subheader("🎯 Goals & Priority")
    goals = list_registry_goals(user, statuses=["New","In Progress","On Hold","Completed"])
    with st.expander("➕ Add / Update Goal", expanded=False):
        g_title = st.text_input("Title")
        g_type  = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0)
        g_stat  = st.selectbox("Status", ["In Progress","New","On Hold","Completed","Archived"], index=0)
        g_prio  = st.select_slider("Priority", options=[1,2,3], value=2, help="High=3, Medium=2, Low=1")
        if st.button("💾 Save Goal"):
            if g_title.strip():
                upsert_registry_goal(user, g_title.strip(), g_type, g_stat, g_prio)
                st.success("Saved goal")
                st.rerun()
            else:
                st.warning("Please enter a title")

    if not goals:
        st.info("Add 3–4 goals to plan your week.")
        return

    # editable priorities
    cols = st.columns(min(4, len(goals)))
    updated = {}
    for i, g in enumerate(goals):
        with cols[i % len(cols)]:
            st.write(f"**{g.get('title','(goal)')}**")
            val = int(g.get("priority_band", 2))
            val = max(1, min(3, val))
            updated[g["goal_id"]] = st.select_slider("Priority", options=[1,2,3], value=val, key=f"prio_{g['goal_id']}")
    if st.button("💾 Update Priorities"):
        for gid, v in updated.items():
            upsert_registry_goal(user, next(x["title"] for x in goals if x["goal_id"]==gid),
                                 goal_type=next(x["goal_type"] for x in goals if x["goal_id"]==gid),
                                 status=next(x["status"] for x in goals if x["goal_id"]==gid),
                                 priority_band=int(v),
                                 target_poms=next(x.get("target_poms",0) for x in goals if x["goal_id"]==gid),
                                 goal_id=gid)
        st.success("Priorities updated.")
        st.rerun()

    st.divider()

    # Allocation
    st.subheader("🧮 Allocate Weekly Pomodoros")
    plan = get_or_create_week_plan(user, week_start)
    capacity = _capacity_for_week(user, week_start)
    weights = {g["goal_id"]: int(updated.get(g["goal_id"], g.get("priority_band",2))) for g in goals}
    auto = _proportional_allocation(capacity, weights)

    edited = {}
    cols2 = st.columns(min(4, len(goals)))
    for i, g in enumerate(goals):
        with cols2[i % len(cols2)]:
            default_val = int(plan.get("allocations_by_goal", {}).get(g["goal_id"], auto[g["goal_id"]]))
            edited[g["goal_id"]] = st.number_input(g.get("title","(goal)"), 0, capacity, value=default_val, step=1, key=f"alloc_{g['goal_id']}")

    sum_alloc = sum(edited.values())
    if sum_alloc != capacity:
        st.warning(f"Allocations total {sum_alloc}, capacity is {capacity}.")
        if st.button("Normalize to capacity"):
            edited = _proportional_allocation(capacity, {k: max(1,v) for k,v in edited.items()})
            for gid, v in edited.items():
                st.session_state[f"alloc_{gid}"] = v
            st.rerun()

    if st.button("📌 Save / Update Weekly Plan", type="primary"):
        save_week_plan(user, week_start, edited)
        st.success("Weekly plan saved.")
        st.rerun()

    st.divider()

    # Rollover from last week
    with st.expander("↪️ Rollover unfinished from last week", expanded=False):
        prev_ws, prev_we = week_start - timedelta(days=7), week_start - timedelta(days=1)
        prev_plan = get_or_create_week_plan(user, prev_ws)
        prev_alloc = prev_plan.get("allocations_by_goal", {}) or {}
        if not prev_alloc:
            st.info("No previous plan found.")
        else:
            if st.button(f"Rollover from {prev_ws} → {prev_we}"):
                df = get_sessions_df(user)
                if df.empty:
                    df = pd.DataFrame(columns=["date","time","pomodoro_type","duration","user","goal_id","task","category"])
                    df["date"] = pd.to_datetime(df["date"])
                df["date_only"] = df["date"].dt.date
                mask = (df["date_only"]>=prev_ws) & (df["date_only"]<=prev_we) & (df["pomodoro_type"]=="Work")
                actual_prev = df[mask].groupby("goal_id").size().to_dict()
                carry = {gid: max(0, int(planned) - int(actual_prev.get(gid, 0))) for gid, planned in prev_alloc.items()}
                carry = {gid: v for gid, v in carry.items() if v > 0}
                if not carry:
                    st.info("Nothing to rollover 🎉")
                else:
                    curr = dict(plan.get("allocations_by_goal", {}))
                    for gid, add in carry.items():
                        curr[gid] = int(curr.get(gid, 0)) + int(add)
                    save_week_plan(user, week_start, curr)
                    st.success("Rolled over unfinished pomodoros.")
                    st.rerun()

    st.divider()
    st.subheader("📜 Last Week Recap")
    _last_week_table(user, week_start)

    st.divider()
    st.subheader("✅ Close-out (set status) & optional manual rollover")
    # Simple status edit table
    for g in goals:
        cols = st.columns([3,2,2])
        with cols[0]:
            st.write(f"**{g.get('title','(goal)')}**")
        with cols[1]:
            status = st.selectbox("Status", ["In Progress","Completed","On Hold","Archived"],
                                  index=["In Progress","Completed","On Hold","Archived"].index(g.get("status","In Progress")),
                                  key=f"status_{g['goal_id']}")
        with cols[2]:
            if st.button("Apply", key=f"apply_{g['goal_id']}"):
                update_registry_goal_status(user, g["goal_id"], status)
                st.success("Status updated.")
                st.rerun()
