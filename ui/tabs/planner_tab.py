# ui/tabs/planner_tab.py
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta

from core.time_utils import now_ist, week_key_from_date
from data_access.plans_repo import get_week_plan, upsert_week_plan
from data_access.goals_repo import get_goals_map
from services.planner_service import get_user_capacity_defaults, get_rank_weight_map, derive_auto_plan_from_active, prev_week_key
from data_access.sessions_repo import aggregate_pe_by_goal_bucket

def render_planner_tab(USER_ID: str):
    st.header("🗂️ Traction Plan")
    st.subheader("📅 Build / Edit Weekly Plan")

    default_monday = (now_ist() - timedelta(days=now_ist().isoweekday() - 1)).date()
    wk_start_date = st.date_input("Week start", value=default_monday, key="wk_start_date")
    wk_end_date = wk_start_date + timedelta(days=6)
    wk = week_key_from_date(wk_start_date)
    st.caption(f"Week range: **{wk_start_date.isoformat()} → {wk_end_date.isoformat()}** • ISO key: **{wk}**")

    wkday_default, wkend_default = get_user_capacity_defaults(USER_ID)
    colWCap1, colWCap2 = st.columns(2)
    with colWCap1:
        wkday = st.number_input("Weekday poms (per day)", 0, 20, value=wkday_default)
    with colWCap2:
        wkend = st.number_input("Weekend poms (per day)", 0, 30, value=wkend_default)
    total_capacity = wkday*5 + wkend*2
    st.caption(f"Total capacity: **{total_capacity}** poms.")

    existing = get_week_plan(USER_ID, wk)
    rwm = get_rank_weight_map(USER_ID)

    goals_map_full = get_goals_map(USER_ID)
    active_goals = [g for g in goals_map_full.values() if g.get("status") == "In Progress"]
    existing_items = {it["goal_id"]: it for it in (existing.get("items", []) if existing else [])}

    if not existing_items:
        _, derived_items = derive_auto_plan_from_active(USER_ID, wk)
        base_items = derived_items
    else:
        base_items = []
        for g in active_goals:
            gid = g["_id"]
            ex = existing_items.get(gid)
            if ex:
                base_items.append(ex)
            else:
                rank = int(g.get("priority", 3))
                base_items.append({
                    "goal_id": gid, "priority_rank": rank, "weight": int(rwm.get(str(rank), 1)),
                    "planned_current": 0, "backlog_in": 0, "total_target": 0,
                    "status_at_plan": "In Progress", "close_action": None, "notes": None
                })

    rows = []
    for it in base_items:
        gid = it["goal_id"]; g = goals_map_full.get(gid, {})
        rows.append({
            "goal_id": gid,
            "title": g.get("title",""),
            "category": g.get("category",""),
            "rank": str(int(it.get("priority_rank", int(g.get("priority",3))))),
            "planned_current": int(it.get("planned_current", 0)),
            "backlog_in": int(it.get("backlog_in", 0)),
            "total_target": int(it.get("total_target", int(it.get("planned_current",0))+int(it.get("backlog_in",0)))),
            "notes": it.get("notes") or ""
        })

    if rows:
        df = pd.DataFrame(rows)
        edited = st.data_editor(
            df,
            column_config={
                "title": st.column_config.TextColumn("Goal"),
                "category": st.column_config.TextColumn("Category"),
                "rank": st.column_config.SelectboxColumn("Priority (1=high)", options=["1","2","3","4","5"], width="small"),
                "planned_current": st.column_config.NumberColumn("Planned (current)", step=1, min_value=0),
                "backlog_in": st.column_config.NumberColumn("Backlog In", step=1, min_value=0),
                "total_target": st.column_config.NumberColumn("Total Target", step=1, min_value=0, disabled=True),
                "notes": st.column_config.TextColumn("Notes"),
            },
            use_container_width=True, hide_index=True, num_rows="fixed"
        )
    else:
        st.info("No active goals found. Add goals below to plan your week.")
        edited = pd.DataFrame([])

    colA1, colA2, colA3 = st.columns([1,1,1])
    auto_go = colA1.button("⚖️ Auto-allocate")
    clear_plan = colA2.button("🧹 Clear planned_current")
    save_plan = colA3.button("💾 Commit Week")

    if auto_go and not edited.empty:
        m = edited.copy()
        if m.empty or total_capacity <= 0:
            st.warning("Need at least one active goal and capacity > 0.")
        else:
            m["weight"] = m["rank"].map(lambda r: int(rwm.get(str(r), 1)))
            weights_sum = max(m["weight"].sum(), 1)
            shares = (m["weight"] / weights_sum) * total_capacity
            base = np.floor(shares).astype(int)
            left = total_capacity - base.sum()
            frac = shares - base
            order = np.argsort(-frac.values)
            for i in range(int(left)):
                base.iloc[order[i]] += 1
            edited["planned_current"] = base.values
            edited["backlog_in"] = edited["backlog_in"].fillna(0).astype(int)
            edited["total_target"] = (edited["planned_current"].astype(int) + edited["backlog_in"].astype(int)).astype(int)
            st.success("Auto-allocation applied.")

    if clear_plan and not edited.empty:
        edited["planned_current"] = 0
        edited["total_target"] = edited["backlog_in"].astype(int)
        st.info("Cleared plan allocations.")

    if not edited.empty:
        planned_sum = int(edited["planned_current"].sum())
        st.caption(f"Planned current sum: **{planned_sum}** / capacity **{total_capacity}**")
        if planned_sum != total_capacity:
            st.warning("Sum of planned_current should equal capacity total.")
        else:
            st.success("Planned_current matches capacity total ✅")

    if save_plan and not edited.empty:
        items = []
        for _, r in edited.iterrows():
            pc = int(r["planned_current"]); bi = int(r["backlog_in"]); rank = int(r["rank"])
            items.append({
                "goal_id": r["goal_id"],
                "priority_rank": rank,
                "weight": int(rwm.get(str(rank), 1)),
                "planned_current": pc,
                "backlog_in": bi,
                "total_target": pc + bi,
                "status_at_plan": "In Progress",
                "close_action": None,
                "notes": r.get("notes") or None
            })
        cap = {"weekday": int(wkday), "weekend": int(wkend), "total": int(total_capacity)}
        upsert_week_plan(USER_ID, wk, wk_start_date.isoformat(), wk_end_date.isoformat(), cap, items)
        st.success(f"Plan saved for ISO week {wk}.")
        st.rerun()

    st.divider()
    st.subheader("📊 Current Week Allocation")
    plan_cur = get_week_plan(USER_ID, wk)
    if not plan_cur:
        cap_auto, items_auto = derive_auto_plan_from_active(USER_ID, wk)
        plan_cur = {"items": items_auto, "capacity": cap_auto}
        st.caption("_Showing derived allocation (not saved yet)._")

    if not plan_cur or not plan_cur.get("items"):
        st.info("No allocations yet.")
    else:
        pe_map = aggregate_pe_by_goal_bucket(USER_ID, wk)
        rows = []
        for it in sorted(plan_cur.get("items", []), key=lambda x: x.get("priority_rank", 99)):
            gid = it["goal_id"]; g = goals_map_full.get(gid, {})
            planned = int(it["planned_current"])
            cur_pe = pe_map.get(gid, {}).get("current", 0.0)
            back_pe = pe_map.get(gid, {}).get("backlog", 0.0)
            rows.append({
                "Priority": it["priority_rank"],
                "Goal": g.get("title", gid),
                "Category": g.get("category", "—"),
                "Planned": planned,
                "Backlog In": int(it["backlog_in"]),
                "Total Target": int(it["total_target"]),
                "Done Current (pe)": round(cur_pe,1),
                "Done Backlog (pe)": round(back_pe,1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("↩️ Rollover Backlog from Previous Week")
        prev_wk = prev_week_key(wk)
        if st.button(f"Compute & Apply Rollover from {prev_wk}", use_container_width=True):
            # keep your same rollover logic here; omitted for brevity
            st.info("Rollover logic same as existing; plug in when ready.")
