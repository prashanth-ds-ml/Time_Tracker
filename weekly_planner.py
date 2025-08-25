# weekly_planner.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date

from user_management import (
    now_ist, week_bounds_ist, week_day_counts,
    get_user_settings, get_user_sessions, fetch_goals, upsert_goal, goal_title_map,
    collection_plans, collection_goals
)

# ---- Capacity & allocation helpers ----
def compute_weekly_capacity(settings, weekdays: int = 5, weekend_days: int = 2) -> int:
    return settings["weekday_poms"] * weekdays + settings["weekend_poms"] * weekend_days

def proportional_allocation(total: int, weights: dict) -> dict:
    total_w = max(1, sum(max(1, int(w)) for w in weights.values()))
    raw = {gid: (max(1, int(w)) / total_w) * total for gid, w in weights.items()}
    allocated = {gid: int(v) for gid, v in raw.items()}
    diff = total - sum(allocated.values())
    if diff != 0:
        fracs = sorted(((gid, raw[gid] - int(raw[gid])) for gid in raw), key=lambda x: x[1], reverse=True)
        idx = 0
        while diff != 0 and fracs:
            gid = fracs[idx % len(fracs)][0]
            allocated[gid] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            idx += 1
    return allocated

# ---- Plans CRUD ----
def get_or_create_weekly_plan(username: str, d: date) -> dict:
    week_start, week_end = week_bounds_ist(d)
    pid = f"{username}|{week_start.isoformat()}"
    plan = collection_plans.find_one({"_id": pid})
    if plan:
        return plan
    settings = get_user_settings(username)
    wd, we = week_day_counts(week_start)
    total_poms = compute_weekly_capacity(settings, weekdays=wd, weekend_days=we)
    doc = {
        "_id": pid,
        "user": username,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "total_poms": total_poms,
        "goals": [],
        "allocations": {},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    collection_plans.insert_one(doc)
    return doc

def save_plan_allocations(plan_id: str, goals: list, allocations: dict):
    goals_unique = sorted(set(goals))
    clean_alloc = {gid: int(allocations.get(gid, 0)) for gid in goals_unique}
    collection_plans.update_one(
        {"_id": plan_id},
        {"$set": {"goals": goals_unique, "allocations": clean_alloc, "updated_at": datetime.utcnow()}}
    )

def sanitize_weight(v, default=2):
    try:
        vv = int(v)
    except Exception:
        vv = default
    return vv if vv in (1,2,3) else default

# ---- Page renderer ----
def render_weekly_planner(user: str, planning_week_date: date):
    st.header("📅 Weekly Planner")

    pick_date = st.date_input("Week of", value=planning_week_date, key="planner_week_picker")
    week_start, week_end = week_bounds_ist(pick_date)

    # --- Capacity controls ---
    settings = get_user_settings(user)
    colA, colB, colC = st.columns(3)
    with colA:
        wp = st.number_input("Weekday avg", 0, 12, value=int(settings["weekday_poms"]), key="cap_wp")
    with colB:
        we = st.number_input("Weekend avg", 0, 12, value=int(settings["weekend_poms"]), key="cap_we")
    with colC:
        wd_count, we_count = week_day_counts(week_start)
        total_live = compute_weekly_capacity({"weekday_poms": wp, "weekend_poms": we}, weekdays=wd_count, weekend_days=we_count)
        st.metric(f"Capacity {week_start} → {week_end}", f"{total_live}")
        if (wp != settings["weekday_poms"]) or (we != settings["weekend_poms"]):
            if st.button("💾 Save as Defaults", use_container_width=True, key="btn_save_defaults"):
                from user_management import users_collection, get_user_settings as _gus
                users_collection.update_one({"username": user}, {"$set": {"weekday_poms": int(wp), "weekend_poms": int(we)}})
                _gus.clear()
                st.success("Saved new defaults")
                st.rerun()

    st.divider()

    # --- Goal Catalog ---
    st.subheader("📌 Goal Catalog")
    goals_df_all = fetch_goals(user)
    if goals_df_all.empty:
        with st.expander("Add Goal"):
            g_title = st.text_input("Title", placeholder="e.g., UGC NET Paper 1", key="new_goal_title")
            g_type = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0, key="new_goal_type")
            g_weight = st.select_slider("Priority", options=[1,2,3], value=2, help="High=3, Medium=2, Low=1", key="new_goal_weight")
            if st.button("💾 Save Goal", key="btn_save_goal"):
                if g_title.strip():
                    upsert_goal(user, g_title.strip(), int(g_weight), g_type, "New")
                    fetch_goals.clear()
                    st.success("Saved goal")
                    st.rerun()
        st.info("Add 3–4 goals to plan the week.")
        return

    col_ongo, col_hold, col_done = st.columns(3)
    with col_ongo:
        st.markdown("**Ongoing** (New/In Progress)")
        g = goals_df_all[goals_df_all["status"].isin(["New","In Progress"])][["title","priority_weight","goal_type"]]
        st.dataframe(g.rename(columns={"priority_weight":"Priority","goal_type":"Type"}), use_container_width=True, hide_index=True, height=min(220, 38+len(g)*32))
    with col_hold:
        st.markdown("**On Hold**")
        g = goals_df_all[goals_df_all["status"].isin(["On Hold"])][["title","priority_weight","goal_type"]]
        st.dataframe(g.rename(columns={"priority_weight":"Priority","goal_type":"Type"}), use_container_width=True, hide_index=True, height=min(220, 38+len(g)*32))
    with col_done:
        st.markdown("**Completed (recent)**")
        g = goals_df_all[goals_df_all["status"].isin(["Completed"])][["title","goal_type"]].head(10)
        st.dataframe(g.rename(columns={"goal_type":"Type"}), use_container_width=True, hide_index=True, height=min(220, 38+len(g)*32))

    st.divider()

    # --- Priority Weights (Ongoing) ---
    st.subheader("🎯 Priority Weights (Ongoing)")
    goals_df = goals_df_all[goals_df_all["status"].isin(["New","In Progress"])].copy()
    weights = {}
    cols = st.columns(min(4, max(1, len(goals_df))))
    for i, (_, row) in enumerate(goals_df.iterrows()):
        with cols[i % len(cols)]:
            st.write(f"**{row['title']}**")
            default_w = sanitize_weight(row.get("priority_weight", 2))
            w = st.select_slider("Priority", options=[1,2,3], value=default_w, key=f"w_{row['_id']}", help="High=3, Medium=2, Low=1")
            weights[row["_id"]] = int(w)

    if st.button("💾 Update Priorities", key="btn_update_weights"):
        for gid, w in weights.items():
            collection_goals.update_one({"_id": gid}, {"$set": {"priority_weight": int(w), "updated_at": datetime.utcnow()}})
        fetch_goals.clear()
        st.success("Priorities updated.")
        st.rerun()

    st.divider()

    # --- Last Week Summary & Close-out ---
    st.subheader("🧭 Last Week: Summary & Close-out")
    prev_start = week_start - timedelta(days=7)
    prev_end = prev_start + timedelta(days=6)
    prev_plan = collection_plans.find_one({"_id": f"{user}|{prev_start.isoformat()}"}) or {}
    prev_alloc = prev_plan.get("allocations", {}) or {}
    titles = goal_title_map(user)

    df_all_user = get_user_sessions(user)

    if df_all_user.empty or not prev_alloc:
        st.info("No previous plan or no data yet.")
        prev_summary_df = pd.DataFrame(columns=["goal_id","Title","Planned","Actual","Delta"])
    else:
        mask_prev = (df_all_user["date"].dt.date >= prev_start) & (df_all_user["date"].dt.date <= prev_end)
        dfw_prev = df_all_user[mask_prev & (df_all_user["pomodoro_type"]=="Work")].copy()
        actual_prev = dfw_prev[dfw_prev["goal_id"].notna()].groupby("goal_id").size().to_dict()

        rows = [{
            "goal_id": gid,
            "Title": titles.get(gid, "(missing)"),
            "Planned": int(planned),
            "Actual": int(actual_prev.get(gid, 0))
        } for gid, planned in prev_alloc.items()]
        prev_summary_df = pd.DataFrame(rows)
        if not prev_summary_df.empty:
            prev_summary_df["Delta"] = prev_summary_df["Actual"] - prev_summary_df["Planned"]

    if not prev_summary_df.empty:
        st.dataframe(
            prev_summary_df[["Title","Planned","Actual","Delta"]].sort_values("Planned", ascending=False),
            use_container_width=True, hide_index=True
        )
        st.caption(f"Week: {prev_start} → {prev_end}")

        st.markdown("**Close-out Actions**")
        for _, r in prev_summary_df.iterrows():
            gid = r["goal_id"]
            carry_default = max(0, int(r["Planned"]) - int(r["Actual"]))
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            with col1:
                st.write(f"**{r['Title']}**")
            with col2:
                status = st.selectbox("Status", ["Completed","Rollover","On Hold","Archived","In Progress"],
                                      index=4, key=f"close_prev_{gid}")
            with col3:
                carry = st.number_input("Carry fwd poms", 0, 200, value=int(carry_default), key=f"carry_prev_{gid}")
            with col4:
                if st.button("✅ Apply", key=f"apply_prev_{gid}"):
                    new_status = ("Completed" if status=="Completed" else
                                  "On Hold" if status=="On Hold" else
                                  "Archived" if status=="Archived" else
                                  "In Progress")
                    collection_goals.update_one({"_id": gid}, {"$set": {"status": new_status}})
                    if status == "Rollover" and carry > 0:
                        curr_plan = get_or_create_weekly_plan(user, week_start)
                        curr_alloc = dict(curr_plan.get("allocations", {}))
                        curr_goals = set(curr_plan.get("goals", []))
                        curr_goals.add(gid)
                        curr_alloc[gid] = int(curr_alloc.get(gid, 0)) + int(carry)
                        save_plan_allocations(curr_plan["_id"], list(curr_goals), curr_alloc)
                    st.success("Updated")
                    st.rerun()

    else:
        st.info("No previous plan or nothing to close out.")

    st.divider()

    # --- Allocate CURRENT week ---
    st.subheader("🧮 Allocate Weekly Pomodoros")
    wd_count, we_count = week_day_counts(week_start)
    total_poms_live = compute_weekly_capacity(get_user_settings(user), weekdays=wd_count, weekend_days=we_count)
    weight_map = {row["_id"]: int(weights.get(row["_id"], sanitize_weight(row.get("priority_weight", 2))))
                  for _, row in goals_df.iterrows()}
    auto = proportional_allocation(int(total_poms_live), weight_map)

    plan = get_or_create_weekly_plan(user, week_start)
    plan_has_alloc = bool(plan.get("allocations"))
    if plan_has_alloc:
        st.caption("A plan already exists for this week. Adjust and save to update.")

    edited = {}
    cols2 = st.columns(min(4, max(1, len(goals_df))))
    cap = int(total_poms_live)
    for i, (_, row) in enumerate(goals_df.iterrows()):
        with cols2[i % len(cols2)]:
            plan_val = int(plan.get("allocations", {}).get(row['_id'], 0))
            auto_val = int(auto.get(row['_id'], 0))
            default_val = plan_val if plan_has_alloc else auto_val
            default_val = max(0, min(default_val, cap))
            val = st.number_input(f"{row['title']}", min_value=0, max_value=cap, value=default_val, step=1, key=f"alloc_{row['_id']}")
            edited[row["_id"]] = int(val)

    sum_edit = sum(edited.values())
    if sum_edit != cap:
        st.warning(f"Allocations sum to {sum_edit}, not {cap}.")
        if st.button("🔁 Normalize to capacity", key="btn_normalize"):
            edited = proportional_allocation(cap, {gid: max(1, v) for gid, v in edited.items()})
            for gid, v in edited.items():
                st.session_state[f"alloc_{gid}"] = v
            st.rerun()

    btn_label = "📌 Save Weekly Plan" if not plan_has_alloc else "📌 Update Weekly Plan"
    if st.button(btn_label, type="primary", key="btn_save_plan"):
        save_plan_allocations(plan["_id"], list(edited.keys()), edited)
        st.success("Weekly plan saved!")
        st.rerun()
