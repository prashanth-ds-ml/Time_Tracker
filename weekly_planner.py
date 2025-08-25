import streamlit as st
import pandas as pd
from datetime import timedelta
from user_management import (
    get_user_settings, week_bounds_ist, week_day_counts, compute_weekly_capacity,
    fetch_goals, upsert_goal, proportional_allocation,
    get_user_sessions, get_or_create_weekly_plan, save_plan_allocations,
    now_ist
)

def _goal_dataframe_for_editor(goals_df: pd.DataFrame) -> pd.DataFrame:
    # Editable subset
    view = goals_df[["_id","title","goal_type","priority_weight","status"]].copy()
    view.rename(columns={
        "_id":"goal_id","goal_type":"Type","priority_weight":"Weight","status":"Status","title":"Title"
    }, inplace=True)
    return view

def render_weekly_planner(user: str, selected_date):
    st.header("📅 Weekly Planner")

    # Week picker
    pick_date = st.date_input("Week of", value=selected_date, key="planner_week_picker")
    week_start, week_end = week_bounds_ist(pick_date)
    if pick_date != selected_date:
        st.session_state.planning_week_date = pick_date
        st.rerun()

    # Capacity form
    settings = get_user_settings(user)
    plan = get_or_create_weekly_plan(user, week_start)
    wd_count, we_count = week_day_counts(week_start)
    with st.form("capacity_form"):
        c1, c2, c3 = st.columns(3)
        with c1: wp = st.number_input("Weekday avg", 0, 12, value=settings["weekday_poms"])
        with c2: we = st.number_input("Weekend avg", 0, 12, value=settings["weekend_poms"])
        with c3:
            total = compute_weekly_capacity({"weekday_poms": wp, "weekend_poms": we}, weekdays=wd_count, weekend_days=we_count)
            st.metric(f"Capacity {week_start} → {week_end}", f"{total}")
        save_defaults = st.form_submit_button("💾 Save Defaults")
    if 'save_defaults' in locals() and save_defaults:
        from user_management import users_collection, get_user_settings as _gus
        users_collection.update_one({"username": user}, {"$set": {"weekday_poms": int(wp), "weekend_poms": int(we)}})
        _gus.clear()
        st.success("Defaults saved.")
        st.rerun()

    st.divider()

    # Goals table (editable)
    st.subheader("🎯 Goals & Priority Weights")
    goals_df = fetch_goals(user, statuses=["New","In Progress","On Hold","Completed","Archived"])
    if goals_df.empty:
        with st.form("add_first_goal"):
            t = st.text_input("Title", placeholder="e.g., UGC NET Paper 1")
            typ = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0)
            w = st.select_slider("Priority", options=[1,2,3], value=2, help="High=3, Medium=2, Low=1")
            submit_new = st.form_submit_button("💾 Save Goal")
        if submit_new and t.strip():
            upsert_goal(user, t.strip(), int(w), typ, "New")
            fetch_goals.clear()
            st.success("Goal added.")
            st.rerun()
        st.info("Add goals to plan the week.")
        return

    editable = _goal_dataframe_for_editor(goals_df)
    with st.form("goals_editor_form"):
        edited = st.data_editor(
            editable,
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Title": st.column_config.TextColumn("Title", required=True, width="medium"),
                "Type": st.column_config.SelectboxColumn("Type",
                    options=["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"]),
                "Weight": st.column_config.SelectboxColumn("Weight", options=[1,2,3]),
                "Status": st.column_config.SelectboxColumn("Status",
                    options=["New","In Progress","On Hold","Completed","Archived"]),
            }
        )
        update_goals = st.form_submit_button("💾 Update Goals")

    if update_goals:
        # Validate titles unique (per user)
        titles = edited["Title"].fillna("").str.strip()
        if titles.eq("").any():
            st.error("Please fill titles for all rows.")
        elif titles.duplicated().any():
            st.error("Duplicate titles detected. Titles must be unique per user.")
        else:
            # Upsert all rows shown (existing + new)
            for _, row in edited.iterrows():
                upsert_goal(
                    user,
                    row["Title"].strip(),
                    int(row["Weight"]),
                    row["Type"],
                    row["Status"],
                    target_poms=0
                )
            fetch_goals.clear()
            st.success("Goals updated.")
            st.rerun()

    st.divider()

    # Allocation block (for active goals only)
    st.subheader("🧮 Allocate Weekly Pomodoros")
    active_goals = fetch_goals(user, statuses=["New","In Progress"])
    if active_goals.empty:
        st.info("No active goals to allocate. Mark some goals as New/In Progress.")
        return

    # Auto-allocate from weights
    wd_count, we_count = week_day_counts(week_start)
    total_poms = compute_weekly_capacity(get_user_settings(user), weekdays=wd_count, weekend_days=we_count)
    weight_map = {row["_id"]: int(row.get("priority_weight", 2)) for _, row in active_goals.iterrows()}
    auto = proportional_allocation(total_poms, weight_map)

    # Actuals this week
    df_all = get_user_sessions(user)
    df_all["date_only"] = pd.to_datetime(df_all["date"]).dt.date if not df_all.empty else pd.Series([], dtype="object")
    mask = (df_all["date_only"] >= week_start) & (df_all["date_only"] <= week_end) if not df_all.empty else pd.Series([], dtype=bool)
    dfw = df_all[mask & (df_all["pomodoro_type"]=="Work")] if not df_all.empty else pd.DataFrame(columns=df_all.columns)
    actual_by_goal = dfw[dfw["goal_id"].notna()].groupby("goal_id").size().to_dict()

    # Build allocation editor table
    titles = {row["_id"]: row["title"] for _, row in active_goals.iterrows()}
    plan_alloc = plan.get("allocations", {}) or {}
    rows = []
    for gid in active_goals["_id"]:
        planned = int(plan_alloc.get(gid, auto.get(gid, 0)))
        actual = int(actual_by_goal.get(gid, 0))
        rows.append({"Goal ID": gid, "Title": titles.get(gid, "(missing)"),
                     "Planned": planned, "Actual": actual, "Δ": planned - actual})
    alloc_df = pd.DataFrame(rows)

    with st.form("alloc_editor_form"):
        edited_alloc = st.data_editor(
            alloc_df[["Title","Planned","Actual","Δ"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Title": st.column_config.TextColumn("Title", disabled=True, width="large"),
                "Planned": st.column_config.NumberColumn("Planned", min_value=0, step=1),
                "Actual": st.column_config.NumberColumn("Actual", disabled=True),
                "Δ": st.column_config.NumberColumn("Δ", disabled=True),
            }
        )
        # capacity bar
        sum_planned = int(edited_alloc["Planned"].sum()) if not edited_alloc.empty else 0
        cap_text = f"Sum = {sum_planned} / Capacity {total_poms}"
        if sum_planned == total_poms:
            st.success(cap_text)
        elif sum_planned < total_poms:
            st.warning(cap_text + " (under)")
        else:
            st.error(cap_text + " (over)")

        colN, colS = st.columns([1,1])
        with colN:
            normalize = st.form_submit_button("🔁 Normalize to Capacity")
        with colS:
            save_plan = st.form_submit_button("📌 Save Weekly Plan", type="primary")

    if 'normalize' in locals() and normalize:
        normalized = proportional_allocation(total_poms, {titles[g]: int(p) if int(p)>0 else 1
                                                          for g, p in zip(active_goals["_id"], edited_alloc["Planned"].tolist())})
        # update session widget values by title matching
        for i, (gid, title) in enumerate(zip(active_goals["_id"], edited_alloc["Title"].tolist())):
            new_val = normalized.get(title, edited_alloc.loc[i, "Planned"])
            edited_alloc.loc[i, "Planned"] = int(new_val)
        st.experimental_rerun()

    if 'save_plan' in locals() and save_plan:
        # Map back Title->gid safely
        title_to_gid = {v: k for k, v in titles.items()}
        new_alloc = {}
        new_goals = []
        for _, row in edited_alloc.iterrows():
            gid = title_to_gid.get(row["Title"])
            if gid:
                new_goals.append(gid)
                new_alloc[gid] = int(row["Planned"])
        save_plan_allocations(plan["_id"], new_goals, new_alloc)
        st.success("Weekly plan saved!")
        st.rerun()

    st.divider()

    # Rollover wizard (optional)
    with st.expander("↪️ Rollover unfinished from last week", expanded=False):
        prev_start = week_start - timedelta(days=7)
        prev_end = prev_start + timedelta(days=6)
        prev_plan = get_or_create_weekly_plan(user, prev_start)
        prev_alloc = prev_plan.get("allocations", {}) or {}
        if not prev_alloc:
            st.info("No previous plan found.")
        else:
            df_prev = get_user_sessions(user)
            if df_prev.empty:
                st.info("No data from last week.")
            else:
                df_prev["date_only"] = pd.to_datetime(df_prev["date"]).dt.date
                mprev = (df_prev["date_only"] >= prev_start) & (df_prev["date_only"] <= prev_end)
                dfw_prev = df_prev[mprev & (df_prev["pomodoro_type"]=="Work") & (df_prev["goal_id"].notna())]
                actual_prev = dfw_prev.groupby("goal_id").size().to_dict()
                rows = []
                titles_prev = {gid: collection_goals.find_one({"_id": gid}, {"title":1}).get("title","(missing)")
                               for gid in prev_alloc.keys()}
                for gid, planned in prev_alloc.items():
                    actual = int(actual_prev.get(gid, 0))
                    carry = max(0, int(planned) - actual)
                    rows.append({"Include": carry>0, "Title": titles_prev.get(gid,"(missing)"),
                                 "Planned": int(planned), "Actual": actual, "Carry": carry, "Goal ID": gid})
                df_roll = pd.DataFrame(rows)
                df_edit = st.data_editor(
                    df_roll[["Include","Title","Planned","Actual","Carry"]],
                    num_rows="dynamic", hide_index=True, use_container_width=True,
                    column_config={
                        "Include": st.column_config.CheckboxColumn("Include"),
                        "Title": st.column_config.TextColumn("Title", disabled=True),
                        "Planned": st.column_config.NumberColumn("Planned", disabled=True),
                        "Actual": st.column_config.NumberColumn("Actual", disabled=True),
                        "Carry": st.column_config.NumberColumn("Carry", disabled=True),
                    }
                )
                if st.button("Apply Rollover"):
                    carry_rows = df_roll[df_edit["Include"] == True]
                    curr_alloc = dict(plan.get("allocations", {}))
                    curr_goals = set(plan.get("goals", []))
                    for _, r in carry_rows.iterrows():
                        gid = r["Goal ID"]
                        if r["Carry"] > 0:
                            curr_goals.add(gid)
                            curr_alloc[gid] = int(curr_alloc.get(gid, 0)) + int(r["Carry"])
                    save_plan_allocations(plan["_id"], list(curr_goals), curr_alloc)
                    st.success("Rolled over unfinished poms into this week.")
                    st.experimental_rerun()

    st.divider()

    # Last week snapshot table and status buckets
    st.subheader("📋 Last Week Snapshot")
    prev_start = week_start - timedelta(days=7); prev_end = prev_start + timedelta(days=6)
    prev_plan = get_or_create_weekly_plan(user, prev_start)
    prev_alloc = prev_plan.get("allocations", {}) or {}
    if prev_alloc:
        df_prev = get_user_sessions(user)
        df_prev["date_only"] = pd.to_datetime(df_prev["date"]).dt.date if not df_prev.empty else pd.Series([], dtype="object")
        mprev = (df_prev["date_only"] >= prev_start) & (df_prev["date_only"] <= prev_end) if not df_prev.empty else pd.Series([], dtype=bool)
        dfw_prev = df_prev[mprev & (df_prev["pomodoro_type"]=="Work") & (df_prev["goal_id"].notna())] if not df_prev.empty else pd.DataFrame(columns=df_prev.columns)
        actual_prev = dfw_prev.groupby("goal_id").size().to_dict()
        rows = []
        for gid, planned in prev_alloc.items():
            title = collection_goals.find_one({"_id": gid}, {"title":1}).get("title","(missing)")
            actual = int(actual_prev.get(gid, 0))
            rows.append({"Title": title, "Planned": int(planned), "Actual": actual, "Δ": int(planned)-actual})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No plan existed for last week.")

    st.subheader("📚 Goals by Status")
    g_all = fetch_goals(user)
    for label, statuses in [
        ("🟢 Ongoing", ["New","In Progress"]),
        ("🟡 On Hold", ["On Hold"]),
        ("✅ Completed", ["Completed"]),
        ("📦 Archived", ["Archived"]),
    ]:
        sub = g_all[g_all["status"].isin(statuses)][["title","goal_type","priority_weight","status"]]
        if not sub.empty:
            st.markdown(f"**{label}**")
            st.dataframe(sub.rename(columns={"title":"Title","goal_type":"Type","priority_weight":"Weight","status":"Status"}),
                         use_container_width=True, hide_index=True)
