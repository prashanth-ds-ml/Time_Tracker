import streamlit as st, time, pandas as pd
from user_management import (
    POMODORO_MIN, BREAK_MIN, sound_alert, get_user_settings, get_user_sessions,
    get_or_create_weekly_plan, save_pomodoro_session, get_daily_target,
    get_adaptive_goal, now_ist, locked_goals_for_user_plan, goal_title_map,
    time_to_minutes
)

# ---- Daily target helpers ----
def render_daily_goal(df: pd.DataFrame):
    if df.empty: return 0, 1, 0
    today = now_ist().date()
    df_work = df[df["pomodoro_type"]=="Work"]
    work_today = df_work[df_work["date"].dt.date==today]
    active_days = len(df_work.groupby(df_work["date"].dt.date).size())
    today_progress = len(work_today)
    today_minutes = int(work_today['duration'].sum())
    adaptive_goal, _, _ = get_adaptive_goal(active_days)
    return today_progress, adaptive_goal, today_minutes

def render_daily_target_planner(user: str, df: pd.DataFrame, today_progress: int):
    st.markdown("## 🎯 Daily Target")
    current = get_daily_target(user)
    if df.empty:
        suggested, phase, _ = 1, "🌱 Building", ""
    else:
        df_work = df[df["pomodoro_type"]=="Work"]
        active_days = len(df_work.groupby(df_work["date"].dt.date).size())
        suggested, phase, _ = get_adaptive_goal(active_days)

    col1, col2 = st.columns([2,3])
    with col1:
        if current is not None:
            st.info(f"Today: **{current}** Pomodoros")
            with st.expander("Change today's target"):
                new_t = st.number_input("New target", 1, 12, value=int(current))
                if st.button("💾 Update Target"):
                    from user_management import save_daily_target
                    save_daily_target(int(new_t), user)
                    st.success("Updated!")
                    st.rerun()
        else:
            st.markdown(f"Suggested: **{suggested}** ({phase})")
            target_input = st.number_input("How many Pomodoros today?", 1, 12, value=int(suggested))
            if st.button("Set Target", use_container_width=True):
                from user_management import save_daily_target
                save_daily_target(int(target_input), user)
                st.success("Saved!")
                st.rerun()
    with col2:
        if current is not None:
            pct = min(100.0, (today_progress / max(1,int(current))) * 100)
            st.progress(pct/100.0, text=f"{pct:.0f}% complete")
        else:
            st.info("Set a target to unlock tracking.")

# ---- glance widgets ----
def this_week_glance_native(user: str, plan: dict, df_work: pd.DataFrame):
    start = pd.to_datetime(plan["week_start"]).date()
    end = pd.to_datetime(plan["week_end"]).date()
    active_ids = plan.get("goals", [])
    alloc = plan.get("allocations", {}) or {}
    if not active_ids or not alloc:
        st.info("No allocations yet for this week. Set them in the Weekly Planner.")
        return

    dfw = df_work.copy()
    dfw["date_only"] = dfw["date"].dt.date
    dfw = dfw[(dfw["date_only"] >= start) & (dfw["date_only"] <= end)]
    by_goal = dfw[dfw["goal_id"].notna()].groupby("goal_id").size().to_dict()
    titles = goal_title_map(user)

    cols = st.columns(2)
    idx = 0
    for gid in active_ids:
        planned = int(alloc.get(gid, 0))
        actual = int(by_goal.get(gid, 0))
        progress = min(1.0, actual / max(1, planned))
        with cols[idx % 2]:
            st.write(f"**{titles.get(gid, '(missing)')}**")
            st.progress(progress, text=f"{actual}/{planned} completed")
        idx += 1

def start_time_sparkline_native(df_work: pd.DataFrame, title="Start-time Stability (median mins from midnight)"):
    if df_work.empty: return
    dfw = df_work.copy()
    dfw["date_only"] = dfw["date"].dt.date
    dfw["start_mins"] = dfw["time"].apply(time_to_minutes)
    dfw = dfw[pd.notna(dfw["start_mins"])]
    if dfw.empty: return
    daily = dfw.groupby("date_only")["start_mins"].median().reset_index().sort_values("date_only")
    daily = daily.rename(columns={"date_only":"date"}).set_index("date")
    st.line_chart(daily, height=220)

# ---- timer widget ----
def render_timer_widget(auto_break: bool, user: str) -> bool:
    if not st.session_state.get("start_time"):
        return False
    duration = BREAK_MIN*60 if st.session_state.is_break else POMODORO_MIN*60
    remaining = int(st.session_state.start_time + duration - time.time())
    if remaining > 0:
        mins, secs = divmod(remaining, 60)
        session_type = "Break" if st.session_state.is_break else f"Working on: {st.session_state.task}"
        st.subheader(f"{'🧘' if st.session_state.is_break else '💼'} {session_type}")
        _, cc, _ = st.columns([1,2,1])
        with cc:
            st.markdown(
                f"<h1 style='text-align:center;font-size:4rem;'>⏱️ {mins:02d}:{secs:02d}</h1>",
                unsafe_allow_html=True
            )
        st.progress(1 - (remaining/duration))
        st.info("🧘 Relax" if st.session_state.is_break else "💪 Stay focused!")
        time.sleep(1)
        st.rerun()
        return True
    else:
        # Finish current session
        was_break = st.session_state.is_break
        save_pomodoro_session(
            user=user,
            is_break=was_break,
            duration=BREAK_MIN if was_break else POMODORO_MIN,
            goal_id=st.session_state.active_goal_id,
            task=st.session_state.task,
            category_label=st.session_state.active_goal_title
        )
        # Play sound BEFORE any rerun/auto-break
        sound_alert()
        st.balloons(); st.success("🎉 Session complete!")

        # Reset work state
        st.session_state.start_time = None
        st.session_state.is_break = False
        st.session_state.task = ""
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = ""

        # Auto-start 5m break after WORK (sound already played)
        if (not was_break) and auto_break:
            st.toast("☕ Auto-starting a 5-minute break")
            st.session_state.start_time = time.time()
            st.session_state.is_break = True
            st.rerun()
        return True

# ---- main page ----
def render_focus_timer(user: str):
    st.header("🎯 Focus Timer")

    settings = get_user_settings(user)
    colset1, _ = st.columns([1, 3])
    with colset1:
        auto_break_ui = st.toggle("Auto-start break", value=settings.get("auto_break", True),
                                  help="Start a 5m break automatically after each 25m work session")
        if auto_break_ui != settings.get("auto_break", True):
            from user_management import users_collection, get_user_settings as _gus
            users_collection.update_one({"username": user}, {"$set": {"auto_break": bool(auto_break_ui)}})
            _gus.clear()

    # Timer running?
    if render_timer_widget(auto_break=get_user_settings(user).get("auto_break", True), user=user):
        return

    plan = get_or_create_weekly_plan(user, now_ist().date())

    df_all = get_user_sessions(user)
    today_progress, adaptive_goal, today_minutes = render_daily_goal(df_all)
    render_daily_target_planner(user, df_all, today_progress)
    st.divider()

    df_work_all = df_all[df_all["pomodoro_type"]=="Work"].copy()
    st.subheader("📌 This Week at a Glance")
    this_week_glance_native(user, plan, df_work_all)
    start_time_sparkline_native(df_work_all)
    st.divider()

    # Mode toggle
    mode = st.radio("Mode", ["Weekly Goal", "Custom (Unplanned)"], horizontal=True)

    if mode == "Weekly Goal":
        active_goal_ids = plan.get("goals", [])
        if not active_goal_ids:
            st.warning("No weekly plan saved yet. Create allocations in **Weekly Planner**.")
        from user_management import fetch_goals
        goals_df = fetch_goals(user, statuses=["New","In Progress"])
        goals_df = goals_df[goals_df["_id"].isin(active_goal_ids)] if not goals_df.empty else goals_df

        locked = set(locked_goals_for_user_plan(user, plan))
        if locked:
            st.warning("⚖️ Balanced Focus: top goals are temporarily locked. Do minimum on others to unlock.")

        # titles only (no 'plan:9' clutter)
        titles_pairs = goals_df[["title","_id"]].values.tolist()
        labels_only = [t for (t, _) in titles_pairs] or ["(no goals)"]

        c1, c2 = st.columns([1,2])
        with c1:
            sel_idx = st.selectbox("Weekly Goal", options=range(len(labels_only)), format_func=lambda i: labels_only[i],
                                   disabled=(len(labels_only)==1 and labels_only[0]=="(no goals)"))
            selected_gid = titles_pairs[sel_idx][1] if titles_pairs else None
            selected_title = titles_pairs[sel_idx][0] if titles_pairs else ""
            if selected_gid in locked:
                st.caption("🔒 This goal is locked for balance. Pick another for now.")
        with c2:
            task = st.text_input("Task (micro-task)", placeholder="e.g., Revise Unit-2 notes")

        st.session_state.active_goal_id = selected_gid
        st.session_state.active_goal_title = selected_title
        st.session_state.task = task

        colw, colb = st.columns(2)
        with colw:
            disabled = (not task.strip()) or (selected_gid in locked if selected_gid else False) or (len(titles_pairs)==0)
            if st.button("▶️ Start Work (25m)", type="primary", use_container_width=True, disabled=disabled):
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
        with colb:
            if st.button("☕ Break (5m)", use_container_width=True):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.session_state.active_goal_id = None
                st.session_state.active_goal_title = ""
                st.session_state.task = ""
                st.rerun()
    else:
        # Custom (Unplanned)
        current_cats = get_user_settings(user).get("custom_categories", ["Learning","Projects","Research","Planning"])
        cat_options = current_cats + ["+ Add New"]
        selected = st.selectbox("📂 Custom Category", cat_options)
        if selected == "+ Add New":
            new_cat = st.text_input("New category", placeholder="e.g., Marketing")
            if new_cat and st.button("✅ Add Category"):
                from user_management import users_collection, get_user_settings as _gus
                if new_cat not in current_cats:
                    users_collection.update_one({"username": user}, {"$addToSet": {"custom_categories": new_cat}})
                    _gus.clear()
                    st.success("Added!")
                    st.rerun()
            category_label = new_cat if new_cat else ""
        else:
            category_label = selected
        task = st.text_input("Task (micro-task)", placeholder="e.g., Draft outreach emails")

        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = category_label
        st.session_state.task = task

        colw, colb = st.columns(2)
        with colw:
            disabled = not (category_label and task.strip())
            if st.button("▶️ Start Work (25m)", type="primary", use_container_width=True, disabled=disabled):
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
        with colb:
            if st.button("☕ Break (5m)", use_container_width=True):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.session_state.active_goal_id = None
                st.session_state.active_goal_title = ""
                st.session_state.task = ""
                st.rerun()

    # Today summary (safe for empty)
    df = get_user_sessions(user)
    if not df.empty:
        today = now_ist().date()
        df["date_only"] = df["date"].dt.date
        today_data = df[df["date_only"] == today]
        work_today = today_data[today_data["pomodoro_type"]=="Work"]
        breaks_today = len(today_data[today_data["pomodoro_type"]=="Break"])
        st.divider(); st.subheader("📊 Today")
        col1,col2,col3,col4 = st.columns(4)
        with col1: st.metric("Work Sessions", len(work_today))
        with col2: st.metric("Focus Minutes", int(work_today['duration'].sum()))
        with col3:
            ratio = (breaks_today / max(1, len(work_today)))
            label = "⚖️ Balanced" if 0.3<=ratio<=0.7 else ("🎯 More focus" if ratio>0.7 else "🧘 Take breaks")
            st.metric("Breaks", breaks_today, help=label)
        with col4:
            current_target = get_daily_target(user)
            if current_target:
                pct = (len(work_today)/max(1,int(current_target)))*100
                st.metric("Target Progress", f"{pct:.0f}%")
            else:
                st.metric("Target Progress", "—")
