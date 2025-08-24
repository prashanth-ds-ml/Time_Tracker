# focus.py
import streamlit as st
import time
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

from user_management import (
    get_user_settings, get_user_sessions, fetch_goals, goal_title_map,
    now_ist, collection_logs, collection_goals, time_to_minutes, safe_div
)
from weekly_planner import get_or_create_weekly_plan

# --- Timer constants & sound ---
POMODORO_MIN = 25
BREAK_MIN = 5
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"

def sound_alert():
    st.components.v1.html(f"""
        <audio id="beep" autoplay><source src="{SOUND_PATH}" type="audio/mpeg"></audio>
        <script>
            const audio = document.getElementById('beep');
            if (audio) {{ audio.volume = 0.6; audio.play().catch(()=>{{}}); }}
        </script>
    """, height=0)

# --- Balanced Focus Locking ---
def is_within_lock_window(plan: dict, days_window: int = 3) -> bool:
    start = datetime.fromisoformat(plan["week_start"]).date()
    today = now_ist().date()
    return (today - start).days <= (days_window - 1)

@st.cache_data(ttl=30)
def locked_goals_for_user_plan(username: str, plan: dict, threshold_pct: float = 0.7, min_other: int = 3):
    if not is_within_lock_window(plan):
        return []
    start = datetime.fromisoformat(plan["week_start"]).date()
    today = now_ist().date()
    df = get_user_sessions(username)
    if df.empty: return []
    mask_week = (df["date"].dt.date >= start) & (df["date"].dt.date <= today)
    dfw = df[mask_week & (df["pomodoro_type"] == "Work")].copy()
    if dfw.empty: return []
    if 'goal_id' not in dfw.columns:
        dfw['goal_id'] = None
    by_goal = dfw.groupby(dfw["goal_id"].astype('object').fillna("NONE")).size().sort_values(ascending=False)
    total = int(by_goal.sum())
    if total < 4: return []
    top2 = by_goal.head(2).sum()
    if (top2 / max(1,total)) >= threshold_pct:
        dominating = [g for g in list(by_goal.head(2).index) if g != "NONE"]
        others = by_goal[~by_goal.index.isin(dominating)]
        if len(others) == 0 or any(others < min_other):
            return dominating
    return []

# --- Save session ---
def save_pomodoro_session(user: str, is_break: bool, duration: int, goal_id, task: str, category_label: str):
    now = now_ist()
    doc = {
        "type": "Pomodoro",
        "date": now.date().isoformat(),
        "time": now.strftime("%I:%M %p"),
        "pomodoro_type": "Break" if is_break else "Work",
        "duration": duration,
        "user": user,
        "goal_id": goal_id if not is_break else None,
        "task": task if not is_break else "",
        "category": category_label if (category_label and not is_break) else "",
        "created_at": datetime.utcnow()
    }
    collection_logs.insert_one(doc)
    if (not is_break) and goal_id:
        collection_goals.update_one({"_id": goal_id}, {"$inc": {"poms_completed": 1}, "$set": {"updated_at": datetime.utcnow()}})
    get_user_sessions.clear()

# --- Daily Targets ---
def get_adaptive_goal(active_days:int):
    if active_days <= 5: return 1, "🌱 Building", "Start small - consistency over intensity"
    elif active_days <= 12: return 2, "🔥 Growing", "Momentum building"
    elif active_days <= 19: return 3, "💪 Strong", "Push your limits"
    else: return 4, "🚀 Peak", "Maintain the peak"

def save_daily_target(target:int, user:str):
    today = now_ist().date().isoformat()
    target_doc = {"type":"DailyTarget","date": today, "target": int(target), "user": user, "created_at": datetime.utcnow()}
    collection_logs.update_one({"type":"DailyTarget","date": today,"user": user},{"$set": target_doc}, upsert=True)

def get_daily_target(user:str):
    today = now_ist().date().isoformat()
    doc = collection_logs.find_one({"type":"DailyTarget","date": today,"user": user})
    return int(doc["target"]) if doc else None

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
    current_target = get_daily_target(user)
    if df.empty:
        suggested_target, phase_name, _ = 1, "🌱 Building", ""
    else:
        df_work = df[df["pomodoro_type"]=="Work"]
        active_days = len(df_work.groupby(df_work["date"].dt.date).size())
        suggested_target, phase_name, _ = get_adaptive_goal(active_days)
    col1, col2 = st.columns([2,3])
    with col1:
        if current_target is not None:
            st.info(f"Today: **{current_target} Pomodoros**")
            with st.expander("Change Today's Target"):
                new_target = st.number_input("New target", 1, 12, value=int(current_target), key="daily_target_input")
                if st.button("💾 Update Target", key="btn_update_daily_target"):
                    save_daily_target(int(new_target), user); st.success("Updated!"); st.rerun()
        else:
            st.markdown(f"Suggested: **{suggested_target}** ({phase_name})")
            target_input = st.number_input("How many Pomodoros today?", 1, 12, value=int(suggested_target), key="daily_target_set_input")
            if st.button("Set Target", use_container_width=True, key="btn_set_daily_target"):
                save_daily_target(int(target_input), user); st.success("Saved!"); st.rerun()
    with col2:
        if current_target is not None:
            pct = min(100.0, (today_progress / max(1,int(current_target))) * 100)
            st.progress(pct/100.0, text=f"{pct:.0f}% complete")
        else:
            st.info("Set a target to unlock tracking.")

# --- Small UI helpers ---
def this_week_glance_native(user: str, plan: dict, df_work: pd.DataFrame):
    start = datetime.fromisoformat(plan["week_start"]).date()
    end = datetime.fromisoformat(plan["week_end"]).date()
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
        progress = min(1.0, (actual / max(1, planned)))
        with cols[idx % 2]:
            st.write(f"**{titles.get(gid, '(missing)')}**")
            st.progress(progress, text=f"{actual}/{planned} completed")
        idx += 1

def start_time_sparkline_native(df_work: pd.DataFrame):
    if df_work.empty: return
    dfw = df_work.copy()
    dfw["date_only"] = dfw["date"].dt.date
    dfw["start_mins"] = dfw["time"].apply(time_to_minutes)
    dfw = dfw[pd.notna(dfw["start_mins"])]
    if dfw.empty: return
    daily = dfw.groupby("date_only")["start_mins"].median().reset_index().sort_values("date_only")
    daily = daily.rename(columns={"date_only":"date"}).set_index("date")
    st.line_chart(daily, height=220)

# --- Timer widget ---
def render_timer_widget(user: str, auto_break: bool) -> bool:
    if not st.session_state.start_time:
        return False
    duration = BREAK_MIN*60 if st.session_state.is_break else POMODORO_MIN*60
    remaining = int(st.session_state.start_time + duration - time.time())
    if remaining > 0:
        mins, secs = divmod(remaining, 60)
        session_type = "Break" if st.session_state.is_break else f"Working on: {st.session_state.task}"
        st.subheader(f"{'🧘' if st.session_state.is_break else '💼'} {session_type}")
        _, cc, _ = st.columns([1,2,1])
        with cc:
            st.markdown(f"<h1 style='text-align:center;font-size:4rem;'>⏱️ {mins:02d}:{secs:02d}</h1>", unsafe_allow_html=True)
        progress = 1 - (remaining/duration)
        st.progress(progress)
        st.info("🧘 Relax" if st.session_state.is_break else "💪 Stay focused!")
        time.sleep(1); st.rerun(); return True
    else:
        was_break = st.session_state.is_break
        save_pomodoro_session(
            user=user,
            is_break=was_break,
            duration=BREAK_MIN if was_break else POMODORO_MIN,
            goal_id=st.session_state.active_goal_id,
            task=st.session_state.task,
            category_label=st.session_state.active_goal_title
        )
        sound_alert()  # play BEFORE any auto-break
        st.balloons(); st.success("🎉 Session complete!")
        # reset state
        st.session_state.start_time = None
        st.session_state.is_break = False
        st.session_state.task = ""
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = ""
        # auto break
        if (not was_break) and auto_break:
            st.toast("☕ Auto-starting a 5-minute break")
            st.session_state.start_time = time.time()
            st.session_state.is_break = True
            st.rerun()
        return True

# --- Focus page ---
def render_focus_timer(user: str):
    st.header("🎯 Focus Timer")

    settings = get_user_settings(user)
    colset1, _ = st.columns([1, 3])
    with colset1:
        auto_break_ui = st.toggle("Auto-start break", value=settings.get("auto_break", True), help="Start a 5m break automatically after each 25m work session")
        if auto_break_ui != settings.get("auto_break", True):
            from user_management import users_collection, get_user_settings as _gus
            users_collection.update_one({"username": user}, {"$set": {"auto_break": bool(auto_break_ui)}})
            _gus.clear()

    if render_timer_widget(user=user, auto_break=get_user_settings(user).get("auto_break", True)):
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
    mode = st.radio("Mode", ["Weekly Goal", "Custom (Unplanned)"], horizontal=True, key="focus_mode")

    if mode == "Weekly Goal":
        active_goal_ids = plan.get("goals", [])
        if not active_goal_ids:
            st.warning("No weekly plan saved yet. Create allocations in **Weekly Planner**.")
        goals_df = fetch_goals(user, statuses=["New","In Progress"])
        goals_df = goals_df[goals_df["_id"].isin(active_goal_ids)] if not goals_df.empty else goals_df

        locked = set(locked_goals_for_user_plan(user, plan))
        if locked:
            st.warning("⚖️ Balanced Focus: top goals are temporarily locked. Do minimum on others to unlock.")

        # titles only
        titles_pairs = goals_df[["title","_id"]].values.tolist()
        labels_only = [t for (t, _) in titles_pairs] or ["(no goals)"]

        c1, c2 = st.columns([1,2])
        with c1:
            sel_idx = st.selectbox("Weekly Goal", options=range(len(labels_only)), format_func=lambda i: labels_only[i],
                                   disabled=len(labels_only)==1 and labels_only[0]=="(no goals)", key="focus_goal_select")
            selected_gid = titles_pairs[sel_idx][1] if titles_pairs else None
            selected_title = titles_pairs[sel_idx][0] if titles_pairs else ""
            if selected_gid in locked:
                st.caption("🔒 This goal is locked for balance. Pick another for now.")
        with c2:
            task = st.text_input("Task (micro-task)", placeholder="e.g., Revise Unit-2 notes", key="focus_task_input")

        st.session_state.active_goal_id = selected_gid
        st.session_state.active_goal_title = selected_title
        st.session_state.task = task

        colw, colb = st.columns(2)
        with colw:
            disabled = (not task.strip()) or (selected_gid in locked if selected_gid else False) or (len(titles_pairs)==0)
            if st.button("▶️ Start Work (25m)", type="primary", use_container_width=True, disabled=disabled, key="btn_start_work"):
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
        with colb:
            if st.button("☕ Break (5m)", use_container_width=True, key="btn_start_break"):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.session_state.active_goal_id = None
                st.session_state.active_goal_title = ""
                st.session_state.task = ""
                st.rerun()
    else:
        # Custom
        current_cats = get_user_settings(user).get("custom_categories", ["Learning","Projects","Research","Planning"])
        cat_options = current_cats + ["+ Add New"]
        selected = st.selectbox("📂 Custom Category", cat_options, key="custom_cat_select")
        if selected == "+ Add New":
            new_cat = st.text_input("New category", placeholder="e.g., Marketing", key="custom_new_cat")
            if new_cat and st.button("✅ Add Category", key="btn_add_cat"):
                if new_cat not in current_cats:
                    from user_management import users_collection, get_user_settings as _gus
                    users_collection.update_one({"username": user}, {"$addToSet": {"custom_categories": new_cat}})
                    _gus.clear()
                    st.success("Added!")
                    st.rerun()
            category_label = new_cat if new_cat else ""
        else:
            category_label = selected
        task = st.text_input("Task (micro-task)", placeholder="e.g., Draft outreach emails", key="custom_task_input")

        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = category_label
        st.session_state.task = task

        colw, colb = st.columns(2)
        with colw:
            disabled = not (category_label and task.strip())
            if st.button("▶️ Start Work (25m)", type="primary", use_container_width=True, disabled=disabled, key="btn_custom_work"):
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
        with colb:
            if st.button("☕ Break (5m)", use_container_width=True, key="btn_custom_break"):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.session_state.active_goal_id = None
                st.session_state.active_goal_title = ""
                st.session_state.task = ""
                st.rerun()

    # Today summary
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
            ratio = safe_div(breaks_today, max(1,len(work_today)))
            label = "⚖️ Balanced" if 0.3<=ratio<=0.7 else ("🎯 More focus" if ratio>0.7 else "🧘 Take breaks")
            st.metric("Breaks", breaks_today, help=label)
        with col4:
            current_target = get_daily_target(user)
            if current_target:
                pct = (len(work_today)/max(1,int(current_target)))*100
                st.metric("Target Progress", f"{pct:.0f}%")
            else:
                st.metric("Target Progress", "—")
