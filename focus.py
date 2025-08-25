# focus.py
import streamlit as st
import time
from datetime import timedelta
from db import (
    now_ist, week_bounds,
    get_sessions_df, get_or_create_week_plan,
    registry_defaults, update_registry_defaults,
    list_registry_goals, append_session
)

POMODORO_MIN = 25
BREAK_MIN = 5
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"

def sound_alert():
    st.components.v1.html(f"""
        <audio id="beep" autoplay>
            <source src="{SOUND_PATH}" type="audio/mpeg">
        </audio>
        <script>
            const a = document.getElementById('beep');
            if (a) {{ a.volume = 0.6; a.play().catch(()=>{{}}); }}
        </script>
    """, height=0)

def _time_to_minutes(tstr):
    try:
        import datetime as _dt
        dt = _dt.datetime.strptime(tstr, "%I:%M %p")
        return dt.hour*60 + dt.minute
    except Exception:
        return None

def _this_week_progress(user: str):
    df = get_sessions_df(user)
    if df.empty:
        st.info("No sessions yet for this week.")
        return
    ws, we = week_bounds(now_ist().date())
    df["date_only"] = df["date"].dt.date
    week = df[(df["date_only"]>=ws) & (df["date_only"]<=we) & (df["pomodoro_type"]=="Work")].copy()
    if week.empty:
        st.info("No work sessions yet this week.")
        return

    # by goal
    by_goal = week.groupby("goal_id").size().sort_values(ascending=False).reset_index(name="actual")
    # label None as "Custom"
    by_goal["goal"] = by_goal["goal_id"].apply(lambda x: "Custom (Unplanned)" if pd.isna(x) else str(x))
    # sparkline start-time (median by day)
    dfw = week[["date_only", "time"]].copy()
    dfw["mins"] = dfw["time"].apply(_time_to_minutes)
    dfw = dfw.dropna()
    if not dfw.empty:
        trend = dfw.groupby("date_only")["mins"].median().reset_index().set_index("date_only")
        st.line_chart(trend, height=150)

import pandas as pd

def this_week_glance(user: str):
    st.subheader("📌 This Week at a Glance")
    plan = get_or_create_week_plan(user, now_ist().date())
    alloc = plan.get("allocations_by_goal", {}) or {}
    embedded = plan.get("goals_embedded", []) or []
    if not alloc:
        st.info("No allocations yet. Add a plan in the Weekly Planner.")
        return
    df = get_sessions_df(user)
    if df.empty:
        df = pd.DataFrame(columns=["date","time","pomodoro_type","duration","user","goal_id","task","category"])
        df["date"] = pd.to_datetime(df["date"])
    ws = pd.to_datetime(plan["week_start"]).date()
    we = pd.to_datetime(plan["week_end"]).date()
    df["date_only"] = df["date"].dt.date
    week = df[(df["date_only"]>=ws) & (df["date_only"]<=we) & (df["pomodoro_type"]=="Work")].copy()
    counts = week.groupby("goal_id").size().to_dict()

    cols = st.columns(2)
    i = 0
    for g in embedded:
        gid = g["goal_id"]
        title = g.get("title","(goal)")
        planned = int(alloc.get(gid, 0))
        actual = int(counts.get(gid, 0))
        pct = 0.0 if planned<=0 else min(1.0, actual/max(1,planned))
        with cols[i % 2]:
            st.write(f"**{title}**")
            st.progress(pct, text=f"{actual}/{planned}")
        i += 1

def render_timer_widget(user: str, auto_break: bool) -> bool:
    if not st.session_state.get("start_time"):
        return False
    duration = BREAK_MIN*60 if st.session_state.get("is_break") else POMODORO_MIN*60
    remaining = int(st.session_state["start_time"] + duration - time.time())
    if remaining > 0:
        mins, secs = divmod(remaining, 60)
        session_type = "Break" if st.session_state.get("is_break") else f"Working on: {st.session_state.get('task','')}"
        st.subheader(f"{'🧘' if st.session_state.get('is_break') else '💼'} {session_type}")
        _, cc, _ = st.columns([1,2,1])
        with cc:
            st.markdown(f"<h1 style='text-align:center;font-size:4rem;'>⏱️ {mins:02d}:{secs:02d}</h1>", unsafe_allow_html=True)
        st.progress(1 - (remaining/duration))
        st.info("🧘 Relax" if st.session_state.get("is_break") else "💪 Stay focused!")
        time.sleep(1)
        st.rerun()
        return True
    else:
        # save & beep BEFORE auto break
        was_break = bool(st.session_state.get("is_break"))
        append_session(
            user=user,
            is_break=was_break,
            minutes=(BREAK_MIN if was_break else POMODORO_MIN),
            time_str=now_ist().strftime("%I:%M %p"),
            goal_id=st.session_state.get("active_goal_id"),
            task=st.session_state.get("task",""),
            category=st.session_state.get("active_goal_title","")
        )
        sound_alert()
        st.balloons()
        st.success("🎉 Session complete!")

        # reset
        st.session_state.start_time = None
        st.session_state.is_break = False
        st.session_state.task = ""
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = ""

        if (not was_break) and auto_break:
            st.toast("☕ Auto-starting a 5-minute break")
            st.session_state.start_time = time.time()
            st.session_state.is_break = True
            st.rerun()
        return True

def render_focus_timer(user: str):
    st.header("🎯 Focus Timer")

    # defaults + toggles
    defs = registry_defaults(user)
    col, _ = st.columns([1,3])
    with col:
        auto_break_ui = st.toggle("Auto-start break", value=bool(defs.get("auto_break", True)))
        if auto_break_ui != bool(defs.get("auto_break", True)):
            update_registry_defaults(user, defs.get("weekday_poms",3), defs.get("weekend_poms",5), auto_break=auto_break_ui)

    # active timer?
    if render_timer_widget(user, auto_break=bool(registry_defaults(user).get("auto_break", True))):
        return

    # daily target quick-progress
    df = get_sessions_df(user)
    today = now_ist().date()
    if not df.empty:
        df["date_only"] = df["date"].dt.date
        today_work = df[(df["date_only"]==today) & (df["pomodoro_type"]=="Work")]
        st.metric("Today's Work Sessions", int(len(today_work)))

    st.divider()
    this_week_glance(user)

    st.divider()
    st.subheader("Start a Session")

    plan = get_or_create_week_plan(user, now_ist().date())
    embedded = plan.get("goals_embedded", []) or []
    titles_pairs = [(g.get("title","(goal)"), g.get("goal_id")) for g in embedded]
    titles_only = [t for (t, _) in titles_pairs] or ["(no goals)"]

    mode = st.radio("Mode", ["Weekly Goal", "Custom"], horizontal=True)
    if mode == "Weekly Goal":
        c1, c2 = st.columns([1,2])
        with c1:
            sel_idx = st.selectbox("Goal", options=range(len(titles_only)),
                                   format_func=lambda i: titles_only[i],
                                   disabled=(len(titles_pairs)==0))
            gid = titles_pairs[sel_idx][1] if titles_pairs else None
            gtitle = titles_pairs[sel_idx][0] if titles_pairs else ""
        with c2:
            task = st.text_input("Task (micro-task)", placeholder="e.g., Draft section 2 notes")
        st.session_state.active_goal_id = gid
        st.session_state.active_goal_title = gtitle
        st.session_state.task = task
        colw, colb = st.columns(2)
        with colw:
            if st.button("▶️ Start Work (25m)", type="primary", use_container_width=True,
                         disabled=(gid is None or not task.strip())):
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
        cats = list(defs.get("custom_categories", ["Learning","Projects","Research","Planning"]))
        choice = st.selectbox("Category", cats + ["+ Add New"])
        if choice == "+ Add New":
            new_cat = st.text_input("New category")
            if new_cat and st.button("Add"):
                cats.append(new_cat)
                update_registry_defaults(user, defs.get("weekday_poms",3), defs.get("weekend_poms",5),
                                         custom_categories=cats)
                st.success("Category added")
                st.rerun()
            category = new_cat if new_cat else ""
        else:
            category = choice
        task = st.text_input("Task (micro-task)")
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = category
        st.session_state.task = task
        colw, colb = st.columns(2)
        with colw:
            if st.button("▶️ Start Work (25m)", type="primary", use_container_width=True,
                         disabled=(not category or not task.strip())):
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
