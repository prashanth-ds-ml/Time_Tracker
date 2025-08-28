import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple

import certifi
import pytz
import pandas as pd
import numpy as np
import streamlit as st
from pymongo import MongoClient
import altair as alt
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Config / constants
# ──────────────────────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

def now_ist() -> datetime:
    return datetime.now(IST)

def today_iso() -> str:
    return now_ist().date().isoformat()

def utc_from_ist(dt_ist: datetime) -> datetime:
    return dt_ist.astimezone(timezone.utc)

def week_key_from_datestr(datestr: str) -> str:
    y, m, d = map(int, datestr.split("-"))
    dt = datetime(y, m, d)
    iso = dt.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def week_start_end_ist(dt_ist: datetime) -> Tuple[str, str, str]:
    monday = (dt_ist - timedelta(days=dt_ist.isoweekday() - 1)).date()
    sunday = monday + timedelta(days=6)
    wk = f"{dt_ist.isocalendar().year}-{dt_ist.isocalendar().week:02d}"
    return monday.isoformat(), sunday.isoformat(), wk

def monday_from_week_key(week_key: str) -> datetime:
    year, wk = map(int, week_key.split("-"))
    # Monday=1
    return IST.localize(datetime.fromisocalendar(year, wk, 1))

def prev_week_key(week_key: str) -> str:
    mon = monday_from_week_key(week_key)
    prev_mon = mon - timedelta(days=7)
    iso = prev_mon.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def pom_equiv(minutes: int) -> float:
    return round(float(minutes) / 25.0, 2)

def pct(n, d) -> float:
    return (n / d * 100.0) if d else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Focus Timer", page_icon="⏱️", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Mongo connection (cached, robust)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    uri = (st.secrets.get("MONGO_URI") or os.getenv("MONGO_URI") or os.getenv("mongo_uri") or "").strip()
    dbname = (st.secrets.get("DB_NAME") or os.getenv("DB_NAME") or "Focus_DB").strip()
    if not uri:
        st.error("MONGO_URI is not configured (set in .streamlit/secrets.toml or env).")
        st.stop()
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=8000, tlsCAFile=certifi.where())
        client.admin.command("ping")
        return client[dbname]
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        st.stop()

db = get_db()
USER_ID = (st.secrets.get("USER_ID") or os.getenv("USER_ID") or "prashanth").strip()

# ──────────────────────────────────────────────────────────────────────────────
# Data access helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_user(uid: str) -> Optional[Dict[str, Any]]:
    return db.users.find_one({"_id": uid})

def get_goals_map(uid: str) -> Dict[str, Dict[str, Any]]:
    return {g["_id"]: g for g in db.goals.find({"user": uid})}

def get_week_plan(uid: str, week_key: str) -> Optional[Dict[str, Any]]:
    return db.weekly_plans.find_one({"user": uid, "week_key": week_key})

def upsert_week_plan(uid: str, week_key: str, week_start: str, week_end: str,
                     capacity: Dict[str, int], items: List[Dict[str, Any]]):
    _id = f"{uid}|{week_key}"
    now = datetime.now(timezone.utc)
    db.weekly_plans.update_one(
        {"_id": _id},
        {"$setOnInsert": {"_id": _id, "user": uid, "created_at": now, "schema_version": 1},
         "$set": {
             "week_key": week_key, "week_start": week_start, "week_end": week_end,
             "capacity": capacity, "items": items, "updated_at": now
         }},
        upsert=True
    )

def upsert_daily_target(uid: str, date_ist: str, target_pomos: int, target_minutes: Optional[int] = None):
    _id = f"{uid}|{date_ist}"
    now = datetime.now(timezone.utc)
    db.daily_targets.update_one(
        {"_id": _id},
        {"$setOnInsert": {"_id": _id, "user": uid, "date_ist": date_ist, "created_at": now, "schema_version": 1},
         "$set": {"target_pomos": int(target_pomos),
                  "target_minutes": int(target_minutes or target_pomos * 25),
                  "source": "user", "updated_at": now}},
        upsert=True
    )

def get_daily_target(uid: str, date_ist: str) -> Optional[Dict[str, Any]]:
    return db.daily_targets.find_one({"user": uid, "date_ist": date_ist})

def sum_pe_for(uid: str, week_key: str, goal_id: str, bucket: str) -> float:
    pipeline = [
        {"$match": {"user": uid, "week_key": week_key, "t": "W",
                    "goal_id": goal_id, "alloc_bucket": bucket}},
        {"$group": {"_id": None,
                    "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
    ]
    doc = next(iter(db.sessions.aggregate(pipeline)), None)
    return float(doc["pe"]) if doc else 0.0

def total_day_pe(uid: str, date_ist: str) -> float:
    pipeline = [
        {"$match": {"user": uid, "date_ist": date_ist, "t": "W"}},
        {"$group": {"_id": None,
                    "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
    ]
    doc = next(iter(db.sessions.aggregate(pipeline)), None)
    return float(doc["pe"]) if doc else 0.0

def determine_alloc_bucket(uid: str, week_key: str, goal_id: str, planned_current: int) -> str:
    done_current_pe = sum_pe_for(uid, week_key, goal_id, "current")
    return "current" if done_current_pe + 1e-6 < float(planned_current) else "backlog"

def insert_session(
    user_id: str,
    t: str,                 # "W" or "B"
    dur_min: int,
    ended_at_ist: datetime,
    *,
    kind: Optional[str] = None,               # "focus" or "activity" (for W)
    activity_type: Optional[str] = None,      # exercise/meditation/breathing/other
    intensity: Optional[str] = None,
    deep_work: Optional[bool] = None,
    goal_mode: Optional[str] = None,          # "weekly" or "custom"
    goal_id: Optional[str] = None,
    task: Optional[str] = None,
    cat: Optional[str] = None,
    alloc_bucket: Optional[str] = None,
    break_autostart: Optional[bool] = None,
    skipped: Optional[bool] = None,
    post_checkin: Optional[Dict[str, Any]] = None,
    device: Optional[str] = "web"
) -> str:
    pe = pom_equiv(dur_min)
    started_at_ist = ended_at_ist - timedelta(minutes=dur_min)

    date_ist = started_at_ist.astimezone(IST).date().isoformat()
    week_key = week_key_from_datestr(date_ist)

    sid = f"{user_id}|{date_ist}|{t}|{int(started_at_ist.timestamp())}|{dur_min}"
    now = datetime.now(timezone.utc)
    doc = {
        "_id": sid, "user": user_id, "date_ist": date_ist, "week_key": week_key,
        "t": t, "kind": kind, "activity_type": activity_type, "intensity": intensity,
        "dur_min": int(dur_min), "pom_equiv": pe,
        "started_at_ist": utc_from_ist(started_at_ist), "ended_at_ist": utc_from_ist(ended_at_ist),
        "deep_work": deep_work, "context_switch": False,
        "goal_mode": goal_mode, "goal_id": goal_id, "task": task, "cat": cat,
        "alloc_bucket": alloc_bucket, "break_autostart": break_autostart, "skipped": skipped,
        "post_checkin": post_checkin, "device": device,
        "created_at": now, "updated_at": now, "schema_version": 1
    }
    db.sessions.update_one({"_id": sid}, {"$setOnInsert": doc, "$set": {"updated_at": now}}, upsert=True)
    return sid

def list_today_sessions(uid: str, date_ist: str) -> List[Dict[str, Any]]:
    return list(db.sessions.find({"user": uid, "date_ist": date_ist}).sort("started_at_ist", 1))

def delete_last_today_session(uid: str, date_ist: str) -> Optional[str]:
    last = db.sessions.find({"user": uid, "date_ist": date_ist}).sort("started_at_ist", -1).limit(1)
    last = next(iter(last), None)
    if last:
        db.sessions.delete_one({"_id": last["_id"]})
        return last["_id"]
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: connection + week summary
# ──────────────────────────────────────────────────────────────────────────────
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

today = today_iso()
today_dt_ist = now_ist()
week_start, week_end, week_key = week_start_end_ist(today_dt_ist)
goals_map = get_goals_map(USER_ID)
plan = get_week_plan(USER_ID, week_key)

st.sidebar.subheader(f"📅 Week {week_key}")
if plan:
    st.sidebar.caption(f"{week_start} → {week_end}")
    cap = plan.get("capacity", {})
    st.sidebar.write(f"Capacity: **{cap.get('total', 0)}** poms (wkday {cap.get('weekday',0)}, wkend {cap.get('weekend',0)})")
    for it in sorted(plan.get("items", []), key=lambda x: x.get("priority_rank", 99))[:5]:
        g = goals_map.get(it["goal_id"], {})
        st.sidebar.write(f"- {g.get('title','?')} — **{it['planned_current']}** + backlog {it['backlog_in']}")
else:
    st.sidebar.info("No weekly plan for this week yet.")

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_timer, tab_planner, tab_analytics = st.tabs(["⏱️ Timer & Log", "🗂️ Weekly Planner", "📈 Analytics"])

# =============================================================================
# TAB 1: Timer & Log (manual + Live Timer)
# =============================================================================
with tab_timer:
    st.header("⏱️ Focus Timer")
    st.caption(f"IST Date: **{today}** • Week: **{week_key}**")

    # Today Target
    st.subheader("🎯 Today’s Target")
    tgt = get_daily_target(USER_ID, today)
    target_val = (tgt or {}).get("target_pomos", None)
    colT1, colT2, colT3 = st.columns([1.2, 0.8, 1])
    with colT1:
        st.metric("Target (poms)", value=target_val if target_val is not None else "—")
        actual_pe = total_day_pe(USER_ID, today)
        st.progress(min(actual_pe / float(target_val or 1), 1.0), text=f"Progress: {actual_pe:.1f} / {target_val or 0} pe")
    with colT2:
        with st.form("target_form", clear_on_submit=False):
            new_target = st.number_input("Set/Update target", min_value=0, max_value=50, value=int(target_val or 6), step=1, key="target_inp")
            submitted = st.form_submit_button("Save target", use_container_width=True)
            if submitted:
                upsert_daily_target(USER_ID, today, int(new_target))
                st.success("Saved target.")
                st.experimental_rerun()
    with colT3:
        todays = list_today_sessions(USER_ID, today)
        focus_cnt = sum(1 for s in todays if s.get("t") == "W" and s.get("kind") != "activity")
        breaks_valid = sum(1 for s in todays if s.get("t") == "B" and (s.get("dur_min",0) >= 4) and not s.get("skipped", False))
        st.metric("Focus sessions", focus_cnt)
        st.metric("Valid breaks", breaks_valid)

    st.divider()

    # ── Manual logger
    st.subheader("🎛️ Log a Session (manual)")
    try:
        sess_type = st.segmented_control("Session Type", options=["Work (focus)", "Work (activity)", "Break"], default="Work (focus)")
    except Exception:
        sess_type = st.radio("Session Type", options=["Work (focus)", "Work (activity)", "Break"], index=0, horizontal=True)

    dur_default = 25 if "Work" in sess_type else 5

    if "Work" in sess_type:
        with st.form("work_form", clear_on_submit=True):
            dur_min = st.number_input("Duration (minutes)", min_value=1, max_value=180, value=dur_default, step=1)
            ended_now = st.checkbox("End at now (IST)", value=True)
            if ended_now:
                end_dt_ist = now_ist()
            else:
                tval = st.time_input("End time (IST)", value=now_ist().time())
                end_dt_ist = IST.localize(datetime.combine(today_dt_ist.date(), tval))

            post_quality = st.slider("Quality (1–5)", 1, 5, 4)
            post_mood    = st.slider("Mood (1–5)", 1, 5, 4)
            post_energy  = st.slider("Energy (1–5)", 1, 5, 4)
            post_note    = st.text_input("Note (optional)")

            kind = "focus"
            activity_type = None
            intensity = None
            deep_work = dur_min >= 23

            if sess_type == "Work (activity)":
                kind = "activity"
                activity_type = st.selectbox("Activity type", ["exercise", "meditation", "breathing", "other"], index=1)
                intensity = st.selectbox("Intensity", ["light", "moderate", "vigorous"], index=0)
                deep_work = None

            mode = st.radio("Work mode", options=["Weekly plan goal", "Custom / unplanned"], horizontal=True)
            goal_id = None
            goal_mode = "custom"
            cat = None
            alloc_bucket = None
            task_text = None

            if mode == "Weekly plan goal" and plan:
                items = sorted(plan.get("items", []), key=lambda x: x.get("priority_rank", 99))
                options = []
                for it in items:
                    gid = it["goal_id"]
                    gtitle = goals_map.get(gid, {}).get("title", gid)
                    options.append((f"{gtitle} — planned {it['planned_current']} (backlog {it['backlog_in']})", gid, it["planned_current"]))
                label = st.selectbox("Choose goal", options=[o[0] for o in options])
                sel = next((o for o in options if o[0] == label), None)
                if sel:
                    goal_id = sel[1]
                    goal_mode = "weekly"
                    alloc_bucket = determine_alloc_bucket(USER_ID, week_key, goal_id, sel[2])
                    cat = goals_map.get(goal_id, {}).get("category")
            else:
                goal_mode = "custom"
                task_text = st.text_input("What did you work on? (short note)")
                cat = st.selectbox("Category (for analytics)", ["Learning","Projects","Certification","Career","Health","Wellbeing","Other"], index=0)

            submit_work = st.form_submit_button("Log Work", use_container_width=True)
            if submit_work:
                sid = insert_session(
                    USER_ID, "W", int(dur_min), end_dt_ist,
                    kind=kind, activity_type=activity_type, intensity=intensity,
                    deep_work=deep_work, goal_mode=goal_mode,
                    goal_id=(goal_id if goal_mode=="weekly" else None),
                    task=(None if goal_mode=="weekly" else task_text),
                    cat=cat, alloc_bucket=(alloc_bucket if goal_mode=="weekly" else None),
                    break_autostart=True, skipped=None,
                    post_checkin={"quality_1to5": int(post_quality),
                                  "mood_1to5": int(post_mood),
                                  "energy_1to5": int(post_energy),
                                  "distraction": None,
                                  "note": (post_note or None)},
                    device="web"
                )
                st.success(f"Logged work. id={sid}")
                st.experimental_rerun()
    else:
        with st.form("break_form", clear_on_submit=True):
            dur_min = st.number_input("Duration (minutes)", min_value=1, max_value=60, value=dur_default, step=1)
            ended_now = st.checkbox("End at now (IST)", value=True)
            if ended_now:
                end_dt_ist = now_ist()
            else:
                tval = st.time_input("End time (IST)", value=now_ist().time())
                end_dt_ist = IST.localize(datetime.combine(today_dt_ist.date(), tval))
            skipped = st.checkbox("Skipped?", value=False)

            submit_break = st.form_submit_button("Log Break", use_container_width=True)
            if submit_break:
                sid = insert_session(
                    USER_ID, "B", int(dur_min), end_dt_ist,
                    kind=None, activity_type=None, intensity=None,
                    deep_work=None, goal_mode=None, goal_id=None, task=None, cat=None,
                    alloc_bucket=None, break_autostart=None, skipped=bool(skipped),
                    post_checkin=None, device="web"
                )
                st.success(f"Logged break. id={sid}")
                st.experimental_rerun()

    st.divider()

    # ── Live Timer (auto-insert + optional auto-break)
    st.subheader("⏳ Live Timer (beta)")
    if "timer" not in st.session_state:
        st.session_state.timer = {
            "running": False, "end_ts": None, "started_at": None, "completed": False,
            "t": None, "dur_min": 25, "kind": "focus", "activity_type": None, "intensity": None,
            "deep_work": True, "goal_mode": "weekly", "goal_id": None, "task": None, "cat": None,
            "alloc_bucket": None, "auto_break": True, "break_min": 5
        }

    timer = st.session_state.timer

    with st.form("live_timer_form", clear_on_submit=False):
        colA, colB, colC = st.columns(3)
        with colA:
            live_type = st.selectbox("Type", ["Work (focus)","Work (activity)","Break"], index=0)
            dur_live = st.number_input("Duration (min)", 1, 180, value=25 if "Work" in live_type else 5)
        with colB:
            auto_break = st.checkbox("Auto-break after Work", value=True)
            break_min = st.number_input("Break length (min)", 1, 30, value=5)
        with colC:
            end_at_now = st.checkbox("Start now", value=True)

        # Goal / task params
        kind = "focus"; activity_type=None; intensity=None; deep_live = (dur_live >= 23)
        goal_mode = "weekly"; goal_id=None; cat=None; task_text=None; alloc_bucket=None

        if live_type == "Work (activity)":
            kind = "activity"; deep_live=None
            activity_type = st.selectbox("Activity type", ["exercise","meditation","breathing","other"], index=1)
            intensity = st.selectbox("Intensity", ["light","moderate","vigorous"], index=0)

        if "Work" in live_type:
            mode = st.radio("Work mode", ["Weekly plan goal","Custom / unplanned"], horizontal=True)
            if mode == "Weekly plan goal" and plan:
                items = sorted(plan.get("items", []), key=lambda x: x.get("priority_rank", 99))
                options = []
                for it in items:
                    gid = it["goal_id"]
                    gtitle = goals_map.get(gid, {}).get("title", gid)
                    options.append((f"{gtitle} — planned {it['planned_current']} (backlog {it['backlog_in']})", gid, it["planned_current"]))
                label = st.selectbox("Choose goal", options=[o[0] for o in options])
                sel = next((o for o in options if o[0] == label), None)
                if sel:
                    goal_id = sel[1]; goal_mode="weekly"
                    alloc_bucket = determine_alloc_bucket(USER_ID, week_key, goal_id, sel[2])
                    cat = goals_map.get(goal_id, {}).get("category")
            else:
                goal_mode = "custom"
                task_text = st.text_input("Task note")
                cat = st.selectbox("Category", ["Learning","Projects","Certification","Career","Health","Wellbeing","Other"], index=0)

        start_live = st.form_submit_button("▶️ Start Timer", use_container_width=True)

    if start_live and not timer["running"]:
        timer.update({
            "running": True, "completed": False, "dur_min": int(dur_live),
            "t": ("W" if "Work" in live_type else "B"),
            "kind": kind, "activity_type": activity_type, "intensity": intensity,
            "deep_work": deep_live, "goal_mode": goal_mode, "goal_id": goal_id,
            "task": task_text, "cat": cat, "alloc_bucket": alloc_bucket,
            "auto_break": bool(auto_break), "break_min": int(break_min),
            "started_at": now_ist(), "end_ts": now_ist() + timedelta(minutes=int(dur_live))
        })

    # Live view
    if timer["running"]:
        remaining = (timer["end_ts"] - now_ist()).total_seconds()
        mins = max(int(remaining // 60), 0)
        secs = max(int(remaining % 60), 0)
        st.info(f"⏳ Time left: **{mins:02d}:{secs:02d}**  •  Type: {('Work' if timer['t']=='W' else 'Break')}  •  Dur: {timer['dur_min']}m")
        st.progress(min(1.0, 1 - (remaining / max(1, timer["dur_min"] * 60))), text="Counting down...")
        st.experimental_rerun() if remaining > 0 else None

        # Controls
        colL, colR = st.columns(2)
        stop_now = colL.button("⏹️ Stop / Cancel", use_container_width=True)
        complete_early = colR.button("✅ Complete now", use_container_width=True)
        if stop_now:
            timer["running"] = False
            st.warning("Timer canceled.")
            st.experimental_rerun()
        if complete_early:
            timer["end_ts"] = now_ist()
            remaining = 0

        # Completion
        if remaining <= 0 and not timer["completed"]:
            ended_at = timer["end_ts"]
            started_at = timer["started_at"]
            dur_min_done = max(1, int(round((ended_at - started_at).total_seconds()/60.0)))
            sid = insert_session(
                USER_ID, timer["t"], dur_min_done, ended_at,
                kind=timer["kind"], activity_type=timer["activity_type"], intensity=timer["intensity"],
                deep_work=timer["deep_work"], goal_mode=timer["goal_mode"], goal_id=timer["goal_id"],
                task=timer["task"], cat=timer["cat"], alloc_bucket=timer["alloc_bucket"],
                break_autostart=(timer["t"]=="W" and timer["auto_break"]), skipped=False,
                post_checkin=None, device="web-live"
            )
            timer["completed"] = True
            timer["running"] = False
            st.success(f"Session saved. id={sid}")

            # Auto-break
            if timer["t"] == "W" and timer["auto_break"] and timer["break_min"] > 0:
                # chain break timer
                timer.update({
                    "running": True, "completed": False,
                    "t": "B", "dur_min": timer["break_min"], "kind": None,
                    "activity_type": None, "intensity": None, "deep_work": None,
                    "goal_mode": None, "goal_id": None, "task": None, "cat": None,
                    "alloc_bucket": None, "auto_break": False,
                    "started_at": now_ist(), "end_ts": now_ist() + timedelta(minutes=timer["break_min"])
                })
                st.info("Starting auto-break…")
            st.experimental_rerun()

    st.divider()
    st.subheader("📝 Today’s Sessions")
    todays = list_today_sessions(USER_ID, today)
    if not todays:
        st.info("No sessions logged yet.")
    else:
        def fmt_row(s):
            kindlab = "Work" if s.get("t") == "W" else "Break"
            if s.get("kind") == "activity": kindlab = "Activity"
            goal_title = goals_map.get(s.get("goal_id"), {}).get("title") if s.get("goal_id") else (s.get("task") or "—")
            return {
                "When (IST)": s.get("started_at_ist").astimezone(IST).strftime("%H:%M"),
                "Type": kindlab, "Dur (min)": s.get("dur_min"), "PE": s.get("pom_equiv"),
                "Goal/Task": goal_title, "Bucket": s.get("alloc_bucket") or "—",
                "Deep": "✓" if s.get("deep_work") else "—",
            }
        st.dataframe([fmt_row(s) for s in todays], use_container_width=True, hide_index=True)
        if st.button("↩️ Undo last entry", use_container_width=True):
            deleted = delete_last_today_session(USER_ID, today)
            st.warning(f"Deleted last session: {deleted}" if deleted else "Nothing to undo.")
            st.experimental_rerun()

# =============================================================================
# TAB 2: Weekly Planner
# =============================================================================
with tab_planner:
    st.header("🗂️ Weekly Planner")
    # Choose week (default current)
    colW1, colW2 = st.columns([1,1])
    with colW1:
        wk = st.text_input("Week key (YYYY-WW)", value=week_key)
    with colW2:
        # capacity inputs (defaults from user prefs if available)
        udoc = get_user(USER_ID) or {}
        prefs = (udoc.get("prefs") or {})
        wkday_default = prefs.get("weekday_poms", 3)
        wkend_default = prefs.get("weekend_poms", 6)
        wkday = st.number_input("Weekday poms (per day)", 0, 20, value=int(wkday_default))
        wkend = st.number_input("Weekend poms (per day)", 0, 30, value=int(wkend_default))
        total_capacity = wkday*5 + wkend*2
    st.caption(f"Total capacity calculated for the week: **{total_capacity}** poms.")

    # Load existing plan if any (for chosen week)
    existing = get_week_plan(USER_ID, wk)
    prev_wk = prev_week_key(wk)
    prev_plan = get_week_plan(USER_ID, prev_wk)

    # Prefill carryover from prev week
    prefill = st.checkbox(f"Prefill backlog from previous week ({prev_wk})", value=True)

    # Build editable goals table
    goals = list(db.goals.find({"user": USER_ID}))
    rank_weight_map = (prefs.get("rank_weight_map") or {"1":5,"2":3,"3":2,"4":1,"5":1})
    rank_choices = ["1","2","3","4","5"]

    # Prepare rows
    def carryover_for(gid: str) -> int:
        if not prev_plan:
            return 0
        # compute leftover = total_target - actual (rounded) last week
        it_prev = next((it for it in prev_plan.get("items", []) if it["goal_id"] == gid), None)
        if not it_prev:
            return 0
        total_target = int(it_prev.get("total_target", 0))
        # actual PE across both buckets
        pe_doc = next(iter(db.sessions.aggregate([
            {"$match": {"user": USER_ID, "week_key": prev_wk, "t":"W", "goal_id": gid}},
            {"$group": {"_id": None, "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
        ])), None)
        actual = int(round(float(pe_doc["pe"]) if pe_doc else 0.0))
        return max(total_target - actual, 0)

    # Merge with existing plan items if present
    existing_items = {it["goal_id"]: it for it in (existing.get("items", []) if existing else [])}

    rows = []
    for g in goals:
        gid = g["_id"]
        ex = existing_items.get(gid)
        backlog_in = int(ex["backlog_in"]) if ex else (carryover_for(gid) if prefill else 0)
        rank_str = str(ex["priority_rank"]) if ex else "3"
        rows.append({
            "include": bool(ex) or True,  # default include
            "goal_id": gid,
            "title": g.get("title",""),
            "category": g.get("category",""),
            "rank": rank_str,
            "weight": int(rank_weight_map.get(rank_str, 1)),
            "planned_current": int(ex["planned_current"]) if ex else 0,
            "backlog_in": backlog_in,
            "status_at_plan": ex["status_at_plan"] if ex else "In Progress",
            "notes": ex["notes"] if ex else ""
        })

    df = pd.DataFrame(rows)
    edited = st.data_editor(
        df,
        column_config={
            "include": st.column_config.CheckboxColumn("Include"),
            "rank": st.column_config.SelectboxColumn("Rank (1 high)", options=rank_choices, width="small"),
            "status_at_plan": st.column_config.SelectboxColumn("Status", options=["In Progress","On Hold","Completed"]),
            "planned_current": st.column_config.NumberColumn("Planned (current)", step=1, min_value=0),
            "backlog_in": st.column_config.NumberColumn("Backlog In", step=1, min_value=0),
            "notes": st.column_config.TextColumn("Notes"),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed"
    )

    # Auto-allocate based on ranks/weights
    colA1, colA2, colA3 = st.columns([1,1,1])
    auto_go = colA1.button("⚖️ Auto-allocate by rank")
    clear_plan = colA2.button("🧹 Clear planned_current")
    save_plan = colA3.button("💾 Save plan")

    if auto_go:
        m = edited[edited["include"]==True].copy()
        if m.empty or total_capacity <= 0:
            st.warning("Select at least one goal and set capacity > 0.")
        else:
            m["weight"] = m["rank"].map(lambda r: int(rank_weight_map.get(str(r), 1)))
            weights_sum = m["weight"].sum()
            shares = (m["weight"] / max(weights_sum, 1)) * total_capacity
            base = np.floor(shares).astype(int)
            left = total_capacity - base.sum()
            # distribute remainder by largest fractional part
            frac = shares - base
            order = np.argsort(-frac.values)
            for i in range(int(left)):
                base.iloc[order[i]] += 1
            # write back
            edited.loc[m.index, "planned_current"] = base.values
            st.success("Auto-allocation applied. Review and adjust if needed.")

    if clear_plan:
        edited["planned_current"] = 0
        st.info("Cleared plan allocations.")

    # Compute totals & invariants
    planned_sum = int(edited.loc[edited["include"]==True, "planned_current"].sum())
    backlog_sum = int(edited.loc[edited["include"]==True, "backlog_in"].sum())
    st.caption(f"Planned current sum: **{planned_sum}** / capacity **{total_capacity}** • Backlog in total: **{backlog_sum}**")
    if planned_sum != total_capacity:
        st.warning("Sum of planned_current should equal capacity total for the week.")
    else:
        st.success("Planned_current matches capacity total ✅")

    # Save
    if save_plan:
        mon = monday_from_week_key(wk).date().isoformat()
        sun = (monday_from_week_key(wk).date() + timedelta(days=6)).isoformat()
        items = []
        for _, r in edited.iterrows():
            if not r["include"]:
                continue
            pc = int(r["planned_current"])
            bi = int(r["backlog_in"])
            items.append({
                "goal_id": r["goal_id"],
                "priority_rank": int(r["rank"]),
                "weight": int(rank_weight_map.get(str(r["rank"]), 1)),
                "planned_current": pc,
                "backlog_in": bi,
                "total_target": pc + bi,
                "status_at_plan": r.get("status_at_plan","In Progress"),
                "close_action": None,
                "notes": r.get("notes") or None
            })
        cap = {"weekday": int(wkday), "weekend": int(wkend), "total": int(total_capacity)}
        upsert_week_plan(USER_ID, wk, mon, sun, cap, items)
        st.success(f"Plan saved for week {wk}.")
        if wk == week_key:
            st.experimental_rerun()

# =============================================================================
# TAB 3: Analytics
# =============================================================================
with tab_analytics:
    st.header("📈 Analytics")

    # Choose range (last N weeks)
    weeks_sessions = sorted(db.sessions.distinct("week_key", {"user": USER_ID, "t": "W"}))
    weeks_plans = sorted(db.weekly_plans.distinct("week_key", {"user": USER_ID}))
    all_weeks = sorted(set(weeks_sessions) | set(weeks_plans))
    if not all_weeks:
        st.info("No data yet.")
    else:
        last_n = st.slider("Show last N weeks", 1, min(12, len(all_weeks)), value=min(6, len(all_weeks)))
        weeks_view = all_weeks[-last_n:]

        # compute metrics per week
        rows = []
        for W in weeks_view:
            planW = get_week_plan(USER_ID, W)
            planned = sum(int(it.get("planned_current", 0)) for it in (planW.get("items", []) if planW else []))

            # actual PE
            pe_doc = next(iter(db.sessions.aggregate([
                {"$match": {"user": USER_ID, "week_key": W, "t": "W"}},
                {"$group": {"_id": None, "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
            ])), None)
            actual_pe = float(pe_doc["pe"]) if pe_doc else 0.0

            focus_total = db.sessions.count_documents({"user": USER_ID, "week_key": W, "t": "W", "kind": {"$ne": "activity"}})
            deep = db.sessions.count_documents({"user": USER_ID, "week_key": W, "t": "W", "kind": {"$ne": "activity"}, "deep_work": True})
            deep_pct = pct(deep, focus_total)

            valid_breaks = db.sessions.count_documents({
                "user": USER_ID, "week_key": W, "t": "B", "dur_min": {"$gte": 4},
                "$or": [{"skipped": {"$exists": False}}, {"skipped": {"$ne": True}}]
            })
            break_pct = pct(min(valid_breaks, focus_total), focus_total)

            pe_by_mode = {row["_id"]: row["pe"] for row in db.sessions.aggregate([
                {"$match": {"user": USER_ID, "week_key": W, "t": "W"}},
                {"$group": {"_id": "$goal_mode", "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
            ])}
            unplan_pct = pct(float(pe_by_mode.get("custom", 0.0)), actual_pe)

            adh = pct(min(actual_pe, planned), planned) if planned else 0.0
            rows.append({"week": W, "planned": planned, "actual_pe": round(actual_pe,1), "adherence_pct": round(adh,1),
                         "deep_pct": round(deep_pct,1), "break_pct": round(break_pct,1), "unplanned_pct": round(unplan_pct,1)})

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Adherence % (line)")
            st.line_chart(df.set_index("week")["adherence_pct"])
        with c2:
            st.subheader("Deep Work % (bar)")
            st.bar_chart(df.set_index("week")["deep_pct"])

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Unplanned % (bar)")
            st.bar_chart(df.set_index("week")["unplanned_pct"])
        with c4:
            st.subheader("Actual PE (bars)")
            st.bar_chart(df.set_index("week")["actual_pe"])

        st.divider()
        st.subheader("Career vs Wellbeing (selected week)")
        sel_week = st.selectbox("Pick a week", weeks_view, index=len(weeks_view)-1)

        # domain map (goals)
        goal_domain = {}
        for g in db.goals.find({"user": USER_ID}, {"_id":1, "category":1}):
            cat = (g.get("category") or "").lower()
            goal_domain[g["_id"]] = "Wellbeing" if cat in ["health","wellbeing"] else "Career"

        career = 0.0; wellbeing = 0.0
        for s in db.sessions.find({"user": USER_ID, "week_key": sel_week, "t":"W"},
                                  {"goal_id":1,"kind":1,"activity_type":1,"pom_equiv":1,"dur_min":1}):
            pe = s.get("pom_equiv") or (s.get("dur_min",0)/25.0)
            if s.get("kind") == "activity":
                wellbeing += pe
            else:
                dom = goal_domain.get(s.get("goal_id"), "Career")
                if dom == "Wellbeing": wellbeing += pe
                else: career += pe
        total = career + wellbeing
        cp = pct(career, total); wp = pct(wellbeing, total)

        fig, ax = plt.subplots()
        ax.pie([career, wellbeing], labels=[f"Career {cp:.1f}%", f"Wellbeing {wp:.1f}%"], autopct="%1.1f%%", startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.caption("Focus Timer • Mongo-backed • IST-aware • Planner + Timer + Analytics")
