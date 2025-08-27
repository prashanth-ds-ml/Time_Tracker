# =============================
# app.py — Main page (Streamlit multipage app)
# =============================
import os
import time
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List

import pytz
import pandas as pd
import streamlit as st
from pymongo import MongoClient, UpdateOne

# -------- App Config --------
st.set_page_config(
    page_title="Focus Timer • Weekly Priorities",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Focus Timer - Weekly Priorities & Pomodoro Management (MVP)"},
)

IST = pytz.timezone("Asia/Kolkata")
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"

# -------- Secrets / Settings --------
MONGO_URI = st.secrets.get("mongo", {}).get("uri", os.getenv("MONGO_URI", "mongodb://localhost:27017"))
DB_NAME   = st.secrets.get("mongo", {}).get("db", os.getenv("DB_NAME", "time_tracker_db"))
DEFAULT_USER = st.secrets.get("auth", {}).get("user", os.getenv("USER_KEY", "prashanth"))

# -------- Utilities --------
@st.cache_resource(show_spinner=False)
def _mongo_client(uri: str):
    return MongoClient(uri)

@st.cache_resource(show_spinner=False)
def get_db():
    return _mongo_client(MONGO_URI)[DB_NAME]

def now_ist() -> datetime:
    return datetime.now(tz=IST)

def iso_week_key(dt_ist: datetime) -> str:
    iso = dt_ist.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

# -------- Session Identity --------
st.sidebar.subheader("👤 User")
user = st.sidebar.text_input("User ID", value=st.session_state.get("user", DEFAULT_USER))
st.session_state["user"] = user

# -------- Daily Target Quick Set --------
st.sidebar.subheader("🎯 Today’s Target")
with st.sidebar:
    target_today = st.number_input("Pomodoros", min_value=0, max_value=32, step=1, value=st.session_state.get("target_today", 6))
    if st.button("Save Target"):
        db = get_db()
        today = now_ist().date().strftime("%Y-%m-%d")
        db.daily_targets.update_one(
            {"user": user, "date": today},
            {"$set": {"user": user, "date": today, "target_pomos": int(target_today), "source": "user", "updated_at": now_ist()},
             "$setOnInsert": {"created_at": now_ist()}},
            upsert=True,
        )
        st.session_state["target_today"] = int(target_today)
        st.success("Saved!")

st.title("📑 Focus Timer — Weekly Priorities")
st.caption("Main dashboard. Use the sidebar to set today’s target. Go to the Focus page to run timers.")

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    st.subheader("📅 This Week")
    db = get_db()
    wk = iso_week_key(now_ist())
    plan = db.weekly_plans.find_one({"user": user, "week_key": wk})
    planned_total = sum(int(it.get("planned_plus_carry", it.get("planned", 0))) for it in (plan.get("items", []) if plan else []))
    actual_total = db.sessions.count_documents({"user": user, "week_key": wk, "t": "W"})
    adh = min(actual_total / planned_total, 1) * 100 if planned_total else 0
    st.metric("Planned Pomodoros", planned_total)
    st.metric("Completed (Work)", actual_total)
    st.progress(int(adh))

with colB:
    st.subheader("🧠 Deep Work")
    deep = db.sessions.count_documents({"user": user, "week_key": wk, "t": "W", "deep_work": True})
    totalW = db.sessions.count_documents({"user": user, "week_key": wk, "t": "W"})
    deep_pct = (deep / totalW * 100) if totalW else 0
    st.metric("Deep‑Work Sessions", deep)
    st.metric("Deep‑Work %", f"{deep_pct:0.1f}%")

with colC:
    st.subheader("🧾 Reflection")
    # reflection completion this week
    # get all dates M..S of this week
    today = now_ist().date()
    monday = today - timedelta(days=today.weekday())
    dates = [(monday + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    refl = db.reflections.count_documents({"user": user, "date": {"$in": dates}, "reflection_submitted": True})
    st.metric("Days Reflected", f"{refl}/7")

st.divider()

st.subheader("🚀 Quick Links")
# Streamlit automatically shows pages in sidebar, but page links are helpful
try:
    st.page_link("pages/focus.py", label="🎯 Go to Focus Timer")
except Exception:
    st.write("Use the left sidebar Pages to open **Focus Timer**.")

st.divider()

st.subheader("📈 Weekly Glance (last 14 days)")
# Simple table: day, work sessions count
two_weeks_ago = (now_ist().date() - timedelta(days=13)).strftime("%Y-%m-%d")
cur = db.sessions.aggregate([
    {"$match": {"user": user, "date": {"$gte": two_weeks_ago}, "t": "W"}},
    {"$group": {"_id": "$date", "work": {"$sum": 1}}},
    {"$sort": {"_id": 1}},
])
rows = list(cur)
if rows:
    df = pd.DataFrame({"date": [r["_id"] for r in rows], "work": [r["work"] for r in rows]})
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No sessions yet — head to the Focus Timer page to start your first Pomodoro.")


# =============================
# pages/focus.py — Focus Timer page
# =============================
# Save this file under a folder named `pages/` as `focus.py` to enable multipage routing.

import os as _os
import time as _time
from datetime import datetime as _dt, timedelta as _td
from typing import Optional as _Optional

import pytz as _pytz
import streamlit as _st
from pymongo import MongoClient as _MongoClient

_IST = _pytz.timezone("Asia/Kolkata")

# ---------- Shared connectors (reuse secrets from main app) ----------
_MONGO_URI = _st.secrets.get("mongo", {}).get("uri", _os.getenv("MONGO_URI", "mongodb://localhost:27017"))
_DB_NAME   = _st.secrets.get("mongo", {}).get("db", _os.getenv("DB_NAME", "time_tracker_db"))
_DEFAULT_USER = _st.secrets.get("auth", {}).get("user", _os.getenv("USER_KEY", "prashanth"))

@_st.cache_resource(show_spinner=False)
def _client():
    return _MongoClient(_MONGO_URI)

@_st.cache_resource(show_spinner=False)
def _db():
    return _client()[_DB_NAME]


def _now() -> _dt:
    return _dt.now(tz=_IST)

def _wk_key(dt: _dt) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

# ---------- State keys ----------
SK = {
    "user": "user",
    "mode": "mode",             # "weekly" | "custom"
    "goal_id": "goal_id",
    "custom_task": "custom_task",
    "auto_break": "auto_break",
    "sound": "sound",
    "timer_running": "timer_running",
    "phase": "phase",           # "work" | "break"
    "start_at": "start_at",
    "end_at": "end_at",
    "planned_sec": "planned_sec",
    "seconds_left": "seconds_left",
}

# ---------- UI ----------
_st.title("🎯 Focus Timer")
_user = _st.session_state.get(SK["user"], _DEFAULT_USER)
_st.session_state[SK["user"]] = _user

col1, col2, col3 = _st.columns([1.5, 1, 1])
with col1:
    today = _now().date().strftime("%Y-%m-%d")
    _st.caption(f"Today: {today} (IST)")
with col2:
    # Daily target from DB
    tgt_doc = _db().daily_targets.find_one({"user": _user, "date": today})
    tgt = int(tgt_doc.get("target_pomos", 6)) if tgt_doc else 6
    done = _db().sessions.count_documents({"user": _user, "date": today, "t": "W"})
    _st.progress(min(int(done/tgt*100) if tgt else 0, 100), text=f"Target progress: {done}/{tgt}")
with col3:
    _st.audio("https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3")

# Mode selection
_st.write("")
mode = _st.segmented_control("Work Mode", ["weekly", "custom"], selection_mode="single", key=SK["mode"]) if hasattr(_st, "segmented_control") else _st.radio("Work Mode", ["weekly", "custom"], horizontal=True, key=SK["mode"]) 

# Weekly goals dropdown
_dbh = _db()
week_key = _wk_key(_now())
plan = _dbh.weekly_plans.find_one({"user": _user, "week_key": week_key})
items = plan.get("items", []) if plan else []
# join titles from goals
goal_title_by_id = {g["_id"]: g["title"] for g in _dbh.goals.find({"user": _user}, {"_id":1, "title":1})}
active_options = [
    (it["goal_id"], goal_title_by_id.get(it["goal_id"], it["goal_id"])) for it in items
]

if mode == "weekly":
    if not active_options:
        _st.info("No active goals in this week’s plan. Switch to **custom** or create a weekly plan.")
    goal_labels = [label for _, label in active_options]
    selected_label = _st.selectbox("Select goal", goal_labels, index=0 if goal_labels else None)
    selected_goal = None
    if goal_labels:
        # reverse map label → id
        for gid, label in active_options:
            if label == selected_label:
                selected_goal = gid
                break
    _st.session_state[SK["goal_id"]] = selected_goal
else:
    task = _st.text_input("Custom task", value=_st.session_state.get(SK["custom_task"], ""))
    _st.session_state[SK["custom_task"]] = task

# Toggles
c1, c2, c3, c4 = _st.columns(4)
with c1:
    auto_break = _st.toggle("Auto‑break (5m)", value=_st.session_state.get(SK["auto_break"], True), key=SK["auto_break"])
with c2:
    sound_on = _st.toggle("Sound", value=_st.session_state.get(SK["sound"], True), key=SK["sound"])    
with c3:
    _st.write("")
with c4:
    _st.write("")

_st.divider()

# ---------- Timer Engine (autorefresh) ----------
if _st.session_state.get(SK["timer_running"], False):
    # lightweight 1s refresh while running
    _st.experimental_rerun  # no-op reference to silence lints
    _st.autorefresh = getattr(_st, "autorefresh", None)
    if _st.autorefresh:
        _st.autorefresh(interval=1000, key="tick")

phase = _st.session_state.get(SK["phase"], "work")
seconds_left = _st.session_state.get(SK["seconds_left"], 25*60 if phase == "work" else 5*60)

# Title + big timer
_b1, _b2, _b3 = _st.columns([2, 1, 1])
with _b1:
    mm, ss = divmod(max(0, int(seconds_left)), 60)
    _st.markdown(f"## ⏱️ {mm:02d}:{ss:02d} — {'Work' if phase=='work' else 'Break'}")
with _b2:
    if not _st.session_state.get(SK["timer_running"], False):
        if _st.button("Start", type="primary"):
            planned = 25*60 if phase == "work" else 5*60
            start_at = _now()
            end_at = start_at + _td(seconds=seconds_left or planned)
            _st.session_state[SK["planned_sec"]] = planned
            _st.session_state[SK["start_at"]] = start_at
            _st.session_state[SK["end_at"]] = end_at
            _st.session_state[SK["timer_running"]] = True
            _st.rerun()
    else:
        if _st.button("Pause"):
            # freeze remaining time
            end_at = _st.session_state.get(SK["end_at"])
            now = _now()
            rem = max(0, int((end_at - now).total_seconds())) if end_at else seconds_left
            _st.session_state[SK["seconds_left"]] = rem
            _st.session_state[SK["timer_running"]] = False
            _st.rerun()
with _b3:
    if _st.button("Reset"):
        for k in [SK["timer_running"], SK["start_at"], SK["end_at"], SK["seconds_left"], SK["planned_sec"]]:
            _st.session_state.pop(k, None)
        _st.session_state[SK["phase"]] = "work"
        _st.rerun()

# Update remaining while running
if _st.session_state.get(SK["timer_running"], False):
    end_at = _st.session_state.get(SK["end_at"]) or (_now() + _td(seconds=seconds_left))
    rem = int((end_at - _now()).total_seconds())
    _st.session_state[SK["seconds_left"]] = rem
    if rem <= 0:
        # complete this session
        planned = _st.session_state.get(SK["planned_sec"], 25*60 if phase == "work" else 5*60)
        start_at = _st.session_state.get(SK["start_at"], _now() - _td(seconds=planned))
        end_at = _now()
        dur_sec = planned  # timer hit 0
        dur_min = max(1, round(dur_sec / 60))

        # Persist to Mongo
        doc = {
            "user": _user,
            "date": start_at.date().strftime("%Y-%m-%d"),
            "week_key": _wk_key(start_at),
            "t": "W" if phase == "work" else "B",
            "dur_min": int(dur_min),
            "started_at_ist": start_at,
            "ended_at_ist": end_at,
            "deep_work": (phase == "work" and dur_min >= 23),
            "context_switch": False,
            "goal_mode": _st.session_state.get(SK["mode"], "custom"),
            "goal_id": _st.session_state.get(SK["goal_id"]),
            "task": _st.session_state.get(SK["custom_task"]) if phase == "work" and _st.session_state.get(SK["mode"]) == "custom" else None,
            "cat": None,
            "break_autostart": _st.session_state.get(SK["auto_break"]) if phase == "break" else None,
            "skipped": False if phase == "break" else None,
        }
        try:
            _dbh.sessions.insert_one(doc)
        except Exception as e:
            _st.warning(f"Write failed: {e}")

        # Auto-break logic
        if phase == "work" and _st.session_state.get(SK["auto_break"], True):
            _st.session_state[SK["phase"]] = "break"
            _st.session_state[SK["planned_sec"]] = 5*60
            _st.session_state[SK["start_at"]] = _now()
            _st.session_state[SK["end_at"]] = _now() + _td(seconds=5*60)
            _st.session_state[SK["timer_running"]] = True
            _st.session_state[SK["seconds_left"]] = 5*60
            _st.rerun()
        else:
            # go idle and reset to work
            _st.session_state[SK["timer_running"]] = False
            _st.session_state[SK["phase"]] = "work"
            _st.session_state[SK["seconds_left"]] = 25*60
            _st.session_state.pop(SK["start_at"], None)
            _st.session_state.pop(SK["end_at"], None)
            _st.success("Session saved!")

_st.divider()

# ---------- Today Log ----------
_st.subheader("📜 Today’s Sessions")
cur = _dbh.sessions.find({"user": _user, "date": today}).sort("started_at_ist", -1)
rows = list(cur)
if rows:
    def _fmt(r):
        t = "Work" if r.get("t") == "W" else "Break"
        when = r.get("started_at_ist").strftime("%H:%M") if r.get("started_at_ist") else "—"
        goal = r.get("goal_id") or (r.get("task") or "—")
        return [when, t, r.get("dur_min", 0), goal, "✅" if r.get("deep_work") else "—"]
    data = [[* _fmt(r)] for r in rows]
    _st.dataframe(
        pd.DataFrame(data, columns=["Start", "Type", "Min", "Goal/Task", "Deep"]),
        use_container_width=True, hide_index=True
    )
else:
    _st.info("No sessions logged yet today.")

_st.divider()

# ---------- Mini Insights ----------
_st.subheader("📊 Mini Insights")
# Start-time sparkline (last 14 days)
start_rows = list(_dbh.sessions.aggregate([
    {"$match": {"user": _user, "t": "W"}},
    {"$group": {"_id": "$date", "first": {"$min": "$started_at_ist"}}},
    {"$sort": {"_id": 1}},
]))
if start_rows:
    times = []
    for r in start_rows[-14:]:
        dt = r.get("first")
        times.append(dt.hour * 60 + dt.minute if dt else None)
    # render as text sparkline substitute
    labels = [str(r["_id"]) for r in start_rows[-14:]]
    _st.write("First work start (mins from midnight):")
    _st.dataframe(pd.DataFrame({"date": labels, "start_min": times}), use_container_width=True, hide_index=True)
else:
    _st.caption("Start-time sparkline will appear after you log a few sessions.")
