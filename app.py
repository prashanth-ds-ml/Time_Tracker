import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import streamlit as st
from pymongo import MongoClient
import pytz

# --- Config ---
IST = pytz.timezone("Asia/Kolkata")
MONGO_URI = os.environ.get("MONGO_URI", "")
DB_NAME   = os.environ.get("DB_NAME", "Focus_DB")
USER_ID   = os.environ.get("USER_ID", "prashanth")

# --- Helpers ---
def now_ist():
    return datetime.now(IST)

def week_key_from_datestr(datestr: str) -> str:
    y, m, d = map(int, datestr.split("-"))
    dt = datetime(y, m, d)
    iso = dt.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def week_start_end_ist(dt_ist: datetime):
    monday = (dt_ist - timedelta(days=dt_ist.isoweekday() - 1)).date()
    sunday = monday + timedelta(days=6)
    return monday.isoformat(), sunday.isoformat(), f"{dt_ist.isocalendar().year}-{dt_ist.isocalendar().week:02d}"

def pom_equiv(minutes: int) -> float:
    return round(float(minutes) / 25.0, 2)

def utc_from_ist(dt_ist: datetime) -> datetime:
    return dt_ist.astimezone(timezone.utc)

# --- Streamlit page config ---
st.set_page_config(page_title="Focus Timer v1", page_icon="⏱️", layout="wide")

# --- Connection (cached) ---
@st.cache_resource
def get_db():
    if not MONGO_URI:
        st.stop()
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

db = get_db()

# --- Data access funcs ---
def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    return db.users.find_one({"_id": user_id})

def get_week_plan(user_id: str, week_key: str) -> Optional[Dict[str, Any]]:
    return db.weekly_plans.find_one({"user": user_id, "week_key": week_key})

def get_goals_map(user_id: str) -> Dict[str, Dict[str, Any]]:
    return {g["_id"]: g for g in db.goals.find({"user": user_id})}

def upsert_daily_target(user_id: str, date_ist: str, target_pomos: int, target_minutes: Optional[int] = None):
    _id = f"{user_id}|{date_ist}"
    doc = {
        "_id": _id,
        "user": user_id,
        "date_ist": date_ist,
        "target_pomos": int(target_pomos),
        "target_minutes": int(target_minutes or target_pomos * 25),
        "source": "user",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "schema_version": 1
    }
    db.daily_targets.update_one({"_id": _id}, {"$setOnInsert": doc, "$set": {
        "target_pomos": doc["target_pomos"],
        "target_minutes": doc["target_minutes"],
        "updated_at": doc["updated_at"]
    }}, upsert=True)

def get_daily_target(user_id: str, date_ist: str) -> Optional[Dict[str, Any]]:
    return db.daily_targets.find_one({"user": user_id, "date_ist": date_ist})

def sum_pe_for(user_id: str, week_key: str, goal_id: str, bucket: str) -> float:
    pipeline = [
        {"$match": {"user": user_id, "week_key": week_key, "t": "W", "goal_id": goal_id, "alloc_bucket": bucket}},
        {"$group": {"_id": None, "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
    ]
    doc = next(iter(db.sessions.aggregate(pipeline)), None)
    return float(doc["pe"]) if doc else 0.0

def total_day_pe(user_id: str, date_ist: str) -> float:
    pipeline = [
        {"$match": {"user": user_id, "date_ist": date_ist, "t": "W"}},
        {"$group": {"_id": None, "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
    ]
    doc = next(iter(db.sessions.aggregate(pipeline)), None)
    return float(doc["pe"]) if doc else 0.0

def determine_alloc_bucket(user_id: str, week_key: str, goal_id: str, planned_current: int) -> str:
    """
    Use pom-equivalents to decide whether we're still in 'current' or must add to 'backlog'.
    """
    done_current_pe = sum_pe_for(user_id, week_key, goal_id, "current")
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
    pe = round(dur_min / 25.0, 2)
    started_at_ist = ended_at_ist - timedelta(minutes=dur_min)

    date_ist = started_at_ist.astimezone(IST).date().isoformat()
    week_key = week_key_from_datestr(date_ist)

    # Deterministic ID to avoid double inserts
    sid = f"{user_id}|{date_ist}|{t}|{int(started_at_ist.timestamp())}|{dur_min}"

    doc = {
        "_id": sid,
        "user": user_id,
        "date_ist": date_ist,
        "week_key": week_key,
        "t": t,
        "kind": kind,
        "activity_type": activity_type,
        "intensity": intensity,
        "dur_min": int(dur_min),
        "pom_equiv": pe,
        "started_at_ist": utc_from_ist(started_at_ist),
        "ended_at_ist": utc_from_ist(ended_at_ist),
        "deep_work": deep_work,
        "context_switch": False,
        "goal_mode": goal_mode,
        "goal_id": goal_id,
        "task": task,
        "cat": cat,
        "alloc_bucket": alloc_bucket,
        "break_autostart": break_autostart,
        "skipped": skipped,
        "post_checkin": post_checkin,
        "device": device,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "schema_version": 1
    }
    db.sessions.update_one({"_id": sid}, {"$setOnInsert": doc, "$set": {"updated_at": doc["updated_at"]}}, upsert=True)
    return sid

# --- UI: Sidebar ---
st.sidebar.header("⚙️ Connection")
st.sidebar.write(f"**DB:** `{DB_NAME}`")
st.sidebar.write(f"**User:** `{USER_ID}`")
if not MONGO_URI:
    st.sidebar.error("MONGO_URI is not set in env.")
    st.stop()

st.sidebar.divider()

# Week plan quick view
today_ist = now_ist()
week_start, week_end, week_key = week_start_end_ist(today_ist)
goals_map = get_goals_map(USER_ID)
plan = get_week_plan(USER_ID, week_key)

st.sidebar.subheader(f"📅 Week {week_key}")
if plan:
    st.sidebar.caption(f"{week_start} → {week_end}")
    cap = plan.get("capacity", {})
    st.sidebar.write(f"Capacity: **{cap.get('total', 0)}** poms  (wkday {cap.get('weekday',0)}, wkend {cap.get('weekend',0)})")
    # list top 5 by rank
    items = sorted(plan.get("items", []), key=lambda x: x.get("priority_rank", 99))[:5]
    for it in items:
        gid = it["goal_id"]
        g = goals_map.get(gid, {})
        st.sidebar.write(f"- {g.get('title','?')} — **{it['planned_current']}** + backlog {it['backlog_in']}")
else:
    st.sidebar.info("No weekly plan for this week yet.")

# --- UI: Main layout ---
left, right = st.columns([1.2, 1])

# == LEFT: Timer & Logging ==
with left:
    st.title("⏱️ Focus Timer")
    st.caption(f"IST Date: **{today_ist.date().isoformat()}** • Week: **{week_key}**")

    # Daily target
    st.subheader("🎯 Today’s Target")
    todays_target = get_daily_target(USER_ID, today_ist.date().isoformat())
    target_val = (todays_target or {}).get("target_pomos", None)
    colT1, colT2 = st.columns([2,1])
    with colT1:
        st.metric("Target (poms)", value=target_val if target_val is not None else "—")
        actual_pe = total_day_pe(USER_ID, today_ist.date().isoformat())
        st.progress(min(actual_pe / float(target_val or 1), 1.0), text=f"Progress: {actual_pe:.1f} / {target_val or 0} pe")
    with colT2:
        new_target = st.number_input("Set/Update target", min_value=0, max_value=50, value=int(target_val or 6), step=1)
        if st.button("Save target", use_container_width=True):
            upsert_daily_target(USER_ID, today_ist.date().isoformat(), int(new_target))
            st.success("Saved target.")

    st.divider()

    st.subheader("🎛️ Log a Session")
    sess_type = st.segmented_control("Session Type", options=["Work (focus)", "Work (activity)", "Break"], default="Work (focus)")

    dur_default = 25 if "Work" in sess_type else 5
    dur_min = st.number_input("Duration (minutes)", min_value=1, max_value=180, value=dur_default, step=1)

    ended_now = st.checkbox("End at now (IST)", value=True)
    if ended_now:
        end_dt_ist = now_ist()
    else:
        end_time_str = st.time_input("End time (IST)", value=now_ist().time())
        end_dt_ist = IST.localize(datetime.combine(today_ist.date(), end_time_str))

    post_quality = st.slider("Quality (1–5)", 1, 5, 4)
    post_mood    = st.slider("Mood (1–5)", 1, 5, 4)
    post_energy  = st.slider("Energy (1–5)", 1, 5, 4)
    post_note    = st.text_input("Note (optional)")

    if "Work" in sess_type:
        # goal selection
        mode = st.radio("Work mode", options=["Weekly plan goal", "Custom / unplanned"], horizontal=True)
        goal_id = None
        goal_mode = "custom"
        cat = None
        activity_type = None
        intensity = None
        kind = "focus"

        if sess_type == "Work (activity)":
            kind = "activity"
            activity_type = st.selectbox("Activity type", ["exercise", "meditation", "breathing", "other"], index=1)
            intensity = st.selectbox("Intensity", ["light", "moderate", "vigorous"], index=0)

        alloc_bucket = None
        deep_work = dur_min >= 23 if kind != "activity" else None

        if mode == "Weekly plan goal" and plan:
            items = sorted(plan.get("items", []), key=lambda x: x.get("priority_rank", 99))
            options = []
            for it in items:
                gid = it["goal_id"]
                gtitle = goals_map.get(gid, {}).get("title", gid)
                options.append((f"{gtitle} — planned {it['planned_current']} (backlog {it['backlog_in']})", gid, it['planned_current']))
            label = st.selectbox("Choose goal", options=[o[0] for o in options])
            sel = next((o for o in options if o[0] == label), None)
            if sel:
                goal_id = sel[1]
                goal_mode = "weekly"
                # decide alloc bucket by pom-equivalents done so far
                alloc_bucket = determine_alloc_bucket(USER_ID, week_key, goal_id, sel[2])
                cat = goals_map.get(goal_id, {}).get("category")

        else:
            goal_mode = "custom"
            task_text = st.text_input("What did you work on? (short note)")
            cat = st.selectbox("Category (for analytics)", ["Learning","Projects","Certification","Career","Health","Wellbeing","Other"], index=0)
            if st.button("Log Work", use_container_width=True):
                sid = insert_session(
                    USER_ID, "W", dur_min, end_dt_ist,
                    kind=kind, activity_type=activity_type, intensity=intensity,
                    deep_work=deep_work,
                    goal_mode=goal_mode, goal_id=None, task=task_text, cat=cat,
                    alloc_bucket=None, break_autostart=True, skipped=None,
                    post_checkin={
                        "quality_1to5": post_quality,
                        "mood_1to5": post_mood,
                        "energy_1to5": post_energy,
                        "distraction": None,
                        "note": post_note or None
                    },
                    device="web"
                )
                st.success(f"Logged work (custom). id={sid}")

        if mode == "Weekly plan goal" and plan and st.button("Log Work", use_container_width=True):
            sid = insert_session(
                USER_ID, "W", dur_min, end_dt_ist,
                kind=kind, activity_type=activity_type, intensity=intensity,
                deep_work=deep_work,
                goal_mode="weekly", goal_id=goal_id, task=None, cat=cat,
                alloc_bucket=alloc_bucket, break_autostart=True, skipped=None,
                post_checkin={
                    "quality_1to5": post_quality,
                    "mood_1to5": post_mood,
                    "energy_1to5": post_energy,
                    "distraction": None,
                    "note": post_note or None
                },
                device="web"
            )
            st.success(f"Logged work on goal `{goals_map.get(goal_id,{}).get('title',goal_id)}` (bucket={alloc_bucket}). id={sid}")

    else:
        # Break
        skipped = st.checkbox("Skipped?", value=False)
        if st.button("Log Break", use_container_width=True):
            sid = insert_session(
                USER_ID, "B", dur_min, end_dt_ist,
                kind=None, activity_type=None, intensity=None,
                deep_work=None, goal_mode=None, goal_id=None, task=None, cat=None,
                alloc_bucket=None, break_autostart=None, skipped=skipped,
                post_checkin=None, device="web"
            )
            st.success(f"Logged break. id={sid}")

# == RIGHT: This week snapshot ==
with right:
    st.title("📊 This Week")
    if plan:
        items = sorted(plan.get("items", []), key=lambda x: x.get("priority_rank", 99))
        rows = []
        for it in items:
            gid = it["goal_id"]
            title = goals_map.get(gid, {}).get("title", gid)
            planned = it["planned_current"]
            cur_pe = sum_pe_for(USER_ID, week_key, gid, "current")
            back_pe = sum_pe_for(USER_ID, week_key, gid, "backlog")
            rows.append((it["priority_rank"], title, planned, it["backlog_in"], planned + it["backlog_in"], round(cur_pe,1), round(back_pe,1)))
        st.dataframe(
            { "Rank": [r[0] for r in rows],
              "Goal": [r[1] for r in rows],
              "Planned": [r[2] for r in rows],
              "Backlog In": [r[3] for r in rows],
              "Total Target": [r[4] for r in rows],
              "Done Current (pe)": [r[5] for r in rows],
              "Done Backlog (pe)": [r[6] for r in rows],
            },
            hide_index=True, use_container_width=True
        )
        planned_total = sum(r[2] for r in rows)
        done_total = sum(r[5] + r[6] for r in rows)
        st.progress(min(done_total / max(planned_total, 1), 1.0), text=f"Adherence: {done_total:.1f} / {planned_total} pe")
    else:
        st.info("No plan for this week yet.")

    st.divider()
    st.caption("Tip: backlog is only used after the current allocation is fully consumed (by pom-equivalents).")

# Footer
st.caption("Focus Timer v1 • Mongo-backed • IST-aware")
