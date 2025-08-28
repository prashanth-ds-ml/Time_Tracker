import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

import certifi
import pytz
import streamlit as st
from pymongo import MongoClient

# ─── Config / constants ────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

def today_iso() -> str:
    return datetime.now(IST).date().isoformat()

def now_ist():
    return datetime.now(IST)

def utc_from_ist(dt_ist: datetime) -> datetime:
    return dt_ist.astimezone(timezone.utc)

def week_key_from_datestr(datestr: str) -> str:
    y, m, d = map(int, datestr.split("-"))
    dt = datetime(y, m, d)
    iso = dt.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def week_start_end_ist(dt_ist: datetime):
    monday = (dt_ist - timedelta(days=dt_ist.isoweekday() - 1)).date()
    sunday = monday + timedelta(days=6)
    wk = f"{dt_ist.isocalendar().year}-{dt_ist.isocalendar().week:02d}"
    return monday.isoformat(), sunday.isoformat(), wk

def pom_equiv(minutes: int) -> float:
    return round(float(minutes) / 25.0, 2)

# ─── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="Focus Timer", page_icon="⏱️", layout="wide")

# ─── Mongo connection (cached, robust) ─────────────────────────────────────────
@st.cache_resource
def get_db():
    uri = (st.secrets.get("MONGO_URI") or os.getenv("MONGO_URI") or os.getenv("mongo_uri") or "").strip()
    dbname = (st.secrets.get("DB_NAME") or os.getenv("DB_NAME") or "Focus_DB").strip()
    if not uri:
        st.error("MONGO_URI is not configured (set in .streamlit/secrets.toml or env).")
        st.stop()
    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=7000,
            tlsCAFile=certifi.where()
        )
        client.admin.command("ping")
        return client[dbname]
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        st.stop()

db = get_db()
USER_ID = (st.secrets.get("USER_ID") or os.getenv("USER_ID") or "prashanth").strip()

# ─── Data access helpers ───────────────────────────────────────────────────────
def get_user(uid: str) -> Optional[Dict[str, Any]]:
    return db.users.find_one({"_id": uid})

def get_goals_map(uid: str) -> Dict[str, Dict[str, Any]]:
    return {g["_id"]: g for g in db.goals.find({"user": uid})}

def get_week_plan(uid: str, week_key: str) -> Optional[Dict[str, Any]]:
    return db.weekly_plans.find_one({"user": uid, "week_key": week_key})

def upsert_daily_target(uid: str, date_ist: str, target_pomos: int, target_minutes: Optional[int] = None):
    _id = f"{uid}|{date_ist}"
    now = datetime.now(timezone.utc)
    db.daily_targets.update_one(
        {"_id": _id},
        {"$setOnInsert": {
            "_id": _id, "user": uid, "date_ist": date_ist,
            "schema_version": 1, "created_at": now
        },
         "$set": {
            "target_pomos": int(target_pomos),
            "target_minutes": int(target_minutes or target_pomos * 25),
            "source": "user",
            "updated_at": now
        }},
        upsert=True
    )

def get_daily_target(uid: str, date_ist: str) -> Optional[Dict[str, Any]]:
    return db.daily_targets.find_one({"user": uid, "date_ist": date_ist})

def sum_pe_for(uid: str, week_key: str, goal_id: str, bucket: str) -> float:
    pipeline = [
        {"$match": {"user": uid, "week_key": week_key, "t": "W", "goal_id": goal_id, "alloc_bucket": bucket}},
        {"$group": {"_id": None, "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
    ]
    doc = next(iter(db.sessions.aggregate(pipeline)), None)
    return float(doc["pe"]) if doc else 0.0

def total_day_pe(uid: str, date_ist: str) -> float:
    pipeline = [
        {"$match": {"user": uid, "date_ist": date_ist, "t": "W"}},
        {"$group": {"_id": None, "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
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

    # Deterministic ID to avoid duplicate inserts on rerun
    sid = f"{user_id}|{date_ist}|{t}|{int(started_at_ist.timestamp())}|{dur_min}"
    now = datetime.now(timezone.utc)

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
        "created_at": now,
        "updated_at": now,
        "schema_version": 1
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

# ─── Sidebar: connection + week summary ────────────────────────────────────────
st.sidebar.header("⚙️ Connection")
db_name = db.name
st.sidebar.write(f"**DB:** `{db_name}`")
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

# ─── Main: Tabs ────────────────────────────────────────────────────────────────
tab_timer, tab_week, tab_diag = st.tabs(["⏱️ Timer & Log", "📊 This Week", "🛠️ Diagnostics"])

# == Tab: Timer & Log ===========================================================
with tab_timer:
    st.header("⏱️ Focus Timer")
    st.caption(f"IST Date: **{today}** • Week: **{week_key}**")

    # ── Today Target
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
        # Quick stats today
        todays = list_today_sessions(USER_ID, today)
        focus_cnt = sum(1 for s in todays if s.get("t") == "W" and s.get("kind") != "activity")
        breaks_valid = sum(1 for s in todays if s.get("t") == "B" and (s.get("dur_min",0) >= 4) and not s.get("skipped", False))
        st.metric("Focus sessions", focus_cnt)
        st.metric("Valid breaks", breaks_valid)

    st.divider()

    # ── Log a Session (forms to avoid double insert on rerun)
    st.subheader("🎛️ Log a Session")

    # segmented_control (fallback to radio for older Streamlit)
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

            # shared defaults
            kind = "focus"
            activity_type = None
            intensity = None
            deep_work = dur_min >= 23

            if sess_type == "Work (activity)":
                kind = "activity"
                activity_type = st.selectbox("Activity type", ["exercise", "meditation", "breathing", "other"], index=1)
                intensity = st.selectbox("Intensity", ["light", "moderate", "vigorous"], index=0)
                deep_work = None  # not applicable

            mode = st.radio("Work mode", options=["Weekly plan goal", "Custom / unplanned"], horizontal=True)
            goal_id = None
            goal_mode = "custom"
            cat = None
            alloc_bucket = None

            if mode == "Weekly plan goal" and plan:
                items = sorted(plan.get("items", []), key=lambda x: x.get("priority_rank", 99))
                options = []
                for it in items:
                    gid = it["goal_id"]
                    gtitle = get_goals_map(USER_ID).get(gid, {}).get("title", gid)
                    options.append((f"{gtitle} — planned {it['planned_current']} (backlog {it['backlog_in']})", gid, it["planned_current"]))
                label = st.selectbox("Choose goal", options=[o[0] for o in options])
                sel = next((o for o in options if o[0] == label), None)
                if sel:
                    goal_id = sel[1]
                    goal_mode = "weekly"
                    alloc_bucket = determine_alloc_bucket(USER_ID, week_key, goal_id, sel[2])
                    cat = get_goals_map(USER_ID).get(goal_id, {}).get("category")
            else:
                goal_mode = "custom"
                task_text = st.text_input("What did you work on? (short note)")
                cat = st.selectbox("Category (for analytics)", ["Learning","Projects","Certification","Career","Health","Wellbeing","Other"], index=0)

            submit_work = st.form_submit_button("Log Work", use_container_width=True)
            if submit_work:
                sid = insert_session(
                    USER_ID, "W", int(dur_min), end_dt_ist,
                    kind=kind, activity_type=activity_type, intensity=intensity,
                    deep_work=deep_work,
                    goal_mode=goal_mode, goal_id=(goal_id if goal_mode=="weekly" else None),
                    task=(None if goal_mode=="weekly" else task_text),
                    cat=cat,
                    alloc_bucket=(alloc_bucket if goal_mode=="weekly" else None),
                    break_autostart=True, skipped=None,
                    post_checkin={
                        "quality_1to5": int(post_quality),
                        "mood_1to5": int(post_mood),
                        "energy_1to5": int(post_energy),
                        "distraction": None,
                        "note": (post_note or None)
                    },
                    device="web"
                )
                st.success(f"Logged work. id={sid}")
                st.experimental_rerun()
    else:
        # Break form
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

    # ── Today’s sessions + undo last
    st.subheader("📝 Today’s Sessions")
    todays = list_today_sessions(USER_ID, today)
    if not todays:
        st.info("No sessions logged yet.")
    else:
        def fmt_row(s):
            kind = s.get("t")
            label = "Work" if kind == "W" else "Break"
            if s.get("kind") == "activity":
                label = "Activity"
            goal_title = goals_map.get(s.get("goal_id"), {}).get("title") if s.get("goal_id") else (s.get("task") or "—")
            return {
                "When (IST)": s.get("started_at_ist").astimezone(IST).strftime("%H:%M"),
                "Type": label,
                "Dur (min)": s.get("dur_min"),
                "PE": s.get("pom_equiv"),
                "Goal/Task": goal_title,
                "Bucket": s.get("alloc_bucket") or "—",
                "Deep": "✓" if s.get("deep_work") else "—",
            }
        st.dataframe([fmt_row(s) for s in todays], use_container_width=True, hide_index=True)
        if st.button("↩️ Undo last entry", use_container_width=True):
            deleted = delete_last_today_session(USER_ID, today)
            if deleted:
                st.warning(f"Deleted last session: {deleted}")
            else:
                st.info("Nothing to undo.")
            st.experimental_rerun()

# == Tab: This Week =============================================================
with tab_week:
    st.header("📊 This Week")
    if not plan:
        st.info("No plan for this week yet.")
    else:
        items = sorted(plan.get("items", []), key=lambda x: x.get("priority_rank", 99))
        rows = []
        for it in items:
            gid = it["goal_id"]
            title = goals_map.get(gid, {}).get("title", gid)
            planned = int(it["planned_current"])
            back_in = int(it["backlog_in"])
            cur_pe = sum_pe_for(USER_ID, week_key, gid, "current")
            back_pe = sum_pe_for(USER_ID, week_key, gid, "backlog")
            rows.append((it["priority_rank"], title, planned, back_in, planned + back_in, round(cur_pe,1), round(back_pe,1)))
        st.dataframe(
            {
                "Rank": [r[0] for r in rows],
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

        st.caption("Backlog is only used after current allocation is fully consumed (by pom-equivalents).")

# == Tab: Diagnostics ============================================================
with tab_diag:
    st.header("🛠️ Diagnostics & Info")
    st.write("User doc:", get_user(USER_ID))
    st.write("Goals:", list(goals_map.values())[:5])
    st.write("Plan:", plan)
