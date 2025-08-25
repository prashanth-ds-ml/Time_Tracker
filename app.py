# app.py — Focus Timer • Weekly Priorities (single-file version)
import streamlit as st
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import math
import hashlib
import pandas as pd
from pymongo import MongoClient
import pytz

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Focus Timer • Weekly Priorities",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'About': "Focus Timer - Dynamic Weekly Priority & Pomodoro Management"}
)

POMODORO_MIN = 25
BREAK_MIN = 5
IST = pytz.timezone("Asia/Kolkata")
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"

# =========================
# TIME / MATH HELPERS
# =========================
def now_ist() -> datetime:
    return datetime.now(IST)

def week_bounds(d: date) -> Tuple[date, date]:
    start = d - timedelta(days=d.weekday())
    return start, start + timedelta(days=6)

def safe_div(n, d, default=0.0):
    try:
        if d is None or d == 0:
            return default
        return float(n) / float(d)
    except Exception:
        return default

def pct_or_dash(n, d, decimals=0):
    if d is None or d <= 0:
        return "—"
    pct = 100.0 * safe_div(n, d, default=0.0)
    return f"{pct:.{decimals}f}%"

def gini_from_counts(counts):
    arr = [c for c in counts if c is not None and c >= 0]
    if not arr:
        return 0.0
    arr = sorted(arr)
    n = len(arr); s = sum(arr)
    if s == 0: return 0.0
    cum = 0.0
    for i, x in enumerate(arr, start=1):
        cum += i * x
    return (2.0 * cum) / (n * s) - (n + 1.0) / n

def entropy_norm_from_counts(counts):
    arr = [c for c in counts if c is not None and c > 0]
    k = len(arr)
    if k <= 1: return 0.0
    s = float(sum(arr))
    H = -sum((c/s) * math.log((c/s), 2) for c in arr)
    return H / math.log(k, 2)

def time_to_minutes(tstr):
    try:
        dt = datetime.strptime(tstr, "%I:%M %p")
        return dt.hour*60 + dt.minute
    except Exception:
        return None

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

# =========================
# MONGO INIT
# =========================
@st.cache_resource
def get_db():
    try:
        client = MongoClient(st.secrets["mongo_uri"])
        return client["time_tracker_db"]
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

def ensure_indexes():
    db = get_db()
    # New schema
    db.user_days.create_index([("user",1),("date",1)], name="user_date", unique=True)
    db.weekly_plans.create_index([("user",1),("type",1)], name="user_type")
    db.weekly_plans.create_index([("user",1),("week_start",1)], name="user_weekstart")
    # Legacy (read-only) quick indexes (if present)
    try:
        db.logs.create_index([("user",1),("type",1),("date",1)], name="legacy_user_type_date")
        db.goals.create_index([("user",1),("title",1)], name="legacy_user_title")
    except Exception:
        pass

# =========================
# DATA ACCESS (NEW SCHEMA)
# =========================
def get_or_create_registry(user: str) -> Dict:
    db = get_db()
    rid = f"{user}|registry"
    reg = db.weekly_plans.find_one({"_id": rid})
    if reg:
        return reg
    doc = {
        "_id": rid,
        "type": "registry",
        "user": user,
        "goals": [],  # [{goal_id,title,goal_type,status,priority_band,target_poms,created_at,updated_at}]
        "defaults": {
            "weekday_poms": 3,
            "weekend_poms": 5,
            "auto_break": True,
            "custom_categories": ["Learning", "Projects", "Research", "Planning"],
        },
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    db.weekly_plans.insert_one(doc)
    return doc

def registry_defaults(user: str) -> Dict:
    return get_or_create_registry(user).get("defaults", {})

def update_registry_defaults(user: str, weekday_poms: int, weekend_poms: int,
                             auto_break: Optional[bool] = None,
                             custom_categories: Optional[List[str]] = None):
    db = get_db()
    reg = get_or_create_registry(user)
    defs = reg.get("defaults", {})
    defs["weekday_poms"] = int(weekday_poms)
    defs["weekend_poms"] = int(weekend_poms)
    if auto_break is not None:
        defs["auto_break"] = bool(auto_break)
    if custom_categories is not None:
        defs["custom_categories"] = list(custom_categories)
    db.weekly_plans.update_one({"_id": reg["_id"]},
        {"$set": {"defaults": defs, "updated_at": datetime.utcnow()}}
    )

def list_registry_goals(user: str, statuses: Optional[List[str]] = None) -> List[Dict]:
    reg = get_or_create_registry(user)
    goals = reg.get("goals", [])
    if statuses:
        goals = [g for g in goals if g.get("status") in statuses]
    return goals

def goal_title_map(user: str) -> Dict[str,str]:
    return {g["goal_id"]: g["title"] for g in get_or_create_registry(user).get("goals", [])}

def upsert_registry_goal(user: str, title: str, goal_type: str = "Other",
                         status: str = "In Progress", priority_band: int = 2,
                         target_poms: int = 0, goal_id: Optional[str] = None) -> str:
    db = get_db()
    reg = get_or_create_registry(user)
    if not goal_id:
        goal_id = hashlib.sha256(f"{user}|{title}".encode()).hexdigest()[:16]
    goals = reg.get("goals", [])
    found = False
    for g in goals:
        if g["goal_id"] == goal_id or g["title"].strip().lower() == title.strip().lower():
            g["title"] = title
            g["goal_type"] = goal_type
            g["status"] = status
            g["priority_band"] = max(1, min(3, int(priority_band)))
            g["target_poms"] = int(target_poms)
            g["updated_at"] = datetime.utcnow()
            found = True
            break
    if not found:
        goals.append({
            "goal_id": goal_id,
            "title": title,
            "goal_type": goal_type,
            "status": status,
            "priority_band": max(1, min(3, int(priority_band))),
            "target_poms": int(target_poms),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        })
    db.weekly_plans.update_one({"_id": reg["_id"]},
        {"$set": {"goals": goals, "updated_at": datetime.utcnow()}}
    )
    return goal_id

def update_registry_goal_status(user: str, goal_id: str, new_status: str):
    db = get_db()
    reg = get_or_create_registry(user)
    goals = reg.get("goals", [])
    for g in goals:
        if g["goal_id"] == goal_id:
            g["status"] = new_status
            g["updated_at"] = datetime.utcnow()
            break
    db.weekly_plans.update_one({"_id": reg["_id"]},
        {"$set": {"goals": goals, "updated_at": datetime.utcnow()}}
    )

def _capacity_from_defaults(ws: date, defs: Dict) -> int:
    weekday_poms = int(defs.get("weekday_poms", 3))
    weekend_poms = int(defs.get("weekend_poms", 5))
    wd = sum(1 for i in range(7) if (ws + timedelta(days=i)).weekday() < 5)
    we = 7 - wd
    return weekday_poms*wd + weekend_poms*we

def get_or_create_week_plan(user: str, d: Optional[date] = None) -> Dict:
    db = get_db()
    if d is None:
        d = now_ist().date()
    ws, we = week_bounds(d)
    pid = f"{user}|{ws.isoformat()}"
    plan = db.weekly_plans.find_one({"_id": pid})
    if plan:
        # adapt legacy plan (no type) if needed
        if plan.get("type") != "plan":
            plan.setdefault("type","plan")
            plan.setdefault("capacity", plan.get("total_poms", _capacity_from_defaults(ws, registry_defaults(user))))
            if "allocations_by_goal" not in plan and "allocations" in plan:
                plan["allocations_by_goal"] = {k:int(v) for k,v in plan.get("allocations", {}).items()}
            plan.setdefault("goals_embedded", [])
            db.weekly_plans.update_one({"_id": pid}, {"$set": plan})
        return plan
    defs = registry_defaults(user)
    doc = {
        "_id": pid,
        "type": "plan",
        "user": user,
        "week_start": ws.isoformat(),
        "week_end": we.isoformat(),
        "capacity": _capacity_from_defaults(ws, defs),
        "allocations_by_goal": {},      # {goal_id: planned}
        "goals_embedded": [],           # snapshot
        "stats": {},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    db.weekly_plans.insert_one(doc)
    return doc

def save_week_plan(user: str, week_start: date, allocations_by_goal: Dict[str, int]):
    db = get_db()
    ws, _ = week_bounds(week_start)
    pid = f"{user}|{ws.isoformat()}"
    plan = get_or_create_week_plan(user, ws)
    titles = goal_title_map(user)
    reg_goals = {g["goal_id"]: g for g in list_registry_goals(user)}
    embedded = []
    for gid, planned in allocations_by_goal.items():
        rg = reg_goals.get(gid, {})
        embedded.append({
            "goal_id": gid,
            "title": titles.get(gid, "(missing)"),
            "priority_band": int(rg.get("priority_band", 2)),
            "status": rg.get("status", "In Progress"),
            "goal_type": rg.get("goal_type", "Other"),
            "planned": int(planned),
        })
    db.weekly_plans.update_one({"_id": pid}, {"$set": {
        "allocations_by_goal": {k:int(v) for k,v in allocations_by_goal.items()},
        "goals_embedded": embedded,
        "updated_at": datetime.utcnow()
    }})

# user_days
def get_or_create_user_day(user: str, iso_date: Optional[str] = None) -> Dict:
    db = get_db()
    if not iso_date:
        iso_date = now_ist().date().isoformat()
    _id = f"{user}|{iso_date}"
    doc = db.user_days.find_one({"_id": _id})
    if doc:
        return doc
    new_doc = {
        "_id": _id,
        "user": user,
        "date": iso_date,
        "sessions": [],     # [{t:'Work'/'Break', minutes:int, time:'HH:MM AM', goal_id?, task?, category?}]
        "totals": {"work_count":0,"break_count":0,"work_minutes":0,"break_minutes":0},
        "daily_target": None,
        "reflection": None,
        "notes": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    db.user_days.insert_one(new_doc)
    return new_doc

def append_session(user: str, is_break: bool, minutes: int,
                   time_str: str, goal_id: Optional[str],
                   task: str, category: str, iso_date: Optional[str] = None):
    db = get_db()
    day = get_or_create_user_day(user, iso_date)
    t = "Break" if is_break else "Work"
    sess = {
        "t": t,
        "minutes": int(minutes),
        "time": time_str,
        **({"goal_id": goal_id} if (not is_break and goal_id) else {}),
        **({"task": task} if (not is_break and task) else {}),
        **({"category": category} if (not is_break and category) else {}),
    }
    totals = day.get("totals", {"work_count":0,"break_count":0,"work_minutes":0,"break_minutes":0})
    if is_break:
        totals["break_count"] = totals.get("break_count",0) + 1
        totals["break_minutes"] = totals.get("break_minutes",0) + int(minutes)
    else:
        totals["work_count"] = totals.get("work_count",0) + 1
        totals["work_minutes"] = totals.get("work_minutes",0) + int(minutes)
    db.user_days.update_one({"_id": day["_id"]}, {"$set": {"updated_at": datetime.utcnow(), "totals": totals},
                                                  "$push": {"sessions": sess}})

def get_sessions_df(user: str) -> pd.DataFrame:
    db = get_db()
    cursor = db.user_days.find({"user": user}, {"_id":1,"date":1,"sessions":1,"user":1})
    recs = []
    for d in cursor:
        iso = d.get("date")
        u = d.get("user", user)
        for s in d.get("sessions", []):
            recs.append({
                "date": iso,
                "time": s.get("time"),
                "pomodoro_type": "Break" if s.get("t")=="Break" else "Work",
                "duration": int(s.get("minutes", 0)),
                "user": u,
                "goal_id": s.get("goal_id"),
                "task": s.get("task",""),
                "category": s.get("category",""),
            })
    if not recs:
        df = pd.DataFrame(columns=["date","time","pomodoro_type","duration","user","goal_id","task","category"])
        df["date"] = pd.to_datetime(df["date"])
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
        return df
    df = pd.DataFrame.from_records(recs)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
    for c in ["time","pomodoro_type","user","goal_id","task","category"]:
        if c not in df.columns:
            df[c] = "" if c!="goal_id" else None
    return df

def set_daily_target(user: str, target: int, iso_date: Optional[str] = None):
    db = get_db()
    day = get_or_create_user_day(user, iso_date)
    db.user_days.update_one({"_id": day["_id"]},
        {"$set": {"daily_target": int(target), "updated_at": datetime.utcnow()}}
    )

def get_daily_target(user: str, iso_date: Optional[str] = None) -> Optional[int]:
    day = get_or_create_user_day(user, iso_date)
    return day.get("daily_target")

def save_reflection(user: str, aligned: str, rating: int, blockers: str, notes: str,
                    iso_date: Optional[str] = None):
    db = get_db()
    day = get_or_create_user_day(user, iso_date)
    db.user_days.update_one({"_id": day["_id"]},
        {"$set": {"reflection": {"aligned": aligned, "focus_rating": int(rating),
                                 "blockers": blockers, "notes": notes},
                  "updated_at": datetime.utcnow()}}
    )

def add_note(user: str, content: str, iso_date: Optional[str] = None):
    db = get_db()
    day = get_or_create_user_day(user, iso_date)
    db.user_days.update_one({"_id": day["_id"]},
        {"$push": {"notes": {"content": content, "ts": datetime.utcnow()}},
         "$set": {"updated_at": datetime.utcnow()}}
    )

def list_users() -> List[str]:
    db = get_db()
    users = set()
    for u in db.weekly_plans.find({"type":"registry"}, {"user":1}):
        users.add(u["user"])
    for u in db.user_days.distinct("user"):
        users.add(u)
    # legacy fallback (first run)
    for udoc in db.users.find({}, {"username":1}):
        if udoc.get("username"):
            users.add(udoc["username"])
    return sorted([u for u in users if u])

def add_user(username: str):
    get_or_create_registry(username)

# ---------- Legacy Backfills (optional, safe no-ops if legacy absent) ----------
def legacy_backfill_registry_from_goals(user: str):
    """If registry has no goals, pull legacy 'goals' collection once."""
    db = get_db()
    reg = get_or_create_registry(user)
    if reg.get("goals"):  # already has goals
        return
    legacy = list(db.goals.find({"user": user}))
    if not legacy:
        return
    for g in legacy:
        title = g.get("title","(goal)")
        prio = int(g.get("priority_weight", 2))
        status = g.get("status","In Progress")
        gtype = g.get("goal_type","Other")
        gid = g.get("_id")
        if isinstance(gid, dict):  # ObjectId or others
            gid = hashlib.sha256(f"{user}|{title}".encode()).hexdigest()[:16]
        upsert_registry_goal(user, title, goal_type=gtype, status=status,
                             priority_band=max(1, min(3, prio if prio<=3 else 3)),
                             target_poms=int(g.get("target_poms",0)),
                             goal_id=str(gid))

def legacy_backfill_user_days_from_logs(user: str):
    """If user_days is empty but legacy logs exist, import once."""
    db = get_db()
    if db.user_days.find_one({"user": user}):
        return
    logs = list(db.logs.find({"type":"Pomodoro","user":user}))
    if not logs:
        return
    # group by date
    by_date: Dict[str, List[Dict]] = {}
    for r in logs:
        iso = r.get("date")
        if not iso:
            continue
        dlist = by_date.setdefault(iso, [])
        dlist.append(r)
    for iso, items in by_date.items():
        get_or_create_user_day(user, iso)
        for r in items:
            append_session(
                user=user,
                is_break=(r.get("pomodoro_type")=="Break"),
                minutes=int(r.get("duration", 0)),
                time_str=r.get("time", "09:00 AM"),
                goal_id=r.get("goal_id"),
                task=r.get("task",""),
                category=r.get("category",""),
                iso_date=iso
            )

# =========================
# SESSION STATE
# =========================
def init_state():
    defaults = {
        "user": None,
        "page": "🎯 Focus Timer",
        "planning_week_date": now_ist().date(),
        "start_time": None,
        "is_break": False,
        "task": "",
        "active_goal_id": None,
        "active_goal_title": "",
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_runtime_state_for_user():
    st.session_state.start_time = None
    st.session_state.is_break = False
    st.session_state.task = ""
    st.session_state.active_goal_id = None
    st.session_state.active_goal_title = ""

# =========================
# UI: COMMON
# =========================
def sidebar_admin(user: str):
    st.sidebar.markdown("### ⚙️ Admin")
    if st.sidebar.button("Ensure Mongo Indexes"):
        ensure_indexes()
        st.sidebar.success("Indexes ensured.")
    # Export
    df = get_sessions_df(user)
    if df.empty:
        st.sidebar.info("No sessions to export.")
    else:
        out = df.sort_values("date")
        st.sidebar.download_button(
            "⬇️ Export Sessions (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            file_name=f"{user}_sessions.csv",
            mime="text/csv"
        )

def proportional_allocation(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    if total <= 0 or not weights:
        return {gid: 0 for gid in weights}
    total_w = sum(max(1, int(w)) for w in weights.values())
    raw = {gid: (max(1, int(w))/total_w)*total for gid, w in weights.items()}
    base = {gid: int(v) for gid, v in raw.items()}
    diff = total - sum(base.values())
    if diff != 0:
        fracs = sorted(((gid, raw[gid]-int(raw[gid])) for gid in raw), key=lambda x: x[1], reverse=True)
        i = 0
        while diff != 0 and fracs:
            g = fracs[i % len(fracs)][0]
            base[g] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            i += 1
    return base

# =========================
# PAGE: FOCUS TIMER
# =========================
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
        # Save current session & beep BEFORE auto break starts
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
        st.balloons(); st.success("🎉 Session complete!")
        # Reset
        st.session_state.start_time = None
        st.session_state.is_break = False
        st.session_state.task = ""
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = ""
        # Autostart break after WORK
        if (not was_break) and auto_break:
            st.toast("☕ Auto-starting a 5-minute break")
            st.session_state.start_time = time.time()
            st.session_state.is_break = True
            st.rerun()
        return True

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

def render_focus_timer(user: str):
    st.header("🎯 Focus Timer")
    defs = registry_defaults(user)
    col, _ = st.columns([1,3])
    with col:
        auto_break_ui = st.toggle("Auto-start break", value=bool(defs.get("auto_break", True)))
        if auto_break_ui != bool(defs.get("auto_break", True)):
            update_registry_defaults(user, defs.get("weekday_poms",3), defs.get("weekend_poms",5), auto_break=auto_break_ui)

    # Active timer
    if render_timer_widget(user, auto_break=bool(registry_defaults(user).get("auto_break", True))):
        return

    # Quick Today
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
    # goal titles only (no "plan:9" clutter)
    embedded = plan.get("goals_embedded", []) or []
    titles_pairs = [(g.get("title","(goal)"), g.get("goal_id")) for g in embedded]
    titles_only = [t for (t, _) in titles_pairs] or ["(no goals)"]

    mode = st.radio("Mode", ["Weekly Goal", "Custom (Unplanned)"], horizontal=True)
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

# =========================
# PAGE: WEEKLY PLANNER
# =========================
def last_week_recap_table(user: str, week_start: date):
    prev_ws, prev_we = week_start - timedelta(days=7), week_start - timedelta(days=1)
    prev_plan = get_or_create_week_plan(user, prev_ws)
    prev_alloc = prev_plan.get("allocations_by_goal", {}) or {}
    if not prev_alloc:
        st.info("No plan existed last week.")
        return
    df = get_sessions_df(user)
    if df.empty:
        df = pd.DataFrame(columns=["date","time","pomodoro_type","duration","user","goal_id","task","category"])
        df["date"] = pd.to_datetime(df["date"])
    df["date_only"] = df["date"].dt.date
    mask = (df["date_only"]>=prev_ws) & (df["date_only"]<=prev_we) & (df["pomodoro_type"]=="Work")
    actual = df[mask].groupby("goal_id").size().to_dict()
    rows = []
    titles = {g["goal_id"]: g.get("title","(goal)") for g in prev_plan.get("goals_embedded", [])}
    for gid, planned in prev_alloc.items():
        rows.append({
            "Goal": titles.get(gid, "(missing)"),
            "Planned": int(planned),
            "Actual": int(actual.get(gid, 0)),
            "Carry": max(0, int(planned) - int(actual.get(gid, 0))),
        })
    view = pd.DataFrame(rows).sort_values("Goal")
    st.dataframe(view, use_container_width=True, hide_index=True)

def render_weekly_planner(user: str, picked_date: date):
    st.header("📅 Weekly Planner")

    week_start, week_end = week_bounds(picked_date)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**Week:** {week_start} → {week_end}")
    with c2:
        defs = registry_defaults(user)
        wp = st.number_input("Weekday avg", 0, 12, value=int(defs.get("weekday_poms",3)))
    with c3:
        wend = st.number_input("Weekend avg", 0, 12, value=int(defs.get("weekend_poms",5)))
    if st.button("💾 Save Capacity Defaults"):
        update_registry_defaults(user, wp, wend)
        st.success("Defaults updated")
        st.rerun()

    st.divider()

    # Goals & Priority
    st.subheader("🎯 Goals & Priority")
    goals = list_registry_goals(user, statuses=["New","In Progress","On Hold","Completed","Archived"])
    with st.expander("➕ Add / Update Goal", expanded=False):
        g_title = st.text_input("Title")
        g_type  = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0)
        g_stat  = st.selectbox("Status", ["In Progress","New","On Hold","Completed","Archived"], index=0)
        g_prio  = st.select_slider("Priority", options=[1,2,3], value=2, help="High=3, Medium=2, Low=1")
        if st.button("💾 Save Goal"):
            if g_title.strip():
                upsert_registry_goal(user, g_title.strip(), g_type, g_stat, int(g_prio))
                st.success("Saved goal")
                st.rerun()
            else:
                st.warning("Please enter a title")

    if not goals:
        st.info("Add 3–4 goals to plan your week.")
        return

    cols = st.columns(min(4, len(goals)))
    updated = {}
    for i, g in enumerate(goals):
        with cols[i % len(cols)]:
            st.write(f"**{g.get('title','(goal)')}**")
            val = int(g.get("priority_band", 2))
            val = max(1, min(3, val))
            updated[g["goal_id"]] = st.select_slider("Priority", options=[1,2,3], value=val, key=f"prio_{g['goal_id']}")
    if st.button("💾 Update Priorities"):
        for g in goals:
            gid = g["goal_id"]
            upsert_registry_goal(user,
                                 title=g["title"],
                                 goal_type=g.get("goal_type","Other"),
                                 status=g.get("status","In Progress"),
                                 priority_band=int(updated.get(gid, g.get("priority_band",2))),
                                 target_poms=int(g.get("target_poms",0)),
                                 goal_id=gid)
        st.success("Priorities updated.")
        st.rerun()

    st.divider()

    # Allocation
    st.subheader("🧮 Allocate Weekly Pomodoros")
    plan = get_or_create_week_plan(user, week_start)
    capacity = plan.get("capacity", _capacity_from_defaults(week_start, registry_defaults(user)))
    weights = {g["goal_id"]: int(updated.get(g["goal_id"], g.get("priority_band",2))) for g in goals}
    auto = proportional_allocation(capacity, weights)

    edited = {}
    cols2 = st.columns(min(4, len(goals)))
    for i, g in enumerate(goals):
        with cols2[i % len(cols2)]:
            default_val = int(plan.get("allocations_by_goal", {}).get(g["goal_id"], auto[g["goal_id"]]))
            edited[g["goal_id"]] = st.number_input(g.get("title","(goal)"), 0, capacity, value=default_val, step=1, key=f"alloc_{g['goal_id']}")

    sum_alloc = sum(edited.values())
    if sum_alloc != capacity:
        st.warning(f"Allocations total {sum_alloc}, capacity is {capacity}.")
        if st.button("Normalize to capacity"):
            edited = proportional_allocation(capacity, {k: max(1,v) for k,v in edited.items()})
            for gid, v in edited.items():
                st.session_state[f"alloc_{gid}"] = v
            st.rerun()

    if st.button("📌 Save / Update Weekly Plan", type="primary"):
        save_week_plan(user, week_start, edited)
        st.success("Weekly plan saved.")
        st.rerun()

    st.divider()
    with st.expander("↪️ Rollover unfinished from last week", expanded=False):
        prev_ws, prev_we = week_start - timedelta(days=7), week_start - timedelta(days=1)
        prev_plan = get_or_create_week_plan(user, prev_ws)
        prev_alloc = prev_plan.get("allocations_by_goal", {}) or {}
        if not prev_alloc:
            st.info("No previous plan found.")
        else:
            if st.button(f"Rollover from {prev_ws} → {prev_we}"):
                df = get_sessions_df(user)
                if df.empty:
                    df = pd.DataFrame(columns=["date","time","pomodoro_type","duration","user","goal_id","task","category"])
                    df["date"] = pd.to_datetime(df["date"])
                df["date_only"] = df["date"].dt.date
                mask = (df["date_only"]>=prev_ws) & (df["date_only"]<=prev_we) & (df["pomodoro_type"]=="Work")
                actual_prev = df[mask].groupby("goal_id").size().to_dict()
                carry = {gid: max(0, int(planned) - int(actual_prev.get(gid, 0))) for gid, planned in prev_alloc.items()}
                carry = {gid: v for gid, v in carry.items() if v > 0}
                if not carry:
                    st.info("Nothing to rollover 🎉")
                else:
                    curr = dict(plan.get("allocations_by_goal", {}))
                    for gid, add in carry.items():
                        curr[gid] = int(curr.get(gid, 0)) + int(add)
                    save_week_plan(user, week_start, curr)
                    st.success("Rolled over unfinished pomodoros.")
                    st.rerun()

    st.divider()
    st.subheader("📜 Last Week Recap")
    last_week_recap_table(user, week_start)

    st.divider()
    st.subheader("✅ Close-out (set status)")
    for g in goals:
        cols = st.columns([3,2,1])
        with cols[0]:
            st.write(f"**{g.get('title','(goal)')}**")
        with cols[1]:
            opts = ["In Progress","Completed","On Hold","Archived"]
            idx = opts.index(g.get("status","In Progress")) if g.get("status","In Progress") in opts else 0
            status = st.selectbox("Status", opts, index=idx, key=f"status_{g['goal_id']}")
        with cols[2]:
            if st.button("Apply", key=f"apply_{g['goal_id']}"):
                update_registry_goal_status(user, g["goal_id"], status)
                st.success("Status updated.")
                st.rerun()

# =========================
# PAGE: ANALYTICS
# =========================
def render_analytics(user: str):
    st.header("📊 Analytics & Review")

    mode = st.radio("Mode", ["Week Review", "Trends"], horizontal=True)

    df_all = get_sessions_df(user)
    if df_all.empty:
        st.info("No sessions yet.")
        return

    df_all["date_only"] = df_all["date"].dt.date
    df_work = df_all[df_all["pomodoro_type"]=="Work"].copy()
    df_break = df_all[df_all["pomodoro_type"]=="Break"].copy()

    if mode == "Week Review":
        pick = st.date_input("Review week of", value=now_ist().date())
        ws, we = week_bounds(pick)
        plan = get_or_create_week_plan(user, ws)
        planned = int(sum((plan.get("allocations_by_goal", {}) or {}).values()))
        mask = (df_all["date_only"]>=ws) & (df_all["date_only"]<=we)
        dfw = df_work[mask].copy()
        dfb = df_break[mask].copy()
        work_goal = dfw[dfw["goal_id"].notna()].copy()
        work_custom = dfw[dfw["goal_id"].isna()].copy()
        deep = len(dfw[dfw["duration"]>=23])
        goal_counts = work_goal.groupby("goal_id").size().values.tolist()

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Plan Adherence", pct_or_dash(len(work_goal), planned))
        with c2: st.metric("Capacity Utilization", pct_or_dash(len(dfw), planned))
        with c3: st.metric("Deep-work %", pct_or_dash(deep, len(dfw)))
        with c4: st.metric("Balance (Entropy)", f"{entropy_norm_from_counts(goal_counts):.2f}")

        c5,c6,c7,c8 = st.columns(4)
        with c5: st.metric("Gini (Goals)", f"{gini_from_counts(goal_counts):.2f}")
        with c6: st.metric("Custom Share", pct_or_dash(len(work_custom), len(dfw)))
        with c7:
            exp_breaks = len(dfw)
            skip = max(0, exp_breaks - len(dfb))
            st.metric("Break Skip", pct_or_dash(skip, exp_breaks))
        with c8:
            extend = max(0, len(dfb) - len(dfw))
            st.metric("Break Extend", pct_or_dash(extend, len(dfw)))

        # Run-rate vs Expected (line chart)
        if planned > 0:
            days = pd.date_range(start=pd.to_datetime(ws), end=pd.to_datetime(min(we, now_ist().date())))
            dfw_goal = dfw[dfw["goal_id"].notna()].copy()
            dfw_goal["date_only"] = dfw_goal["date"].dt.date
            actual_cum, exp_cum = [], []
            for i, ts in enumerate(days):
                cutoff = ts.date()
                actual_to_d = int((dfw_goal["date_only"]<=cutoff).sum())
                expected_to_d = int(round(planned * ((i+1)/len(days))))
                actual_cum.append(actual_to_d)
                exp_cum.append(expected_to_d)
            rr = pd.DataFrame({"Expected": exp_cum, "Actual": actual_cum}, index=[d.date() for d in days])
            st.line_chart(rr, height=280)

    else:
        # Trends
        today = now_ist().date()
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("🎯 Total Sessions", len(df_work))
        with c2: st.metric("⏱️ Total Hours", int(df_work["duration"].sum()//60))
        with c3: st.metric("📅 Active Days", int(df_work.groupby("date_only").size().shape[0]))
        with c4:
            avg_daily = df_work.groupby("date_only").size().mean() if len(df_work) else 0
            st.metric("📊 Avg Daily", f"{avg_daily:.1f}")

        st.divider()
        st.subheader("📈 Daily Performance (Last 30 days)")
        day_range = pd.date_range(end=pd.to_datetime(today), periods=30)
        mins = []
        for d in day_range:
            d0 = d.date()
            mins.append(int(df_work[df_work["date_only"]==d0]["duration"].sum()))
        perf = pd.DataFrame({"minutes": mins}, index=[d.date() for d in day_range])
        st.bar_chart(perf, height=220)

        # Insights
        st.markdown("#### 🔍 Insights (Last 30 days)")
        df30 = df_work[df_work["date_only"] >= (today - pd.Timedelta(days=30))].copy()
        if not df30.empty:
            by_day = df30.groupby("date_only")["duration"].sum().sort_values(ascending=False)
            best_day_val = int(by_day.iloc[0]) if len(by_day) else 0
            best_day_lbl = by_day.index[0].strftime("%a %d %b") if len(by_day) else "—"
            starts = [m for m in df30["time"].dropna().map(time_to_minutes) if m is not None]
            if starts:
                top_hour = pd.Series([s//60 for s in starts]).mode().iloc[0]
                ampm = "AM" if top_hour < 12 else "PM"
                focus_hour = f"{(top_hour if 1<=top_hour<=12 else (12 if top_hour%12==0 else top_hour%12))}{ampm}"
            else:
                focus_hour = "—"
            by_cat = df30.groupby("category")["duration"].sum().sort_values(ascending=False)
            top_cat = by_cat.index[0] if len(by_cat) else "—"
            top_share = 100.0 * safe_div(by_cat.iloc[0], by_cat.sum()) if len(by_cat) else 0
            df30_break = df_all[(df_all["pomodoro_type"]=="Break") & (df_all["date_only"] >= (today - pd.Timedelta(days=30)))]
            skip_rate = pct_or_dash(max(0, len(df30) - len(df30_break)), len(df30))
            extend_rate = pct_or_dash(max(0, len(df30_break) - len(df30)), len(df30))
            i1,i2,i3,i4 = st.columns(4)
            with i1: st.metric("Best day (mins)", f"{best_day_val}", best_day_lbl)
            with i2: st.metric("Focus window", focus_hour)
            with i3: st.metric("Top category share", f"{top_share:.0f}%")
            with i4: st.metric("Break skip / extend", f"{skip_rate} / {extend_rate}")

        st.divider()
        st.subheader("🎯 Category mix")
        period = st.selectbox("Period", ["Last 7 days","Last 30 days","All time"], index=1)
        if period == "Last 7 days":
            cutoff = today - pd.Timedelta(days=7)
            fw = df_work[df_work["date_only"]>=cutoff]
        elif period == "Last 30 days":
            cutoff = today - pd.Timedelta(days=30)
            fw = df_work[df_work["date_only"]>=cutoff]
        else:
            fw = df_work
        if fw.empty:
            st.info("No data for selected period.")
            return
        by_cat = fw.groupby("category")["duration"].sum().sort_values(ascending=False)
        if not by_cat.empty:
            st.bar_chart(by_cat, height=240)
        st.caption("Tip: if one bar dominates, consider rebalancing next week.")

        st.subheader("🗂️ Top Tasks")
        tstats = fw.groupby(["category","task"]).agg(total_minutes=("duration","sum"),
                                                     sessions=("duration","count")).reset_index()
        tstats = tstats.sort_values("total_minutes", ascending=False).head(12)
        if tstats.empty:
            st.info("No tasks recorded.")
        else:
            view = tstats.rename(columns={"total_minutes":"Minutes","sessions":"Sessions","task":"Task","category":"Category"})
            st.dataframe(view[["Category","Task","Minutes","Sessions"]], use_container_width=True, hide_index=True)
            total_time = view["Minutes"].sum()
            top = view.iloc[0]
            share = 100.0*safe_div(top["Minutes"], total_time)
            if share > 50:
                st.warning("⚖️ One task dominates your time. Consider splitting or capping it.")
            elif share > 25:
                st.info("🎯 Clear primary task emerging this period.")
            else:
                st.success("✅ Time is well distributed across tasks.")

# =========================
# PAGE: JOURNAL
# =========================
def render_journal(user: str):
    st.header("🧾 Journal")
    tab1, tab2, tab3 = st.tabs(["Reflection", "Daily Target", "Notes"])
    today_iso = now_ist().date().isoformat()

    with tab1:
        st.subheader("End-of-Day Reflection")
        with st.form("reflection_form", clear_on_submit=True):
            aligned = st.selectbox("Aligned with weekly plan?", ["Yes","Partly","No"])
            rating = st.slider("Focus quality (1-5)", 1, 5, 3)
            blockers = st.text_area("Blockers / distractions")
            notes = st.text_area("Insights / anything to note")
            submitted = st.form_submit_button("💾 Save Reflection")
            if submitted:
                save_reflection(user, aligned, rating, blockers.strip(), notes.strip(), today_iso)
                st.success("Saved ✨")

        st.subheader("Recent Reflections")
        rows = []
        for i in range(14):
            d = (now_ist().date() - pd.Timedelta(days=i)).isoformat()
            doc = get_or_create_user_day(user, d)
            if doc.get("reflection"):
                r = doc["reflection"]
                rows.append({"date": d, "aligned": r.get("aligned"), "focus_rating": r.get("focus_rating"),
                             "blockers": r.get("blockers",""), "notes": r.get("notes","")})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No recent reflections yet.")

    with tab2:
        st.subheader("Daily Target")
        cur = get_daily_target(user, today_iso)
        if cur is not None:
            st.info(f"Today's target: **{cur}**")
            new = st.number_input("Update target", 1, 12, value=int(cur))
            if st.button("💾 Update target"):
                set_daily_target(user, int(new), today_iso)
                st.success("Updated target.")
                st.rerun()
        else:
            val = st.number_input("Set target (pomodoros)", 1, 12, value=1)
            if st.button("Set"):
                set_daily_target(user, int(val), today_iso)
                st.success("Saved target.")
                st.rerun()

    with tab3:
        st.subheader("Notes")
        with st.form("note_form", clear_on_submit=True):
            content = st.text_area("Your thoughts...", height=140)
            sub = st.form_submit_button("💾 Save Note")
            if sub:
                if content.strip():
                    add_note(user, content.strip(), today_iso)
                    st.success("Saved")
                else:
                    st.warning("Add some content")

# =========================
# HEADER + ROUTER
# =========================
def main_header_and_router():
    init_state()
    ensure_indexes()

    # Build user choices (with legacy backfills first time)
    users = list_users()
    if not users:
        add_user("prashanth")
        users = list_users()

    # select user (persist)
    c1, c2, c3 = st.columns([2,3,2])
    with c1:
        idx = users.index(st.session_state.user) if st.session_state.user in users else 0
        sel = st.selectbox("👤 User", users, index=idx, key="user_select_header")
        if sel != st.session_state.user:
            st.session_state.user = sel
            reset_runtime_state_for_user()
            # one-time safe backfills for existing data
            legacy_backfill_registry_from_goals(sel)
            legacy_backfill_user_days_from_logs(sel)
            st.experimental_rerun()

    with c2:
        pages = ["🎯 Focus Timer","📅 Weekly Planner","📊 Analytics & Review","🧾 Journal"]
        st.session_state.page = st.selectbox("📍 Navigate", pages,
                                             index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
    with c3:
        with st.expander("➕ Add User"):
            u = st.text_input("Username", key="new_user_input")
            if st.button("Add", key="add_user_btn") and u:
                add_user(u.strip())
                st.success("✅ User added!")
                st.rerun()

    st.divider()
    sidebar_admin(st.session_state.user)

    page = st.session_state.page
    if page == "🎯 Focus Timer":
        render_focus_timer(st.session_state.user)
    elif page == "📅 Weekly Planner":
        # week picker
        pick_date = st.date_input("Plan week of", value=st.session_state.planning_week_date)
        if pick_date != st.session_state.planning_week_date:
            st.session_state.planning_week_date = pick_date
            st.rerun()
        render_weekly_planner(st.session_state.user, st.session_state.planning_week_date)
    elif page == "📊 Analytics & Review":
        render_analytics(st.session_state.user)
    elif page == "🧾 Journal":
        render_journal(st.session_state.user)

if __name__ == "__main__":
    main_header_and_router()
