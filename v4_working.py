# app.py
import streamlit as st
import time
import hashlib
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import pytz
import pandas as pd
from pymongo import MongoClient
import math

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Focus Timer • Weekly Priorities",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'About': "Focus Timer - Weekly Priorities & Pomodoro Management"}
)

POMODORO_MIN = 25
BREAK_MIN = 5
IST = pytz.timezone("Asia/Kolkata")
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"

# =========================
# DB INIT (v2 schema only)
# =========================
@st.cache_resource
def init_db():
    try:
        client = MongoClient(st.secrets["mongo_uri"])
        db = client["time_tracker_db"]
        return db, db["user_days"], db["weekly_plans"]
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

db, col_user_days, col_weekly = init_db()

# =========================
# HELPERS
# =========================
def now_ist() -> datetime:
    return datetime.now(IST)

def week_bounds_ist(d: date) -> Tuple[date, date]:
    weekday = d.weekday()
    start = d - timedelta(days=weekday)
    end = start + timedelta(days=6)
    return start, end

def week_day_counts(week_start: date) -> Tuple[int, int]:
    wd = 0; we = 0
    for i in range(7):
        day = week_start + timedelta(days=i)
        if day.weekday() < 5: wd += 1
        else: we += 1
    return wd, we

def time_to_minutes(tstr: str) -> Optional[int]:
    try:
        dt = datetime.strptime(tstr, "%I:%M %p")
        return dt.hour * 60 + dt.minute
    except Exception:
        return None

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

def gini_from_counts(counts: List[int]) -> float:
    arr = [c for c in counts if c is not None and c >= 0]
    if not arr: return 0.0
    arr = sorted(arr)
    n = len(arr); s = sum(arr)
    if s == 0: return 0.0
    cum = 0.0
    for i, x in enumerate(arr, start=1):
        cum += i * x
    return (2.0 * cum) / (n * s) - (n + 1.0) / n

def entropy_norm_from_counts(counts: List[int]) -> float:
    arr = [c for c in counts if c is not None and c > 0]
    k = len(arr)
    if k <= 1: return 0.0
    s = float(sum(arr))
    H = -sum((c/s) * math.log((c/s), 2) for c in arr)
    return H / math.log(k, 2)

def clamp_priority(v: int) -> int:
    try:
        return max(1, min(3, int(v)))
    except Exception:
        return 2

def goal_id(user: str, title: str) -> str:
    return hashlib.sha256(f"{user}|{title}".encode()).hexdigest()[:16]

def sound_alert():
    # Minimal HTML just to play a sound
    st.components.v1.html(f"""
        <audio id="done" autoplay>
            <source src="{SOUND_PATH}" type="audio/mpeg">
        </audio>
        <script>
            const a = document.getElementById('done');
            if (a) {{ a.volume = 0.6; a.play().catch(() => {{}}); }}
        </script>
    """, height=0)

# =========================
# DATA ACCESS (v2)
# =========================
DEFAULTS = {
    "weekday_poms": 3,
    "weekend_poms": 5,
    "auto_break": True,
    "custom_categories": ["Learning", "Projects", "Research", "Planning"]
}

def ensure_indexes():
    try:
        col_user_days.create_index([("user", 1), ("date", 1)], name="user_date")
        col_user_days.create_index([("_id", 1)], name="_id")
        col_weekly.create_index([("user", 1), ("type", 1)], name="user_type")
        col_weekly.create_index([("user", 1), ("week_start", 1)], name="user_week")
        col_weekly.create_index([("_id", 1)], name="_id")
        st.toast("Indexes ensured.")
    except Exception as e:
        st.warning(f"Index notice: {e}")

def list_users() -> List[str]:
    users_w = col_weekly.distinct("user") or []
    users_d = col_user_days.distinct("user") or []
    users = sorted({u for u in users_w + users_d if isinstance(u, str) and u.strip()})
    return users

def create_registry_if_missing(user: str):
    rid = f"{user}|registry"
    doc = col_weekly.find_one({"_id": rid})
    if doc:
        # Patch legacy or missing keys
        patch = {}
        if "type" not in doc: patch["type"] = "registry"
        if "schema_version" not in doc: patch["schema_version"] = 2
        if "user_defaults" not in doc: patch["user_defaults"] = DEFAULTS.copy()
        if "goals" not in doc: patch["goals"] = {}
        if patch:
            patch["updated_at"] = datetime.utcnow()
            col_weekly.update_one({"_id": rid}, {"$set": patch})
        return

    col_weekly.insert_one({
        "_id": rid,
        "user": user,
        "type": "registry",
        "schema_version": 2,
        "user_defaults": DEFAULTS.copy(),
        "goals": {},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })

def get_registry(user: str) -> Dict:
    create_registry_if_missing(user)
    return col_weekly.find_one({"_id": f"{user}|registry"})

def save_registry_defaults(user: str, weekday_poms: int, weekend_poms: int, auto_break: bool, custom_categories: List[str]):
    col_weekly.update_one(
        {"_id": f"{user}|registry"},
        {"$set": {
            "user_defaults.weekday_poms": int(weekday_poms),
            "user_defaults.weekend_poms": int(weekend_poms),
            "user_defaults.auto_break": bool(auto_break),
            "user_defaults.custom_categories": list(custom_categories),
            "updated_at": datetime.utcnow()
        }}
    )

def upsert_registry_goal(user: str, title: str, goal_type: str, priority_weight: int, status: str = "In Progress", target_poms: int = 0):
    rid = f"{user}|registry"
    gid = goal_id(user, title.strip())
    gdoc = {
        "title": title.strip(),
        "goal_type": goal_type,
        "status": status,
        "priority_weight": clamp_priority(priority_weight),
        "target_poms": int(target_poms),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    # only set created_at on insert
    existing = col_weekly.find_one({"_id": rid, f"goals.{gid}": {"$exists": True}})
    if existing:
        gdoc.pop("created_at", None)
    col_weekly.update_one({"_id": rid}, {"$set": {f"goals.{gid}": gdoc, "updated_at": datetime.utcnow()}}, upsert=True)
    return gid

def set_goal_fields(user: str, gid: str, fields: Dict):
    fields = dict(fields)
    if "priority_weight" in fields:
        fields["priority_weight"] = clamp_priority(fields["priority_weight"])
    fields["updated_at"] = datetime.utcnow()
    col_weekly.update_one({"_id": f"{user}|registry"}, {"$set": {**{f"goals.{gid}.{k}": v for k, v in fields.items()}}})

def get_plan(user: str, week_start: date, create_if_missing=True) -> Dict:
    pid = f"{user}|{week_start.isoformat()}"
    plan = col_weekly.find_one({"_id": pid})
    if plan or not create_if_missing:
        return plan or {}
    reg = get_registry(user)
    wd, we = week_day_counts(week_start)
    w = int(reg["user_defaults"].get("weekday_poms", DEFAULTS["weekday_poms"]))
    e = int(reg["user_defaults"].get("weekend_poms", DEFAULTS["weekend_poms"]))
    total = w * wd + e * we
    doc = {
        "_id": pid,
        "type": "plan",
        "schema_version": 2,
        "user": user,
        "week_start": week_start.isoformat(),
        "week_end": (week_start + timedelta(days=6)).isoformat(),
        "capacity": {"weekday": w, "weekend": e, "total": total},
        "allocations": {},
        "goals": [],
        "goals_embedded": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    col_weekly.insert_one(doc)
    return doc

def save_plan_allocations(user: str, week_start: date, allocations: Dict[str, int]):
    plan = get_plan(user, week_start, create_if_missing=True)
    pid = plan["_id"]
    # keep allocations clean ints >= 0
    clean = {gid: max(0, int(v)) for gid, v in allocations.items()}
    reg = get_registry(user)
    embedded = []
    for gid, planned in clean.items():
        g = reg["goals"].get(gid, {})
        embedded.append({
            "goal_id": gid,
            "title": g.get("title", "(missing)"),
            "priority_weight": int(g.get("priority_weight", 2)),
            "status_at_plan": g.get("status", "In Progress"),
            "planned": int(planned),
            "carryover_in": int(0),
            "carryover_out": int(0),
        })
    col_weekly.update_one(
        {"_id": pid},
        {"$set": {
            "allocations": clean,
            "goals": list(clean.keys()),
            "goals_embedded": embedded,
            "updated_at": datetime.utcnow()
        }}
    )

def proportional_allocation(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    if total <= 0 or not weights:
        return {k: 0 for k in weights.keys()}
    wsum = sum(max(1, int(w)) for w in weights.values())
    raw = {gid: (max(1, int(w)) / wsum) * total for gid, w in weights.items()}
    alloc = {gid: int(v) for gid, v in raw.items()}
    diff = total - sum(alloc.values())
    if diff != 0:
        fracs = sorted(((gid, raw[gid] - int(raw[gid])) for gid in raw), key=lambda x: x[1], reverse=True)
        i = 0
        while diff != 0 and fracs:
            gid = fracs[i % len(fracs)][0]
            alloc[gid] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            i += 1
    return alloc

def user_days_between(user: str, start: date, end: date) -> List[Dict]:
    return list(col_user_days.find({"user": user, "date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}))

def week_actuals_by_goal(user: str, week_start: date) -> Dict[str, int]:
    end = week_start + timedelta(days=6)
    days = user_days_between(user, week_start, end)
    counts: Dict[str, int] = {}
    for d in days:
        for s in d.get("sessions", []):
            if s.get("t") == "W" and s.get("gid"):
                counts[s["gid"]] = counts.get(s["gid"], 0) + 1
    return counts

def week_counts(user: str, week_start: date) -> Dict[str, int]:
    end = week_start + timedelta(days=6)
    days = user_days_between(user, week_start, end)
    work = 0; brk = 0; deep = 0; custom = 0
    for d in days:
        for s in d.get("sessions", []):
            if s.get("t") == "W":
                work += 1
                if int(s.get("dur", 0)) >= 23:
                    deep += 1
                if not s.get("gid"):
                    custom += 1
            elif s.get("t") == "B":
                brk += 1
    return {"work": work, "break": brk, "deep": deep, "custom": custom}

def flatten_user_days(user: str, start: Optional[date] = None, end: Optional[date] = None) -> pd.DataFrame:
    q: Dict = {"user": user}
    if start and end:
        q["date"] = {"$gte": start.isoformat(), "$lte": end.isoformat()}
    docs = list(col_user_days.find(q))
    rows = []
    for d in docs:
        ddate = d.get("date")
        for s in d.get("sessions", []):
            rows.append({
                "date": ddate,
                "time": s.get("time", ""),
                "t": s.get("t"),
                "dur": int(s.get("dur", 0)),
                "gid": s.get("gid"),
                "cat": s.get("cat", ""),
                "task": s.get("task", "")
            })
    if not rows:
        return pd.DataFrame(columns=["date","time","t","dur","gid","cat","task"])
    df = pd.DataFrame(rows)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def update_user_day_after_append(doc: Dict):
    # recompute totals, deep, by_category_minutes, start_time_mins, switches
    sessions = doc.get("sessions", [])
    work = sum(1 for s in sessions if s.get("t") == "W")
    work_min = sum(int(s.get("dur", 0)) for s in sessions if s.get("t") == "W")
    deep = sum(1 for s in sessions if s.get("t") == "W" and int(s.get("dur", 0)) >= 23)
    brk = sum(1 for s in sessions if s.get("t") == "B")
    brk_min = sum(int(s.get("dur", 0)) for s in sessions if s.get("t") == "B")
    by_cat: Dict[str, int] = {}
    for s in sessions:
        if s.get("t") == "W" and s.get("cat"):
            by_cat[s["cat"]] = by_cat.get(s["cat"], 0) + int(s.get("dur", 0))
    starts = []
    for s in sessions:
        m = time_to_minutes(s.get("time", "")) if s.get("time") else None
        if m is not None:
            starts.append(m)
    # switches by key (gid or category)
    switches = 0
    prev = None
    for s in sessions:
        key = s.get("gid") or (f"CAT::{s.get('cat','')}" if s.get("cat") else "NA")
        if prev is not None and key != prev:
            switches += 1
        prev = key
    doc["totals"] = {
        "work_sessions": work, "work_minutes": work_min,
        "break_sessions": brk, "break_minutes": brk_min,
        "deep_work_sessions": deep
    }
    doc["by_category_minutes"] = by_cat
    doc["start_time_mins"] = starts
    doc["switches"] = switches
    doc["updated_at"] = datetime.utcnow()

def append_session(user: str, kind: str, dur_min: int, time_str: str,
                   task: str = "", cat: str = "", gid: Optional[str] = None):
    today = now_ist().date().isoformat()
    _id = f"{user}|{today}"
    doc = col_user_days.find_one({"_id": _id})
    if not doc:
        doc = {
            "_id": _id, "user": user, "date": today, "schema_version": 2,
            "sessions": [], "notes": [], "created_at": datetime.utcnow()
        }
    # Append new block
    block = {"t": ("B" if kind == "B" else "W"), "dur": int(dur_min), "time": time_str}
    if block["t"] == "W":
        if gid: block["gid"] = gid
        if cat: block["cat"] = cat
        if task: block["task"] = task
    doc["sessions"].append(block)
    # Recompute aggregates
    update_user_day_after_append(doc)
    col_user_days.update_one({"_id": _id}, {"$set": doc}, upsert=True)

def export_sessions_csv(user: str) -> bytes:
    df = flatten_user_days(user)
    if df.empty:
        return b""
    df = df.sort_values("date_dt")
    return df.to_csv(index=False).encode("utf-8")

# =========================
# SESSION STATE
# =========================
def init_session_state():
    defaults = {
        "user": None,
        "page": "🎯 Focus Timer",
        "start_time": None,
        "is_break": False,
        "task": "",
        "active_goal_id": None,
        "active_goal_title": "",
        "planning_week_date": now_ist().date(),
        "review_week_date": now_ist().date(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_runtime_state_for_user():
    st.session_state.start_time = None
    st.session_state.is_break = False
    st.session_state.task = ""
    st.session_state.active_goal_id = None
    st.session_state.active_goal_title = ""

init_session_state()

# =========================
# UI COMPONENTS
# =========================
def this_week_glance(user: str):
    today = now_ist().date()
    wk_start, wk_end = week_bounds_ist(today)
    plan = get_plan(user, wk_start, create_if_missing=True)
    alloc = plan.get("allocations", {}) or {}
    if not alloc:
        st.info("No allocations yet for this week. Set them in the Weekly Planner.")
        return
    actual = week_actuals_by_goal(user, wk_start)
    reg = get_registry(user)
    titles = {gid: reg["goals"].get(gid, {}).get("title", "(missing)") for gid in alloc.keys()}

    cols = st.columns(2)
    i = 0
    for gid, planned in alloc.items():
        with cols[i % 2]:
            a = int(actual.get(gid, 0))
            st.write(f"**{titles.get(gid, '(missing)')}**")
            st.progress(min(1.0, safe_div(a, max(1, int(planned)))), text=f"{a}/{int(planned)}")
        i += 1

def start_time_spark(user: str, title="Start-time Stability"):
    today = now_ist().date()
    df = flatten_user_days(user, today - timedelta(days=30), today)
    if df.empty:
        return
    dfw = df[df["t"] == "W"].copy()
    if dfw.empty: return
    dfw["mins"] = dfw["time"].apply(time_to_minutes)
    dfw = dfw[pd.notna(dfw["mins"])]
    if dfw.empty: return
    daily = dfw.groupby("date")["mins"].median().reset_index()
    daily = daily.rename(columns={"date": "Date", "mins": "Median start (mins)"})
    daily = daily.set_index("Date")
    st.caption(title)
    st.line_chart(daily, height=200)

# =========================
# TIMER
# =========================
def render_timer_widget(auto_break: bool) -> bool:
    if not st.session_state.start_time:
        return False
    duration = BREAK_MIN * 60 if st.session_state.is_break else POMODORO_MIN * 60
    remaining = int(st.session_state.start_time + duration - time.time())
    if remaining > 0:
        mins, secs = divmod(remaining, 60)
        session_type = "Break" if st.session_state.is_break else f"Working on: {st.session_state.task or '(untitled)'}"
        st.subheader(f"{'🧘' if st.session_state.is_break else '💼'} {session_type}")
        _, cc, _ = st.columns([1, 2, 1])
        with cc:
            st.markdown(f"<h1 style='text-align:center;font-size:4rem;'>⏱️ {mins:02d}:{secs:02d}</h1>", unsafe_allow_html=True)
        st.progress(1 - (remaining / duration))
        st.info("🧘 Relax" if st.session_state.is_break else "💪 Stay focused!")
        time.sleep(1)
        st.rerun()
        return True
    else:
        # Complete current block
        was_break = st.session_state.is_break
        # Save
        now = now_ist()
        append_session(
            user=st.session_state.user,
            kind=("B" if was_break else "W"),
            dur_min=(BREAK_MIN if was_break else POMODORO_MIN),
            time_str=now.strftime("%I:%M %p"),
            task=(st.session_state.task if not was_break else ""),
            cat=(st.session_state.active_goal_title if (not was_break and st.session_state.active_goal_id is None) else ""),
            gid=(st.session_state.active_goal_id if (not was_break) else None)
        )
        # Play sound immediately
        sound_alert()
        st.balloons()
        st.success("🎉 Session complete!")

        # Reset
        st.session_state.start_time = None
        st.session_state.is_break = False
        st.session_state.task = ""
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = ""

        # Auto break after WORK (sound already played)
        if (not was_break) and auto_break:
            st.toast("☕ Auto-starting a 5-minute break")
            st.session_state.start_time = time.time()
            st.session_state.is_break = True
            st.rerun()
        return True

# =========================
# PAGES
# =========================
def render_focus_timer(user: str):
    st.header("🎯 Focus Timer")

    # Settings toggle (auto-break)
    reg = get_registry(user)
    auto_break_current = bool(reg["user_defaults"].get("auto_break", True))
    left, _ = st.columns([1, 3])
    with left:
        new_val = st.toggle("Auto-start break", value=auto_break_current)
        if new_val != auto_break_current:
            save_registry_defaults(
                user,
                weekday_poms=int(reg["user_defaults"].get("weekday_poms", DEFAULTS["weekday_poms"])),
                weekend_poms=int(reg["user_defaults"].get("weekend_poms", DEFAULTS["weekend_poms"])),
                auto_break=bool(new_val),
                custom_categories=reg["user_defaults"].get("custom_categories", DEFAULTS["custom_categories"])
            )

    # Active timer?
    if render_timer_widget(auto_break=bool(get_registry(user)["user_defaults"].get("auto_break", True))):
        return

    # Daily target widget
    # (Stored on user_days as "daily_target")
    today_id = f"{user}|{now_ist().date().isoformat()}"
    day_doc = col_user_days.find_one({"_id": today_id})
    st.markdown("## 🎯 Daily Target")
    colA, colB = st.columns([2, 3])
    with colA:
        current_target = (day_doc or {}).get("daily_target")
        if current_target is not None:
            st.info(f"Today: **{int(current_target)} Pomodoros**")
            with st.expander("Change Today's Target"):
                new_t = st.number_input("New target", 1, 12, value=int(current_target))
                if st.button("💾 Update Target"):
                    col_user_days.update_one({"_id": today_id},
                                             {"$set": {"daily_target": int(new_t), "updated_at": datetime.utcnow()}},
                                             upsert=True)
                    st.success("Updated!")
                    st.rerun()
        else:
            suggested = 2
            tval = st.number_input("How many Pomodoros today?", 1, 12, value=int(suggested))
            if st.button("Set Target", use_container_width=True):
                col_user_days.update_one({"_id": today_id},
                                         {"$set": {"daily_target": int(tval), "updated_at": datetime.utcnow(),
                                                   "user": user, "date": now_ist().date().isoformat(), "schema_version": 2},
                                          "$setOnInsert": {"created_at": datetime.utcnow()}},
                                         upsert=True)
                st.success("Saved!")
                st.rerun()
    with colB:
        # progress bar for today
        if day_doc and day_doc.get("totals", {}).get("work_sessions"):
            ws = int(day_doc["totals"]["work_sessions"])
        else:
            ws = 0
        tgt = (day_doc or {}).get("daily_target")
        if tgt:
            pct = min(100.0, (ws / max(1, int(tgt))) * 100)
            st.progress(pct/100.0, text=f"{pct:.0f}% complete")
        else:
            st.info("Set a target to unlock tracking.")

    st.divider()

    # This week glance + start-time spark
    st.subheader("📌 This Week at a Glance")
    this_week_glance(user)
    start_time_spark(user)
    st.divider()

    # Mode
    mode = st.radio("Mode", ["Weekly Goal", "Custom (Unplanned)"], horizontal=True)

    if mode == "Weekly Goal":
        wk_start, _ = week_bounds_ist(now_ist().date())
        plan = get_plan(user, wk_start, create_if_missing=True)
        alloc_ids = list(plan.get("allocations", {}).keys())
        reg = get_registry(user)
        titles_pairs = [(reg["goals"].get(gid, {}).get("title", "(missing)"), gid) for gid in alloc_ids]
        labels = [t for (t, _) in titles_pairs] or ["(no goals planned)"]

        c1, c2 = st.columns([1, 2])
        with c1:
            sel_idx = st.selectbox("Weekly Goal", options=range(len(labels)), format_func=lambda i: labels[i],
                                   disabled=(len(labels) == 1 and labels[0] == "(no goals planned)"))
            selected_gid = titles_pairs[sel_idx][1] if titles_pairs else None
            selected_title = titles_pairs[sel_idx][0] if titles_pairs else ""
        with c2:
            task = st.text_input("Task (micro-task)", placeholder="e.g., Revise Unit-2 notes")

        st.session_state.active_goal_id = selected_gid
        st.session_state.active_goal_title = selected_title
        st.session_state.task = task

        colw, colb = st.columns(2)
        with colw:
            disabled = (not task.strip()) or (selected_gid is None)
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
        # Custom
        reg = get_registry(user)
        current_cats = list(reg["user_defaults"].get("custom_categories", DEFAULTS["custom_categories"]))
        cat_options = current_cats + ["+ Add New"]
        selected = st.selectbox("📂 Custom Category", cat_options)
        if selected == "+ Add New":
            new_cat = st.text_input("New category", placeholder="e.g., Marketing")
            if new_cat and st.button("✅ Add Category"):
                if new_cat not in current_cats:
                    current_cats.append(new_cat)
                    save_registry_defaults(
                        user,
                        weekday_poms=int(reg["user_defaults"].get("weekday_poms", DEFAULTS["weekday_poms"])),
                        weekend_poms=int(reg["user_defaults"].get("weekend_poms", DEFAULTS["weekend_poms"])),
                        auto_break=bool(reg["user_defaults"].get("auto_break", True)),
                        custom_categories=current_cats
                    )
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

    # Today's compact summary
    td = col_user_days.find_one({"_id": f"{user}|{now_ist().date().isoformat()}"}) or {}
    totals = td.get("totals", {})
    st.divider(); st.subheader("📊 Today")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Work Sessions", int(totals.get("work_sessions", 0)))
    with c2:
        st.metric("Focus Minutes", int(totals.get("work_minutes", 0)))
    with c3:
        st.metric("Breaks", int(totals.get("break_sessions", 0)))
    with c4:
        tgt = td.get("daily_target")
        if tgt:
            pct = safe_div(int(totals.get("work_sessions", 0)), max(1, int(tgt))) * 100
            st.metric("Target Progress", f"{pct:.0f}%")
        else:
            st.metric("Target Progress", "—")

def render_weekly_planner(user: str):
    st.header("📅 Weekly Planner")

    # Week picker
    pick_date = st.date_input("Week of", value=st.session_state.planning_week_date)
    wk_start, wk_end = week_bounds_ist(pick_date)
    if pick_date != st.session_state.planning_week_date:
        st.session_state.planning_week_date = pick_date
        st.rerun()

    # Defaults / capacity
    reg = get_registry(user)
    wp = int(reg["user_defaults"].get("weekday_poms", DEFAULTS["weekday_poms"]))
    we = int(reg["user_defaults"].get("weekend_poms", DEFAULTS["weekend_poms"]))
    colA, colB, colC = st.columns(3)
    with colA:
        new_wp = st.number_input("Weekday avg", 0, 12, value=wp)
    with colB:
        new_we = st.number_input("Weekend avg", 0, 12, value=we)
    with colC:
        wd, wec = week_day_counts(wk_start)
        capacity_total = new_wp * wd + new_we * wec
        st.metric(f"Capacity {wk_start} → {wk_end}", f"{capacity_total}")
        if new_wp != wp or new_we != we:
            if st.button("💾 Save Capacity Defaults", use_container_width=True):
                save_registry_defaults(
                    user, weekday_poms=int(new_wp), weekend_poms=int(new_we),
                    auto_break=bool(reg["user_defaults"].get("auto_break", True)),
                    custom_categories=reg["user_defaults"].get("custom_categories", DEFAULTS["custom_categories"])
                )
                st.success("Saved defaults")
                st.rerun()

    st.divider()

    # Goals & Priority
    st.subheader("🎯 Goals & Priority")
    with st.expander("➕ Add or Update Goal"):
        g_title = st.text_input("Title", placeholder="e.g., UGC NET Paper 1")
        g_type = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0)
        raw_weight = st.number_input("Priority weight (1=Low, 3=High)", 1, 5, value=3)
        g_weight = clamp_priority(raw_weight)  # robust: clamps 1..3 even if legacy had 5
        g_status = st.selectbox("Status", ["New","In Progress","Completed","On Hold","Archived"], index=1)
        if st.button("💾 Save Goal"):
            if g_title.strip():
                upsert_registry_goal(user, g_title.strip(), g_type, g_weight, g_status)
                st.success("Saved goal")
                st.rerun()
            else:
                st.warning("Please provide a title")

    reg = get_registry(user)  # refresh
    goals_dict = reg.get("goals", {})
    if not goals_dict:
        st.info("Add 3–4 goals to plan the week.")
        return

    # Editable priority weights (robust against legacy 4/5)
    cols = st.columns(min(4, max(1, len(goals_dict))))
    i = 0
    new_weights = {}
    for gid, g in goals_dict.items():
        with cols[i % len(cols)]:
            st.write(f"**{g.get('title','(untitled)')}**")
            w_val = clamp_priority(g.get("priority_weight", 2))
            w = st.select_slider("Priority", options=[1,2,3], value=w_val, key=f"w_{gid}")
            new_weights[gid] = int(w)
        i += 1
    if st.button("💾 Update Priorities"):
        for gid, w in new_weights.items():
            set_goal_fields(user, gid, {"priority_weight": int(w)})
        st.success("Priorities updated.")
        st.rerun()

    st.divider()

    # Allocation
    st.subheader("🧮 Allocate Weekly Pomodoros")
    wd, wec = week_day_counts(wk_start)
    cap_total = int(get_registry(user)["user_defaults"].get("weekday_poms", DEFAULTS["weekday_poms"])) * wd + \
                int(get_registry(user)["user_defaults"].get("weekend_poms", DEFAULTS["weekend_poms"])) * wec

    weight_map = {gid: int(new_weights.get(gid, clamp_priority(g.get("priority_weight", 2))))
                  for gid, g in goals_dict.items()}
    auto_alloc = proportional_allocation(cap_total, weight_map)

    plan = get_plan(user, wk_start, create_if_missing=True)
    current_alloc = plan.get("allocations", {}) or {}

    edited = {}
    cols2 = st.columns(min(4, max(1, len(weight_map))))
    i = 0
    for gid, w in weight_map.items():
        with cols2[i % len(cols2)]:
            title = goals_dict.get(gid, {}).get("title", "(untitled)")
            default_val = int(current_alloc.get(gid, auto_alloc.get(gid, 0)))
            val = st.number_input(title, min_value=0, max_value=cap_total, value=default_val, step=1, key=f"a_{gid}")
            edited[gid] = int(val)
        i += 1

    if sum(edited.values()) != cap_total:
        st.warning(f"Allocations sum to {sum(edited.values())}, not {cap_total}.")
        if st.button("🔁 Normalize to capacity"):
            edited = proportional_allocation(cap_total, edited)
            for gid, v in edited.items():
                st.session_state[f"a_{gid}"] = v
            st.rerun()

    if st.button(("📌 Save Weekly Plan" if not current_alloc else "📌 Update Weekly Plan"), type="primary"):
        save_plan_allocations(user, wk_start, edited)
        st.success("Weekly plan saved!")
        st.rerun()

    # Rollover
    with st.expander("↪️ Rollover unfinished from last week", expanded=False):
        prev_start = wk_start - timedelta(days=7)
        prev_plan = get_plan(user, prev_start, create_if_missing=False)
        if st.button(f"Compute & Rollover from {prev_start} → {prev_start+timedelta(days=6)}"):
            if not prev_plan or not prev_plan.get("allocations"):
                st.warning("No previous plan found.")
            else:
                actual_prev = week_actuals_by_goal(user, prev_start)
                carry = {gid: max(0, int(planned) - int(actual_prev.get(gid, 0)))
                         for gid, planned in prev_plan["allocations"].items()}
                carry = {gid: v for gid, v in carry.items() if v > 0}
                if not carry:
                    st.info("No unfinished items to rollover.")
                else:
                    merged = dict(current_alloc)
                    for gid, add in carry.items():
                        merged[gid] = int(merged.get(gid, 0)) + int(add)
                    save_plan_allocations(user, wk_start, merged)
                    st.success("Rolled over unfinished poms into this week.")
                    st.rerun()

    st.divider()

    # Last Week Recap + Close-out
    st.subheader("📜 Last Week Recap")
    prev_start = wk_start - timedelta(days=7)
    prev_plan = get_plan(user, prev_start, create_if_missing=False)
    if not prev_plan or not prev_plan.get("allocations"):
        st.info("No plan for last week.")
    else:
        actual_prev = week_actuals_by_goal(user, prev_start)
        rows = []
        titles = {gid: goals_dict.get(gid, {}).get("title", "(missing)") for gid in prev_plan["allocations"].keys()}
        for gid, planned in prev_plan["allocations"].items():
            rows.append({
                "Goal": titles.get(gid, "(missing)"),
                "Planned": int(planned),
                "Actual": int(actual_prev.get(gid, 0)),
                "Carry": max(0, int(planned) - int(actual_prev.get(gid, 0)))
            })
        df_rec = pd.DataFrame(rows)
        st.dataframe(df_rec, use_container_width=True, hide_index=True)

        st.subheader("✅ Close-out (Update Goal Status)")
        for gid in prev_plan["allocations"].keys():
            g = goals_dict.get(gid, {})
            col1, col2 = st.columns([3, 2])
            with col1:
                st.write(f"**{g.get('title','(missing)')}**")
            with col2:
                new_status = st.selectbox(
                    "Status", ["Completed", "In Progress", "On Hold", "Archived"],
                    index=["New","In Progress","Completed","On Hold","Archived"].index(g.get("status","In Progress")) if g.get("status") in ["New","In Progress","Completed","On Hold","Archived"] else 1,
                    key=f"s_{gid}_{prev_start}"
                )
            if st.button("Apply", key=f"apply_{gid}_{prev_start}"):
                set_goal_fields(user, gid, {"status": new_status})
                st.success("Updated")

def render_analytics_review(user: str):
    st.header("📊 Analytics & Review")

    try:
        mode = st.segmented_control("Mode", options=["Week Review", "Trends"], default="Week Review", key="analytics_mode")
    except Exception:
        mode = st.radio("Mode", ["Week Review", "Trends"], horizontal=True, index=0)

    # Build a flat df for convenience
    df_all = flatten_user_days(user)
    if df_all.empty:
        st.info("No sessions yet. Start a Pomodoro to populate analytics.")
        return

    df_all["date_only"] = df_all["date_dt"].dt.date

    if mode == "Week Review":
        pick_date = st.date_input("Review week of", value=st.session_state.review_week_date)
        if pick_date != st.session_state.review_week_date:
            st.session_state.review_week_date = pick_date
            st.rerun()
        wk_start, wk_end = week_bounds_ist(pick_date)
        plan = get_plan(user, wk_start, create_if_missing=False) or {}
        planned_alloc = plan.get("allocations", {}) or {}
        total_planned = int(sum(planned_alloc.values())) if planned_alloc else 0

        mask = (df_all["date_only"] >= wk_start) & (df_all["date_only"] <= wk_end)
        dfw = df_all[mask & (df_all["t"] == "W")].copy()
        dfb = df_all[mask & (df_all["t"] == "B")].copy()

        work_goal = dfw[pd.notna(dfw["gid"])].copy()
        work_custom = dfw[pd.isna(dfw["gid"])].copy()
        deep = int((dfw["dur"] >= 23).sum())

        goal_counts = work_goal.groupby("gid").size().values.tolist() if not work_goal.empty else []

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Plan Adherence", pct_or_dash(len(work_goal), total_planned))
        with c2:
            st.metric("Capacity Utilization", pct_or_dash(len(dfw), total_planned))
        with c3:
            st.metric("Deep-work %", pct_or_dash(deep, len(dfw)))
        with c4:
            st.metric("Balance (Entropy)", f"{entropy_norm_from_counts(goal_counts):.2f}")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Gini (Goals)", f"{gini_from_counts(goal_counts):.2f}")
        with c6:
            st.metric("Custom Share", pct_or_dash(len(work_custom), len(dfw)))
        with c7:
            expected_breaks = len(dfw)
            skip = max(0, expected_breaks - len(dfb))
            st.metric("Break Skip", pct_or_dash(skip, expected_breaks))
        with c8:
            extend = max(0, len(dfb) - expected_breaks)
            st.metric("Break Extend", pct_or_dash(extend, expected_breaks))

        st.divider()
        st.subheader("Planned vs Actual (per goal)")
        if planned_alloc:
            reg = get_registry(user)
            titles = {gid: reg["goals"].get(gid, {}).get("title", "(missing)") for gid in planned_alloc.keys()}
            actual_by_gid = week_actuals_by_goal(user, wk_start)
            rows = []
            for gid, p in planned_alloc.items():
                rows.append({
                    "Goal": titles.get(gid, "(missing)"),
                    "Planned": int(p),
                    "Actual": int(actual_by_gid.get(gid, 0))
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No planned allocations this week.")

        # Run-rate vs Expected
        if total_planned > 0:
            days = pd.date_range(start=pd.to_datetime(wk_start), end=pd.to_datetime(min(wk_end, now_ist().date())))
            dfw_goal = dfw[pd.notna(dfw["gid"])].copy()
            dfw_goal["date_only"] = dfw_goal["date_dt"].dt.date
            actual_cum, exp_cum = [], []
            for i, ts in enumerate(days):
                cutoff = ts.date()
                actual_to_d = int((dfw_goal["date_only"] <= cutoff).sum())
                expected_to_d = int(round(total_planned * ((i + 1) / len(days))))
                actual_cum.append(actual_to_d)
                exp_cum.append(expected_to_d)
            rr = pd.DataFrame({"day": [ts.strftime("%a %d") for ts in days],
                               "Expected": exp_cum, "Actual": actual_cum}).set_index("day")
            st.subheader("Run-Rate vs Expected (Goals only)")
            st.line_chart(rr, height=260)

    else:
        # Trends
        today = now_ist().date()
        st.subheader("Overview")
        dfw = df_all[df_all["t"] == "W"].copy()
        mins_total = int(dfw["dur"].sum()) if not dfw.empty else 0
        sessions_total = int(dfw.shape[0])
        active_days = int(dfw.groupby("date_only").size().shape[0]) if not dfw.empty else 0
        avg_daily = float(dfw.groupby("date_only").size().mean()) if not dfw.empty else 0.0

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("🎯 Total Sessions", sessions_total)
        with c2: st.metric("⏱️ Total Hours", mins_total // 60)
        with c3: st.metric("📅 Active Days", active_days)
        with c4: st.metric("📊 Avg Daily", f"{avg_daily:.1f}")

        st.divider()
        st.subheader("📈 Daily Focus Minutes (Last 30 Days)")
        daily = []
        for i in range(30):
            d = today - timedelta(days=29 - i)
            dmins = int(dfw[dfw["date_only"] == d]["dur"].sum()) if not dfw.empty else 0
            daily.append({"Date": d.strftime("%m/%d"), "Minutes": dmins})
        ddf = pd.DataFrame(daily).set_index("Date")
        st.bar_chart(ddf, height=240)

        # Insights
        st.markdown("#### 🔍 Insights (Last 30 days)")
        df30 = dfw[dfw["date_only"] >= (today - timedelta(days=30))].copy()
        if not df30.empty:
            by_day = df30.groupby("date_only")["dur"].sum().sort_values(ascending=False)
            best_day = by_day.index[0] if len(by_day) else None
            best_day_min = int(by_day.iloc[0]) if len(by_day) else 0
            starts = [time_to_minutes(t) for t in df30["time"].tolist() if isinstance(t, str)]
            starts = [s for s in starts if s is not None]
            if starts:
                top_hour = pd.Series([s // 60 for s in starts]).mode().iloc[0]
                ampm = "AM" if top_hour < 12 else "PM"
                hour_disp = f"{(top_hour if 1 <= top_hour <= 12 else (12 if top_hour%12==0 else top_hour%12))}{ampm}"
            else:
                hour_disp = "—"
            by_cat = df30.groupby("cat")["dur"].sum().sort_values(ascending=False)
            if len(by_cat) > 0:
                top_cat = by_cat.index[0] if isinstance(by_cat.index[0], str) and by_cat.index[0] else "Uncategorized"
                top_share = safe_div(by_cat.iloc[0], by_cat.sum()) * 100
            else:
                top_cat, top_share = "—", 0
            # Break hygiene
            df30b = df_all[(df_all["t"] == "B") & (df_all["date_only"] >= (today - timedelta(days=30)))]
            skip_rate = pct_or_dash(max(0, len(df30) - len(df30b)), len(df30))
            extend_rate = pct_or_dash(max(0, len(df30b) - len(df30)), len(df30))

            cI1, cI2, cI3, cI4 = st.columns(4)
            with cI1: st.metric("Best day (mins)", f"{best_day_min}", f"{best_day.strftime('%a %d %b') if best_day else '—'}")
            with cI2: st.metric("Focus window", hour_disp)
            with cI3: st.metric("Top category share", f"{top_share:.0f}%")
            with cI4: st.metric("Break skip / extend", f"{skip_rate} / {extend_rate}")

        st.divider()
        st.subheader("🎯 Category Mix")
        opt = st.selectbox("Time Period", ["Last 7 days", "Last 30 days", "All time"], index=1)
        if opt == "Last 7 days":
            cutoff = today - timedelta(days=7)
            fw = dfw[dfw["date_only"] >= cutoff]
        elif opt == "Last 30 days":
            cutoff = today - timedelta(days=30)
            fw = dfw[dfw["date_only"] >= cutoff]
        else:
            fw = dfw
        if fw.empty:
            st.info("No data for selected period.")
            return
        cat = fw.groupby("cat")["dur"].sum().sort_values(ascending=False).reset_index().rename(columns={"dur":"Minutes"})
        cat = cat.set_index("cat")
        st.bar_chart(cat, height=260)

        st.subheader("🎯 Task Performance")
        tstats = fw.groupby(["cat","task"]).agg(total_minutes=("dur","sum"), sessions=("dur","count")).reset_index()
        if tstats.empty:
            st.info("No tasks logged.")
        else:
            top_tasks = tstats.sort_values("total_minutes", ascending=False).head(12)
            # Table with insights
            view = top_tasks.copy()
            view = view.rename(columns={"cat":"Category","task":"Task","total_minutes":"Minutes","sessions":"Sessions"})
            st.dataframe(view, use_container_width=True, hide_index=True)
            # Quick insights
            total_time = tstats["total_minutes"].sum()
            top = tstats.sort_values("total_minutes", ascending=False).iloc[0]
            share = safe_div(top["total_minutes"], total_time) * 100
            if share > 50:
                st.warning("⚖️ One task dominates your time. Consider rebalancing.")
            elif share > 25:
                st.info("🎯 Clear primary task this period.")
            else:
                st.success("✅ Time is well distributed across tasks.")

def render_journal(user: str):
    st.header("🧾 Journal")
    tab1, tab2, tab3 = st.tabs(["Reflection", "Daily Target", "Notes"])

    today_iso = now_ist().date().isoformat()
    day_id = f"{user}|{today_iso}"

    with tab1:
        st.subheader("End-of-Day Reflection")
        doc = col_user_days.find_one({"_id": day_id}) or {}
        ref = doc.get("reflection", {})
        with st.form("reflection_form", clear_on_submit=False):
            aligned = st.selectbox("Aligned with weekly plan?", ["Yes","Partly","No"], index=["Yes","Partly","No"].index(ref.get("aligned","Yes")) if ref.get("aligned") in ["Yes","Partly","No"] else 0)
            rating = st.slider("Focus quality (1-5)", 1, 5, int(ref.get("focus_rating", 3)))
            blockers = st.text_area("Blockers / distractions", value=ref.get("blockers",""))
            notes = st.text_area("Insights / anything to note", value=ref.get("notes",""))
            if st.form_submit_button("💾 Save Reflection"):
                col_user_days.update_one(
                    {"_id": day_id},
                    {"$set": {
                        "user": user, "date": today_iso, "schema_version": 2,
                        "reflection": {"aligned": aligned, "focus_rating": int(rating),
                                       "blockers": blockers.strip(), "notes": notes.strip()},
                        "updated_at": datetime.utcnow()
                    }, "$setOnInsert": {"created_at": datetime.utcnow()}},
                    upsert=True
                )
                st.success("Saved ✨")

    with tab2:
        st.subheader("Daily Target")
        doc = col_user_days.find_one({"_id": day_id}) or {}
        cur = doc.get("daily_target")
        c1, c2 = st.columns([1,2])
        with c1:
            new_t = st.number_input("Pomodoros today", 1, 12, value=int(cur) if cur else 2)
        with c2:
            if st.button("💾 Save Target"):
                col_user_days.update_one({"_id": day_id},
                                         {"$set": {"daily_target": int(new_t), "updated_at": datetime.utcnow(),
                                                   "user": user, "date": today_iso, "schema_version": 2},
                                          "$setOnInsert": {"created_at": datetime.utcnow()}},
                                         upsert=True)
                st.success("Saved")

    with tab3:
        st.subheader("Notes")
        doc = col_user_days.find_one({"_id": day_id}) or {}
        with st.form("note_form", clear_on_submit=True):
            content = st.text_area("Your note...", height=140)
            if st.form_submit_button("💾 Save Note"):
                if content.strip():
                    notes = doc.get("notes", [])
                    notes.append({"content": content.strip(), "created_at": datetime.utcnow()})
                    col_user_days.update_one({"_id": day_id},
                                             {"$set": {"notes": notes, "user": user, "date": today_iso, "schema_version": 2,
                                                       "updated_at": datetime.utcnow()},
                                              "$setOnInsert": {"created_at": datetime.utcnow()}},
                                             upsert=True)
                    st.success("Saved")
                else:
                    st.warning("Add some content")

# =========================
# HEADER + ROUTER
# =========================
def main_header_and_router():
    # Initial users
    users = list_users()
    if not users:
        # Bootstrap with a default user
        create_registry_if_missing("prashanth")
        users = list_users()

    # Ensure session has a user
    if st.session_state.user is None or st.session_state.user not in users:
        st.session_state.user = users[0]

    # Sidebar Admin
    st.sidebar.markdown("### ⚙️ Admin")
    if st.sidebar.button("Ensure Indexes"):
        ensure_indexes()
    data = export_sessions_csv(st.session_state.user)
    if data:
        st.sidebar.download_button("⬇️ Export Sessions (CSV)", data, file_name=f"{st.session_state.user}_sessions.csv", mime="text/csv")
    else:
        st.sidebar.info("No sessions to export yet.")

    # Top controls
    c1, c2, c3 = st.columns([2,3,2])
    with c1:
        try:
            idx = users.index(st.session_state.user) if st.session_state.user in users else 0
        except Exception:
            idx = 0
        sel = st.selectbox("👤 User", users, index=idx, key="user_select_header")
        if sel != st.session_state.user:
            st.session_state.user = sel
            reset_runtime_state_for_user()
            st.rerun()
    with c2:
        pages = ["🎯 Focus Timer", "📅 Weekly Planner", "📊 Analytics & Review", "🧾 Journal"]
        current = st.session_state.get("page", pages[0])
        st.session_state.page = st.selectbox("📍 Navigate", pages, index=pages.index(current) if current in pages else 0)
    with c3:
        with st.expander("➕ Add User"):
            u = st.text_input("Username", key="new_user_input")
            if st.button("Add", key="add_user_btn") and u:
                uname = u.strip()
                if uname in list_users():
                    st.warning("User already exists!")
                else:
                    create_registry_if_missing(uname)
                    st.session_state.user = uname
                    reset_runtime_state_for_user()
                    st.success("✅ User added!")
                    st.rerun()

    st.divider()
    page = st.session_state.page
    if page == "🎯 Focus Timer":
        render_focus_timer(st.session_state.user)
    elif page == "📅 Weekly Planner":
        render_weekly_planner(st.session_state.user)
    elif page == "📊 Analytics & Review":
        render_analytics_review(st.session_state.user)
    elif page == "🧾 Journal":
        render_journal(st.session_state.user)

if __name__ == "__main__":
    main_header_and_router()
