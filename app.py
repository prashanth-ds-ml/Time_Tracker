# app.py — Weekly Planner + Analytics upgrade (clean + stable)
import streamlit as st
import time
import hashlib
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import pytz
import pandas as pd
from pymongo import MongoClient
import math
import altair as alt

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
    weekday = d.weekday()  # Mon=0
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

def clamp_priority(v: int) -> int:
    try:
        return max(1, min(3, int(v)))
    except Exception:
        return 2

def clamp_w(v: int) -> int:
    try:
        return max(1, min(10, int(v)))
    except Exception:
        return 5

def goal_id(user: str, title: str) -> str:
    return hashlib.sha256(f"{user}|{title}".encode()).hexdigest()[:16]

def sound_alert():
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

def _has_index_with_keys(col, keys_dict: dict) -> bool:
    try:
        for ix in col.list_indexes():
            if dict(ix.get("key", {})) == keys_dict:
                return True
    except Exception:
        pass
    return False

def ensure_indexes():
    try:
        # --- user_days ---
        if not _has_index_with_keys(col_user_days, {"user": 1, "date": 1}):
            col_user_days.create_index([("user", 1), ("date", 1)], name="user_date")
        for kdict, name in [
            ({"sessions.gid": 1},        "sessions_gid"),
            ({"sessions.linked_gid": 1}, "sessions_linked_gid"),
            ({"sessions.unplanned": 1},  "sessions_unplanned"),
            ({"sessions.cat": 1},        "sessions_cat"),
        ]:
            if not _has_index_with_keys(col_user_days, kdict):
                col_user_days.create_index(list(kdict.items()), name=name)

        # --- weekly_plans ---
        if not _has_index_with_keys(col_weekly, {"user": 1, "type": 1}):
            col_weekly.create_index([("user", 1), ("type", 1)], name="user_type")
        if not _has_index_with_keys(col_weekly, {"user": 1, "week_start": 1}):
            col_weekly.create_index([("user", 1), ("week_start", 1)], name="user_week")
        st.toast("Indexes ensured.")
    except Exception:
        return

def list_users() -> List[str]:
    users_w = col_weekly.distinct("user") or []
    users_d = col_user_days.distinct("user") or []
    users = sorted({u for u in users_w + users_d if isinstance(u, str) and u.strip()})
    return users

def create_registry_if_missing(user: str):
    rid = f"{user}|registry"
    doc = col_weekly.find_one({"_id": rid})
    if doc:
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
    col_weekly.update_one({"_id": f"{user}|{ 'registry' }"}, {"$set": {**{f"goals.{gid}.{k}": v for k, v in fields.items()}}})

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
        "weights": {},                 # <-- per-week weights (gid -> 1..10)
        "allocations": {},             # <-- gid -> int
        "goals": [],
        "goals_embedded": [],
        "rollover_from": None,         # <-- guard to avoid double-rollover
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    col_weekly.insert_one(doc)
    return doc

def save_plan_allocations(user: str, week_start: date, allocations: Dict[str, int], weights: Optional[Dict[str,int]] = None, meta: Optional[Dict]=None):
    plan = get_plan(user, week_start, create_if_missing=True)
    pid = plan["_id"]
    clean_alloc = {gid: max(0, int(v)) for gid, v in (allocations or {}).items()}
    weights = weights or {}
    reg = get_registry(user)
    embedded = []
    for gid, planned in clean_alloc.items():
        g = (reg.get("goals") or {}).get(gid, {})
        embedded.append({
            "goal_id": gid,
            "title": g.get("title", "(missing)"),
            "priority_weight": int(g.get("priority_weight", 2)),
            "status_at_plan": g.get("status", "In Progress"),
            "planned": int(planned),
            "carryover_in": int(0),
            "carryover_out": int(0),
        })
    update = {
        "allocations": clean_alloc,
        "goals": list(clean_alloc.keys()),
        "goals_embedded": embedded,
        "updated_at": datetime.utcnow()
    }
    if weights:
        update["weights"] = {k: clamp_w(v) for k, v in weights.items()}
    if meta:
        update.update(meta)
    col_weekly.update_one({"_id": pid}, {"$set": update})

def user_days_between(user: str, start: date, end: date) -> List[Dict]:
    return list(col_user_days.find({"user": user, "date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}))

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
                "linked_gid": s.get("linked_gid"),
                "cat": s.get("cat", ""),
                "task": s.get("task", "")
            })
    if not rows:
        return pd.DataFrame(columns=["date","time","t","dur","gid","linked_gid","cat","task"])
    df = pd.DataFrame(rows)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def update_user_day_after_append(doc: Dict):
    sessions = doc.get("sessions", [])
    work = [s for s in sessions if s.get("t") == "W"]
    brk  = [s for s in sessions if s.get("t") == "B"]
    deep = [s for s in work if int(s.get("dur",0)) >= 23]

    by_cat: Dict[str, int] = {}
    for s in work:
        cat = s.get("cat")
        if cat:
            by_cat[cat] = by_cat.get(cat, 0) + int(s.get("dur",0))

    starts = []
    for s in sessions:
        m = time_to_minutes(s.get("time", "")) if s.get("time") else None
        if m is not None:
            starts.append(m)

    switches = 0
    prev = None
    for s in sessions:
        key = s.get("gid") or s.get("linked_gid") or (f"CAT::{s.get('cat','')}" if s.get("cat") else "NA")
        if prev is not None and key != prev:
            switches += 1
        prev = key

    doc["totals"] = {
        "work_sessions": len(work),
        "work_minutes": sum(int(s.get("dur",0)) for s in work),
        "break_sessions": len(brk),
        "break_minutes": sum(int(s.get("dur",0)) for s in brk),
        "deep_work_sessions": len(deep)
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
    block = {"t": ("B" if kind == "B" else "W"), "dur": int(dur_min), "time": time_str}
    if block["t"] == "W":
        if gid:
            block["gid"] = gid
        else:
            if cat: block["cat"] = cat
        if task: block["task"] = task
    doc["sessions"].append(block)
    update_user_day_after_append(doc)
    col_user_days.update_one({"_id": _id}, {"$set": doc}, upsert=True)

def export_sessions_csv(user: str) -> bytes:
    df = flatten_user_days(user)
    if df.empty:
        return b""
    df = df.sort_values("date_dt")
    return df.to_csv(index=False).encode("utf-8")

# =========================
# DISCIPLINE SCORE (lightweight)
# =========================

# Weights (sum=100)
WEIGHT_PLAN_ADHERENCE = 25
WEIGHT_PRIORITY_ALIGN = 20
WEIGHT_DEEP_WORK      = 15
WEIGHT_UNPLANNED      = 20
WEIGHT_REFLECTION     = 20

def _iter_sessions(user: str, start_d: date, end_d: date):
    docs = user_days_between(user, start_d, end_d)
    for d in docs:
        for s in d.get("sessions", []):
            yield s

def _is_work(s): return s.get("t") == "W"
def _is_deep(s): 
    try: return _is_work(s) and int(s.get("dur",0)) >= 23
    except: return False
def _has_gid(s): return bool(s.get("gid")) or bool(s.get("linked_gid"))

def plan_adherence_ratio(user: str, week_start: date, upto_date: Optional[date] = None) -> Tuple[float, int, int]:
    plan = get_plan(user, week_start, create_if_missing=False) or {}
    planned_total = int(sum((plan.get("allocations") or {}).values())) if plan else 0
    week_end = week_start + timedelta(days=6)
    cutoff = min(week_end, upto_date) if upto_date else week_end
    days_elapsed = (cutoff - week_start).days + 1
    expected = int(round(planned_total * (days_elapsed / 7.0))) if planned_total > 0 else 0
    actual = 0
    for s in _iter_sessions(user, week_start, cutoff):
        if _is_work(s) and _has_gid(s):
            actual += 1
    if planned_total == 0 or expected <= 0:
        return (1.0, actual, 0)
    return (min(1.0, actual / expected), actual, expected)

def top2_goals(user: str) -> List[str]:
    reg = get_registry(user) or {}
    goals = reg.get("goals", {}) or {}
    if not goals: return []
    items = sorted(goals.items(), key=lambda kv: (int(kv[1].get("priority_weight", 2)), kv[1].get("updated_at", datetime.min)), reverse=True)
    return [gid for gid, _ in items[:2]]

def priority_alignment_ratio(user: str, start_d: date, end_d: date) -> Tuple[float,int,int]:
    t2 = set(top2_goals(user))
    total = 0; hits = 0
    for s in _iter_sessions(user, start_d, end_d):
        if not _is_work(s): continue
        total += 1
        gid = s.get("gid") or s.get("linked_gid")
        if gid in t2: hits += 1
    if total == 0: return (0.0, 0, 0)
    return (hits/total, hits, total)

def deep_work_ratio(user: str, start_d: date, end_d: date) -> Tuple[float,int,int]:
    total = 0; deep = 0
    for s in _iter_sessions(user, start_d, end_d):
        if not _is_work(s): continue
        total += 1
        if _is_deep(s): deep += 1
    if total == 0: return (0.0, 0, 0)
    return (deep/total, deep, total)

def unplanned_share_ratio(user: str, start_d: date, end_d: date) -> Tuple[float,int,int]:
    total = 0; unp = 0
    for s in _iter_sessions(user, start_d, end_d):
        if not _is_work(s): continue
        total += 1
        if not _has_gid(s): unp += 1
    if total == 0: return (0.0, 0, 0)
    return (unp/total, unp, total)

def reflection_bits_for_day(user: str, d: date) -> Dict:
    day_doc = col_user_days.find_one({"_id": f"{user}|{d.isoformat()}"}) or {}
    ref = day_doc.get("reflection", {}) or {}
    has_reflection = bool(ref) or (day_doc.get("daily_target") is not None) or bool(day_doc.get("notes"))
    focus = int(ref.get("focus_rating", 0)) if str(ref.get("focus_rating","")).isdigit() else 0
    return {"has": has_reflection, "focus": focus}

def daily_discipline_score(user: str, day: date) -> Dict:
    wk_start, wk_end = week_bounds_ist(day)
    pa_ratio, pa_actual, pa_expected = plan_adherence_ratio(user, wk_start, upto_date=day)
    pa_score = pa_ratio * WEIGHT_PLAN_ADHERENCE

    pr_ratio, pr_hits, pr_total = priority_alignment_ratio(user, day, day)
    pr_score = pr_ratio * WEIGHT_PRIORITY_ALIGN

    dw_ratio, dw_hits, dw_total = deep_work_ratio(user, day, day)
    dw_score = dw_ratio * WEIGHT_DEEP_WORK

    up_ratio, up_count, up_total = unplanned_share_ratio(user, day, day)
    up_score = max(0.0, (1.0 - up_ratio)) * WEIGHT_UNPLANNED

    rb = reflection_bits_for_day(user, day)
    ref_score = 0.0
    if rb["has"]: ref_score += 10
    if rb["focus"] > 0: ref_score += (rb["focus"]/5.0)*10  # up to 10
    ref_score = min(WEIGHT_REFLECTION, ref_score)

    total = round(pa_score + pr_score + dw_score + up_score + ref_score, 1)
    return {"total": total, "components": {
        "plan_adherence": {"score": round(pa_score,1), "ratio": round(pa_ratio,3), "actual": pa_actual, "expected": pa_expected},
        "priority_alignment": {"score": round(pr_score,1), "ratio": round(pr_ratio,3), "hits": pr_hits, "total": pr_total},
        "deep_work": {"score": round(dw_score,1), "ratio": round(dw_ratio,3), "deep": dw_hits, "total": dw_total},
        "unplanned": {"score": round(up_score,1), "ratio": round(up_ratio,3), "unplanned": up_count, "total": up_total},
        "reflection": {"score": round(ref_score,1), "has": rb["has"], "focus": rb["focus"]}
    }}

def weekly_discipline_score(user: str, week_start: date) -> Dict:
    week_end = week_start + timedelta(days=6)
    pa_ratio_full, pa_actual, pa_expected = plan_adherence_ratio(user, week_start, upto_date=week_end)
    plan = get_plan(user, week_start, create_if_missing=False) or {}
    planned_total = int(sum((plan.get("allocations") or {}).values())) if plan else 0
    if planned_total > 0:
        pa_ratio_full = min(1.0, pa_actual / planned_total)
        pa_expected = planned_total
    pa_score = pa_ratio_full * WEIGHT_PLAN_ADHERENCE

    pr_ratio, pr_hits, pr_total = priority_alignment_ratio(user, week_start, week_end)
    pr_score = pr_ratio * WEIGHT_PRIORITY_ALIGN

    dw_ratio, dw_hits, dw_total = deep_work_ratio(user, week_start, week_end)
    dw_score = dw_ratio * WEIGHT_DEEP_WORK

    up_ratio, up_count, up_total = unplanned_share_ratio(user, week_start, week_end)
    up_score = max(0.0, (1.0 - up_ratio)) * WEIGHT_UNPLANNED

    # reflection weekly proxy
    answered = 0; focus_sum = 0; focus_cnt = 0
    for i in range(7):
        d = week_start + timedelta(days=i)
        rb = reflection_bits_for_day(user, d)
        if rb["has"]: answered += 1
        if rb["focus"] > 0:
            focus_sum += rb["focus"]; focus_cnt += 1
    ref_score = 0.0
    ref_score += (answered/7.0)*10.0
    if focus_cnt>0: ref_score += ((focus_sum/(5.0*focus_cnt))*10.0)
    ref_score = min(WEIGHT_REFLECTION, ref_score)

    total = round(pa_score + pr_score + dw_score + up_score + ref_score, 1)
    return {"total": total, "components":{
        "plan_adherence":{"score": round(pa_score,1), "ratio": round(pa_ratio_full,3), "actual": pa_actual, "planned_total": planned_total},
        "priority_alignment":{"score": round(pr_score,1), "ratio": round(pr_ratio,3), "hits": pr_hits, "total": pr_total},
        "deep_work":{"score": round(dw_score,1), "ratio": round(dw_ratio,3), "deep": dw_hits, "total": dw_total},
        "unplanned":{"score": round(up_score,1), "ratio": round(up_ratio,3), "unplanned": up_count, "total": up_total},
        "reflection":{"score": round(ref_score,1), "answered_days": answered}
    }}

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
# SMALL UTILITIES FOR PLANS
# =========================
def actuals_by_gid(user: str, start_d: date, end_d: date) -> Dict[str,int]:
    docs = user_days_between(user, start_d, end_d)
    counts: Dict[str,int] = {}
    for d in docs:
        for s in d.get("sessions", []):
            if s.get("t") == "W" and (s.get("gid") or s.get("linked_gid")):
                gid = s.get("gid") or s.get("linked_gid")
                counts[gid] = counts.get(gid, 0) + 1
    return counts

def proportional_allocation(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    if total <= 0 or not weights:
        return {k: 0 for k in weights.keys()}
    wsum = sum(max(1, int(w)) for w in weights.values())
    raw = {gid: (max(1, int(w))/wsum)*total for gid, w in weights.items()}
    alloc = {gid: int(v) for gid, v in raw.items()}
    diff = total - sum(alloc.values())
    if diff != 0:
        fracs = sorted(((gid, raw[gid]-int(raw[gid])) for gid in raw), key=lambda x: x[1], reverse=True)
        i = 0
        while diff != 0 and fracs:
            gid = fracs[i % len(fracs)][0]
            alloc[gid] += 1 if diff>0 else -1
            diff += -1 if diff>0 else 1
            i += 1
    return alloc

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
    actual = actuals_by_gid(user, wk_start, wk_end)
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
        was_break = st.session_state.is_break
        now = now_ist()
        append_session(
            user=st.session_state.user,
            kind=("B" if was_break else "W"),
            dur_min=(BREAK_MIN if was_break else POMODORO_MIN),
            time_str=now.strftime("%I:%M %p"),
            task=(st.session_state.task if not was_break else ""),
            cat=(st.session_state.active_goal_title if (not was_break and st.session_state.active_goal_id is None) else ""),
            gid=(st.session_state.active_goal_id if (not was_break) else None),
        )
        sound_alert()
        st.balloons()
        st.success("🎉 Session complete!")
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

# =========================
# PAGES
# =========================
def render_focus_timer(user: str):
    st.header("🎯 Focus Timer")
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

    if render_timer_widget(auto_break=bool(get_registry(user)["user_defaults"].get("auto_break", True))):
        return

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
        ws = int((day_doc or {}).get("totals", {}).get("work_sessions", 0))
        tgt = (day_doc or {}).get("daily_target")
        if tgt:
            pct = min(100.0, (ws / max(1, int(tgt))) * 100)
            st.progress(pct/100.0, text=f"{pct:.0f}% complete")
        else:
            st.info("Set a target to unlock tracking.")

    st.divider()
    st.subheader("📌 This Week at a Glance")
    this_week_glance(user)
    start_time_spark(user)
    st.divider()

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

# ---------- Weekly Planner ----------
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
    st.subheader("🎯 Goals")
    with st.expander("➕ Add/Update Goal"):
        g_title = st.text_input("Title", placeholder="e.g., UGC NET Paper 1")
        g_type = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0)
        raw_weight = st.number_input("Registry priority weight (1=Low, 3=High)", 1, 5, value=3)
        g_weight = clamp_priority(raw_weight)
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

    # Weekly weights (dynamic)
    st.subheader("⚖️ Weekly Weights (1–10)")
    plan = get_plan(user, wk_start, create_if_missing=True)
    weekly_w = dict(plan.get("weights", {}))
    cols = st.columns(min(4, max(1, len(goals_dict))))
    new_weekly_w = {}
    i = 0
    for gid, g in goals_dict.items():
        with cols[i % len(cols)]:
            st.write(f"**{g.get('title','(untitled)')}**")
            cur = int(weekly_w.get(gid, clamp_w(5)))
            w = st.slider("Weight", 1, 10, value=cur, key=f"ww_{gid}")
            new_weekly_w[gid] = int(w)
        i += 1
    col_wx, col_rw = st.columns([1,1])
    with col_wx:
        if st.button("💾 Save Weekly Weights"):
            save_plan_allocations(user, wk_start, plan.get("allocations", {}), weights=new_weekly_w)
            st.success("Weekly weights saved.")
            st.rerun()
    with col_rw:
        if st.button("📤 Copy Weekly Weights to Registry Priorities (2↔=low/med/high)"):
            # map 1..10 -> 1..3 rough buckets
            for gid, w in new_weekly_w.items():
                to_registry = 1 if w <= 3 else (2 if w <= 7 else 3)
                set_goal_fields(user, gid, {"priority_weight": to_registry})
            st.success("Copied to registry priorities.")
            st.rerun()

    st.divider()

    # Allocation
    st.subheader("🧮 Allocate Weekly Pomodoros")
    wd, wec = week_day_counts(wk_start)
    cap_total = int(get_registry(user)["user_defaults"].get("weekday_poms", DEFAULTS["weekday_poms"])) * wd + \
                int(get_registry(user)["user_defaults"].get("weekend_poms", DEFAULTS["weekend_poms"])) * wec

    # Use per-week weights by default for auto allocation
    weight_map = {gid: int(new_weekly_w.get(gid, 5)) for gid in goals_dict.keys()}
    auto_alloc = proportional_allocation(cap_total, weight_map)

    current_alloc = plan.get("allocations", {}) or {}
    edited = {}
    cols2 = st.columns(min(4, max(1, len(weight_map))))
    i = 0
    for gid, _ in weight_map.items():
        with cols2[i % len(cols2)]:
            title = goals_dict.get(gid, {}).get("title", "(untitled)")
            default_val = int(current_alloc.get(gid, auto_alloc.get(gid, 0)))
            val = st.number_input(title, min_value=0, max_value=cap_total, value=default_val, step=1, key=f"a_{gid}")
            edited[gid] = int(val)
        i += 1

    sum_alloc = sum(edited.values())
    if sum_alloc != cap_total:
        st.warning(f"Allocations sum to {sum_alloc}, not {cap_total}.")
        if st.button("🔁 Normalize to capacity"):
            edited = proportional_allocation(cap_total, edited)
            # push to widgets
            for gid, v in edited.items():
                st.session_state[f"a_{gid}"] = v
            st.rerun()

    if st.button(("📌 Save Weekly Plan" if not current_alloc else "📌 Update Weekly Plan"), type="primary"):
        save_plan_allocations(user, wk_start, edited, weights=new_weekly_w)
        st.success("Weekly plan saved!")
        st.rerun()

    st.divider()
    # ----- ROLLOVER -----
    st.subheader("↪️ Rollover Unfinished from Previous Week")
    prev_start = wk_start - timedelta(days=7)
    prev_end = prev_start + timedelta(days=6)
    prev_plan = get_plan(user, prev_start, create_if_missing=False)
    if not prev_plan or not prev_plan.get("allocations"):
        st.info("No previous plan found.")
    else:
        prev_actual = actuals_by_gid(user, prev_start, prev_end)
        carry = {}
        for gid, p in (prev_plan.get("allocations") or {}).items():
            g = goals_dict.get(gid, {})
            status = (g.get("status") or "In Progress")
            if status in ["Completed", "Archived"]:
                continue  # do not rollover closed goals
            bal = max(0, int(p) - int(prev_actual.get(gid, 0)))
            if bal > 0:
                carry[gid] = bal

        if not carry:
            st.info("No unfinished items to rollover. Great job!")
        else:
            # Preview table
            rows = []
            for gid, bal in carry.items():
                rows.append({"Goal": goals_dict.get(gid, {}).get("title","(missing)"),
                             "Prev Planned": int(prev_plan["allocations"].get(gid,0)),
                             "Prev Actual": int(prev_actual.get(gid,0)),
                             "Carryover": int(bal)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            already = plan.get("rollover_from") == prev_start.isoformat()
            btn_lbl = "✅ Already Applied" if already else f"Apply Rollover from {prev_start} → {prev_end}"
            disabled = already
            if st.button(btn_lbl, disabled=disabled):
                merged = dict(current_alloc)
                for gid, add in carry.items():
                    merged[gid] = int(merged.get(gid, 0)) + int(add)
                save_plan_allocations(user, wk_start, merged, weights=new_weekly_w, meta={"rollover_from": prev_start.isoformat()})
                st.success("Rolled over unfinished poms into this week.")
                st.rerun()

    st.divider()
    # ----- THIS WEEK PLAN TABLE -----
    st.subheader("📋 This Week’s Plan")
    actual_now = actuals_by_gid(user, wk_start, wk_end)
    rows_active, rows_hold, rows_done = [], [], []
    for gid, planned in (get_plan(user, wk_start, True).get("allocations") or {}).items():
        g = goals_dict.get(gid, {}) or {}
        title = g.get("title","(missing)")
        status = g.get("status","In Progress")
        actual = int(actual_now.get(gid,0))
        remaining = max(0, int(planned) - actual)
        row = {"Goal": title, "Planned": int(planned), "Actual": actual, "Remaining": remaining, "Status": status}
        if status in ["On Hold"]:
            rows_hold.append(row)
        elif status in ["Completed","Archived"]:
            rows_done.append(row)
        else:
            rows_active.append(row)

    if rows_active:
        st.dataframe(pd.DataFrame(rows_active), use_container_width=True, hide_index=True)
    else:
        st.info("No active goals planned this week.")

    with st.expander("🟡 On-Hold (balance shown)"):
        if rows_hold:
            st.dataframe(pd.DataFrame(rows_hold), use_container_width=True, hide_index=True)
        else:
            st.caption("None.")

    with st.expander("✅ Completed (balance shown)"):
        if rows_done:
            st.dataframe(pd.DataFrame(rows_done), use_container_width=True, hide_index=True)
        else:
            st.caption("None.")

# ---------- Analytics ----------
def render_analytics_review(user: str):
    st.header("📊 Analytics")

    # Build a flat df
    df_all = flatten_user_days(user)
    if df_all.empty:
        st.info("No sessions yet. Start a Pomodoro to populate analytics.")
        return
    df_all["date_only"] = df_all["date_dt"].dt.date

    try:
        mode = st.segmented_control("Mode", options=["Week Review", "Trends"], default="Week Review", key="analytics_mode")
    except Exception:
        mode = st.radio("Mode", ["Week Review", "Trends"], horizontal=True, index=0)

    if mode == "Week Review":
        # Week selector + optional custom range
        pick_date = st.date_input("Review week of", value=st.session_state.review_week_date)
        wk_start, wk_end = week_bounds_ist(pick_date)
        custom = st.checkbox("Use custom range instead of calendar week", value=False)
        if custom:
            c1, c2 = st.columns(2)
            with c1:
                s = st.date_input("Start", value=wk_start)
            with c2:
                e = st.date_input("End", value=wk_end)
            if e < s:
                st.warning("End date < Start date. Adjusted to 7-day window.")
                e = s + timedelta(days=6)
            range_start, range_end = s, e
        else:
            range_start, range_end = wk_start, wk_end

        # Metrics
        mask = (df_all["date_only"] >= range_start) & (df_all["date_only"] <= range_end)
        dfw = df_all[mask & (df_all["t"] == "W")].copy()
        dfb = df_all[mask & (df_all["t"] == "B")].copy()

        # Planned vs actual (only for true week_start plans)
        plan = get_plan(user, wk_start, create_if_missing=False) or {}
        planned_alloc = plan.get("allocations", {}) or {}
        total_planned = int(sum(planned_alloc.values())) if (not custom) else 0

        work_goal = dfw[(pd.notna(dfw["gid"])) | (pd.notna(dfw["linked_gid"]))].copy()
        deep = int((dfw["dur"] >= 23).sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Plan Adherence", pct_or_dash(len(work_goal), total_planned))
        with c2:
            st.metric("Deep-work %", pct_or_dash(deep, len(dfw)))
        with c3:
            st.metric("Breaks Skipped", pct_or_dash(max(0, len(dfw) - len(dfb)), len(dfw)))
        with c4:
            # Weekly discipline score (for the calendar week)
            wds = weekly_discipline_score(user, wk_start)
            st.metric("Weekly Score", f"{wds['total']}/100")

        st.divider()
        st.subheader("📈 Daily Discipline Score")
        days = pd.date_range(start=range_start, end=range_end)
        day_scores = []
        for ts in days:
            d = ts.date()
            ds = daily_discipline_score(user, d)
            day_scores.append({"Day": ts.strftime("%a %d"), "Score": ds["total"]})
        if day_scores:
            df_scores = pd.DataFrame(day_scores)
            chart = alt.Chart(df_scores).mark_line(point=True).encode(
                x=alt.X('Day', sort=None),
                y=alt.Y('Score', scale=alt.Scale(domain=[0,100]))
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)

        st.subheader("Planned vs Actual (per goal)")
        if planned_alloc and not custom:
            reg = get_registry(user)
            titles = {gid: reg["goals"].get(gid, {}).get("title", "(missing)") for gid in planned_alloc.keys()}
            act = {}
            for _, row in work_goal.iterrows():
                g = row["gid"] if pd.notna(row["gid"]) else row["linked_gid"]
                act[g] = act.get(g, 0) + 1
            rows = []
            for gid, p in planned_alloc.items():
                rows.append({"Goal": titles.get(gid, "(missing)"), "Planned": int(p), "Actual": int(act.get(gid, 0))})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No planned allocations for this selection.")

        st.divider()
        # Month overview: weekly scores across the month of pick_date
        st.subheader("🗓️ Month Overview — Weekly Scores")
        month_start = date(pick_date.year, pick_date.month, 1)
        next_month = month_start.replace(day=28) + timedelta(days=4)
        month_end = (next_month - timedelta(days=next_month.day)).replace(day=(next_month - timedelta(days=next_month.day)).day)
        # enumerate week starts in month
        ws = week_bounds_ist(month_start)[0]
        week_labels, week_scores = [], []
        while ws <= month_end:
            wd = weekly_discipline_score(user, ws)
            week_labels.append(f"{ws.strftime('%d %b')}")
            week_scores.append(wd["total"])
            ws = ws + timedelta(days=7)
        if week_scores:
            dfm = pd.DataFrame({"Week": week_labels, "Score": week_scores})
            ch = alt.Chart(dfm).mark_bar().encode(x='Week', y=alt.Y('Score', scale=alt.Scale(domain=[0,100])))
            st.altair_chart(ch, use_container_width=True)
            st.metric("Month Avg Score", f"{(sum(week_scores)/len(week_scores)):.1f}/100")

    else:
        # Trends
        today = now_ist().date()
        st.subheader("Overview")
        opt = st.selectbox("Time Period", ["Last 7 days", "Last 30 days", "Custom"], index=1)
        if opt == "Last 7 days":
            start_d = today - timedelta(days=6)
            end_d = today
        elif opt == "Last 30 days":
            start_d = today - timedelta(days=29)
            end_d = today
        else:
            c1, c2 = st.columns(2)
            with c1: start_d = st.date_input("Start", value=today - timedelta(days=29))
            with c2: end_d = st.date_input("End", value=today)
            if end_d < start_d:
                end_d = start_d
        mask = (df_all["date_only"] >= start_d) & (df_all["date_only"] <= end_d)
        dfw = df_all[mask & (df_all["t"] == "W")].copy()

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
        st.subheader("🧩 Category Mix")
        by_cat = dfw.groupby("cat")["dur"].sum().reset_index().rename(columns={"dur":"Minutes"})
        by_cat = by_cat[by_cat["cat"].fillna("").astype(str) != ""]
        if by_cat.empty:
            st.info("No categories found in this period.")
        else:
            # Pie chart via Altair
            pie = alt.Chart(by_cat).mark_arc().encode(
                theta=alt.Theta(field="Minutes", type="quantitative"),
                color=alt.Color(field="cat", type="nominal", legend=alt.Legend(title="Category")),
                tooltip=["cat","Minutes"]
            ).properties(height=320)
            st.altair_chart(pie, use_container_width=True)

        st.subheader("📌 Task Performance (Top 12)")
        tstats = dfw.groupby(["cat","task"]).agg(minutes=("dur","sum"), sessions=("dur","count")).reset_index()
        if tstats.empty:
            st.info("No tasks logged.")
        else:
            top = tstats.sort_values("minutes", ascending=False).head(12)
            bar = alt.Chart(top).mark_bar().encode(
                x=alt.X("minutes:Q", title="Minutes"),
                y=alt.Y("task:N", sort='-x', title="Task"),
                color=alt.Color("cat:N", legend=alt.Legend(title="Category")),
                tooltip=["cat","task","minutes","sessions"]
            ).properties(height=320)
            st.altair_chart(bar, use_container_width=True)

# ---------- Journal ----------
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
            aligned = st.selectbox("Aligned with weekly plan?", ["Yes","Partly","No"],
                                   index=["Yes","Partly","No"].index(ref.get("aligned","Yes")) if ref.get("aligned") in ["Yes","Partly","No"] else 0)
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
        pages = ["🎯 Focus Timer", "📅 Weekly Planner", "📊 Analytics", "🧾 Journal"]
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
    elif page == "📊 Analytics":
        render_analytics_review(st.session_state.user)
    elif page == "🧾 Journal":
        render_journal(st.session_state.user)

if __name__ == "__main__":
    main_header_and_router()
