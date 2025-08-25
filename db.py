# db.py — thin adapter over the new Mongo shape
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import hashlib
import pandas as pd
from pymongo import MongoClient
import pytz

IST = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(IST)

def week_bounds(d: date) -> Tuple[date, date]:
    start = d - timedelta(days=d.weekday())
    return start, start + timedelta(days=6)

# ---- Mongo init --------------------------------------------------------------

def get_db():
    import streamlit as st
    client = MongoClient(st.secrets["mongo_uri"])
    return client["time_tracker_db"]

def ensure_indexes():
    db = get_db()
    # user_days
    db.user_days.create_index([("user", 1), ("date", 1)], name="user_date")
    # weekly_plans
    db.weekly_plans.create_index([("user", 1), ("type", 1)], name="user_type")
    db.weekly_plans.create_index([("user", 1), ("week_start", 1)], name="user_weekstart")

# ---- Registries (per-user settings + goals) ---------------------------------

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
            "custom_categories": ["Learning", "Projects", "Research", "Planning"]
        },
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    db.weekly_plans.insert_one(doc)
    return doc

def list_registry_goals(user: str, statuses: Optional[List[str]] = None) -> List[Dict]:
    reg = get_or_create_registry(user)
    goals = reg.get("goals", [])
    if statuses:
        goals = [g for g in goals if g.get("status") in statuses]
    return goals

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
        if g["goal_id"] == goal_id or g["title"] == title:
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

def registry_defaults(user: str) -> Dict:
    reg = get_or_create_registry(user)
    return reg.get("defaults", {})

def update_registry_defaults(user: str, weekday_poms: int, weekend_poms: int,
                             auto_break: Optional[bool] = None,
                             custom_categories: Optional[List[str]] = None):
    db = get_db()
    reg = get_or_create_registry(user)
    defaults = reg.get("defaults", {})
    defaults["weekday_poms"] = int(weekday_poms)
    defaults["weekend_poms"] = int(weekend_poms)
    if auto_break is not None:
        defaults["auto_break"] = bool(auto_break)
    if custom_categories is not None:
        defaults["custom_categories"] = list(custom_categories)
    db.weekly_plans.update_one({"_id": reg["_id"]},
        {"$set": {"defaults": defaults, "updated_at": datetime.utcnow()}}
    )

def goal_title_map(user: str) -> Dict[str, str]:
    return {g["goal_id"]: g["title"] for g in list_registry_goals(user)}

# ---- Plans -------------------------------------------------------------------

def _capacity_from_defaults(ws: date, defs: Dict) -> int:
    weekday_poms = int(defs.get("weekday_poms", 3))
    weekend_poms = int(defs.get("weekend_poms", 5))
    wd = sum(1 for i in range(7) if (ws + timedelta(days=i)).weekday() < 5)
    we = 7 - wd
    return weekday_poms * wd + weekend_poms * we

def get_or_create_week_plan(user: str, d: Optional[date] = None) -> Dict:
    db = get_db()
    if d is None:
        d = now_ist().date()
    ws, we = week_bounds(d)
    pid = f"{user}|{ws.isoformat()}"
    plan = db.weekly_plans.find_one({"_id": pid})
    if plan:
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
        "goals_embedded": [],           # snapshot of goal meta
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
        "allocations_by_goal": {k: int(v) for k, v in allocations_by_goal.items()},
        "goals_embedded": embedded,
        "updated_at": datetime.utcnow()
    }})

# ---- Days & Sessions ---------------------------------------------------------

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

    db.user_days.update_one({"_id": day["_id"]}, {"$set": {
        "updated_at": datetime.utcnow(),
        "totals": totals
    }, "$push": {"sessions": sess}})

def get_sessions_df(user: str) -> pd.DataFrame:
    db = get_db()
    cursor = db.user_days.find({"user": user}, {"_id":1,"date":1,"sessions":1,"user":1})
    records = []
    for d in cursor:
        iso = d.get("date")
        u = d.get("user", user)
        for s in d.get("sessions", []):
            records.append({
                "date": iso,
                "time": s.get("time"),
                "pomodoro_type": "Break" if s.get("t")=="Break" else "Work",
                "duration": int(s.get("minutes", 0)),
                "user": u,
                "goal_id": s.get("goal_id"),
                "task": s.get("task",""),
                "category": s.get("category",""),
            })
    if not records:
        df = pd.DataFrame(columns=["date","time","pomodoro_type","duration","user","goal_id","task","category"])
        df["date"] = pd.to_datetime(df["date"])
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
        return df
    df = pd.DataFrame.from_records(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
    for c in ["time","pomodoro_type","user","goal_id","task","category"]:
        if c not in df.columns:
            df[c] = "" if c!="goal_id" else None
    return df

# ---- Daily target / notes / reflection --------------------------------------

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
        {"$set": {
            "reflection": {"aligned": aligned, "focus_rating": int(rating),
                           "blockers": blockers, "notes": notes},
            "updated_at": datetime.utcnow()
        }}
    )

def add_note(user: str, content: str, iso_date: Optional[str] = None):
    db = get_db()
    day = get_or_create_user_day(user, iso_date)
    db.user_days.update_one({"_id": day["_id"]},
        {"$push": {"notes": {"content": content, "ts": datetime.utcnow()}},
         "$set": {"updated_at": datetime.utcnow()}}
    )

# ---- Users list --------------------------------------------------------------

def list_users() -> List[str]:
    db = get_db()
    users = set()
    for u in db.weekly_plans.find({"type":"registry"}, {"user":1}):
        users.add(u["user"])
    # fallback: user_days
    for u in db.user_days.distinct("user"):
        users.add(u)
    return sorted(users)

def add_user(username: str):
    get_or_create_registry(username)
