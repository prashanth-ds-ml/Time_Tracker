# services/sessions_service.py
from typing import Any, Dict, Optional
from pymongo.errors import WriteError
import streamlit as st
from core.db import get_db, USER_ID
from core.time_utils import (
    IST, utc_now_naive, _to_utc_naive, pom_equiv, week_key_from_datestr
)
from core.constants import ALLOWED_ACTIVITY_TYPES, ALLOWED_BUCKETS

db = get_db()

def insert_session(
    user_id: str,
    t: str,                 # "W" or "B"
    dur_min: int,
    ended_at_ist,
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
    t = "B" if str(t).upper() == "B" else "W"
    dur_min = max(1, int(dur_min))
    pe = pom_equiv(dur_min)

    if ended_at_ist.tzinfo is None:
        ended_at_ist = IST.localize(ended_at_ist)
    started_at_ist = ended_at_ist - __import__("datetime").timedelta(minutes=dur_min)

    started_utc = _to_utc_naive(started_at_ist)
    ended_utc   = _to_utc_naive(ended_at_ist)

    date_ist = started_at_ist.astimezone(IST).date().isoformat()
    week_key = week_key_from_datestr(date_ist)

    kind = (kind or None)
    if kind is not None:
        kind = str(kind).strip().lower()
        if kind not in {"focus", "activity"}:
            kind = "focus" if t == "W" else None

    if kind == "activity":
        if not activity_type or str(activity_type).strip().lower() not in ALLOWED_ACTIVITY_TYPES:
            activity_type = "other"
        else:
            activity_type = str(activity_type).strip().lower()
        goal_mode = None
        goal_id = None
        alloc_bucket = None
        deep_work = None
        cat = cat or "Wellbeing"
    else:
        activity_type = None

    if kind == "focus" and goal_mode == "weekly" and goal_id:
        if alloc_bucket:
            alloc_bucket = str(alloc_bucket).strip().lower()
            if alloc_bucket not in ALLOWED_BUCKETS:
                alloc_bucket = None
    else:
        goal_mode = ("custom" if (kind == "focus" and t == "W" and not goal_id) else goal_mode)
        alloc_bucket = None

    safe_pc = None
    if isinstance(post_checkin, dict):
        pc = {}
        q = post_checkin.get("quality_1to5")
        m = post_checkin.get("mood_1to5")
        e = post_checkin.get("energy_1to5")
        if isinstance(q, int) and 1 <= q <= 5: pc["quality_1to5"] = q
        if isinstance(m, int) and 1 <= m <= 5: pc["mood_1to5"] = m
        if isinstance(e, int) and 1 <= e <= 5: pc["energy_1to5"] = e
        if post_checkin.get("distraction") is not None:
            pc["distraction"] = str(post_checkin.get("distraction"))
        if post_checkin.get("note") is not None:
            pc["note"] = str(post_checkin.get("note"))
        if pc:
            safe_pc = pc

    sid = f"{user_id}|{date_ist}|{t}|{int(started_at_ist.timestamp())}|{dur_min}"
    now = utc_now_naive()

    full_doc = {
        "_id": sid,
        "user": user_id,
        "date_ist": date_ist,
        "week_key": week_key,
        "t": t,
        "dur_min": int(dur_min),
        "pom_equiv": float(pe),
        "started_at_ist": started_utc,
        "ended_at_ist": ended_utc,
        "schema_version": 1,
        "kind": kind,
        "activity_type": activity_type,
        "intensity": (str(intensity).strip().lower() if intensity else None),
        "deep_work": (bool(deep_work) if isinstance(deep_work, bool) else None),
        "context_switch": False,
        "goal_mode": (str(goal_mode).strip().lower() if goal_mode else None),
        "goal_id": goal_id,
        "task": (str(task) if task else None),
        "cat": (str(cat) if cat else None),
        "alloc_bucket": alloc_bucket,
        "break_autostart": (bool(break_autostart) if isinstance(break_autostart, bool) else None),
        "skipped": (bool(skipped) if isinstance(skipped, bool) else None),
        "post_checkin": safe_pc,
        "device": (str(device) if device else None),
        "created_at": now,
        "updated_at": now,
    }
    full_doc = {k: v for k, v in full_doc.items() if v is not None}
    doc_on_insert = dict(full_doc); doc_on_insert.pop("updated_at", None)

    try:
        db.sessions.update_one(
            {"_id": sid},
            {"$setOnInsert": doc_on_insert, "$set": {"updated_at": now}},
            upsert=True
        )
    except WriteError as e:
        details = getattr(e, "details", None) or {}
        err = details.get("errmsg") or str(e)
        reasons = details.get("errInfo") or {}
        st.error("❌ Failed to write session (schema validation).")
        with st.expander("Debug details (validator)"):
            st.write({"error": err, "reasons": reasons, "doc_keys": list(full_doc.keys())})
        raise

    st.cache_data.clear()
    return sid

def update_session_post_checkin(sid: str, payload: Dict[str, Any]):
    db.sessions.update_one({"_id": sid, "user": USER_ID},
                           {"$set": {"post_checkin": payload, "updated_at": utc_now_naive()}})
    st.cache_data.clear()
