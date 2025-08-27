#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DB Health Checker for Time Tracker
Run:
  python check_db_health.py --uri "mongodb+srv://..." [--db time_tracker_db] [--fix] [--drop-stray-indexes]
"""

import argparse
import re
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any
from pymongo import MongoClient
from pymongo.errors import OperationFailure

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^(0?[1-9]|1[0-2]):[0-5]\d [AP]M$")  # 12h "HH:MM AM/PM"

# Accept either the canonical name or the legacy alias
EXPECTED_INDEXES = {
    "user_days": [
        ({"user": 1, "date": 1}, "user_date", []),
        ({"sessions.gid": 1}, "sessions_gid", []),
        ({"sessions.linked_gid": 1}, "sessions_linked_gid", []),
        ({"sessions.unplanned": 1}, "sessions_unplanned", []),
        ({"sessions.cat": 1}, "sessions_cat", []),
    ],
    "weekly_plans": [
        # keys, preferred_name, allowed_alias_names
        ({"user": 1, "type": 1}, "user_type", ["type_user"]),
        ({"user": 1, "week_start": 1}, "user_week", ["user_weekstart"]),
    ],
}


def week_bounds(d: date) -> Tuple[date, date]:
    start = d - timedelta(days=d.weekday())  # Monday
    end = start + timedelta(days=6)          # Sunday
    return start, end

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--uri", required=True, help="MongoDB connection URI")
    p.add_argument("--db", default="time_tracker_db", help="Database name")
    p.add_argument("--fix", action="store_true", help="Apply safe fixes (schema_version, type, week_end, capacity)")
    p.add_argument("--drop-stray-indexes", action="store_true", help="Drop unknown custom indexes (never _id_)")
    return p.parse_args()

def keydict(ixkey) -> Dict[str, int]:
    # ixkey is a SON like [("field", 1), ...] or dict-like; normalize
    if isinstance(ixkey, dict):
        return dict(ixkey)
    try:
        return {k: v for k, v in ixkey}
    except Exception:
        return {}
    
def _has_index_by_keys(col, keys_dict: dict) -> bool:
    try:
        for ix in col.list_indexes():
            if dict(ix.get("key", {})) == keys_dict:
                return True
    except Exception:
        pass
    return False

def ensure_expected_indexes(col, expected_defs, create=False):
    """
    Inspect current indexes. Consider index 'present' if KEYS match, even if name differs (alias).
    Optionally create missing ones using the preferred name.
    Returns a summary with present/missing/stray (by name for visibility).
    """
    current = list(col.list_indexes())
    cur_names = [ix.get("name") for ix in current]
    cur_keys_list = [dict(ix.get("key", {})) for ix in current]

    present = []
    missing = []
    stray = []

    # Build allowlist of names (preferred + aliases) + default _id_
    allowed_names = {"_id_"}
    for keys, pref_name, aliases in expected_defs:
        allowed_names.add(pref_name)
        for a in aliases:
            allowed_names.add(a)

    # Detect present/missing by KEYS
    for keys, pref_name, aliases in expected_defs:
        if any(keys == k for k in cur_keys_list):
            present.append(pref_name)  # mark present by preferred label
        else:
            missing.append(pref_name)
            if create:
                try:
                    # Create with preferred name only if keys truly missing
                    col.create_index(list(keys.items()), name=pref_name)
                except Exception as e:
                    print(f"  ⚠️  Could not create index {pref_name} on {col.name}: {e}")

    # Stray = names not in allowlist and not the default _id_
    for ix in current:
        n = ix.get("name")
        if n not in allowed_names and dict(ix.get("key", {})) != {"_id": 1}:
            stray.append(n)

    return {"present": present, "missing": missing, "stray": stray}

def drop_stray_indexes(col, stray_names: List[str]):
    dropped = 0
    for n in stray_names:
        try:
            col.drop_index(n)
            dropped += 1
        except OperationFailure as e:
            print(f"  ⚠️  Could not drop index {n} on {col.name}: {e}")
        except Exception as e:
            print(f"  ⚠️  Could not drop index {n} on {col.name}: {e}")
    return dropped

def validate_user_days(col, fix=False) -> Dict[str, int]:
    counts = {
        "docs": 0,
        "bad_date_format": 0,
        "missing_user": 0,
        "missing_sessions": 0,
        "bad_t": 0,
        "bad_dur": 0,
        "bad_time": 0,
        "set_schema_v2": 0,
    }
    cursor = col.find({}, {"_id": 1, "user": 1, "date": 1, "sessions": 1, "schema_version": 1})
    for d in cursor:
        counts["docs"] += 1
        did = d.get("_id")
        user = d.get("user")
        date_str = d.get("date")
        sessions = d.get("sessions", [])

        if not user:
            counts["missing_user"] += 1
        if not isinstance(sessions, list):
            counts["missing_sessions"] += 1
        if not (isinstance(date_str, str) and ISO_DATE_RE.match(date_str)):
            counts["bad_date_format"] += 1

        # schema_version
        if d.get("schema_version") != 2 and fix:
            col.update_one({"_id": did}, {"$set": {"schema_version": 2, "updated_at": datetime.utcnow()}})
            counts["set_schema_v2"] += 1

        # session validations
        if isinstance(sessions, list):
            for s in sessions:
                t = s.get("t")
                dur = s.get("dur")
                tm = s.get("time")
                if t not in ("W", "B"):
                    counts["bad_t"] += 1
                try:
                    dur_i = int(dur)
                    if dur_i <= 0 or dur_i > 60:  # allow up to 60 to be lenient
                        counts["bad_dur"] += 1
                except Exception:
                    counts["bad_dur"] += 1
                if tm and not TIME_RE.match(str(tm)):
                    counts["bad_time"] += 1
    return counts

def validate_weekly_plans(col, fix=False) -> Dict[str, int]:
    counts = {
        "docs": 0,
        "registries": 0,
        "plans": 0,
        "bad_week_dates": 0,
        "fixed_week_end": 0,
        "capacity_mismatch": 0,
        "fixed_capacity": 0,
        "alloc_over_capacity": 0,
        "set_schema_v2": 0,
        "missing_type": 0,
    }
    cursor = col.find({})
    for d in cursor:
        counts["docs"] += 1
        did = d.get("_id")
        typ = d.get("type")
        if typ == "registry":
            counts["registries"] += 1
        elif typ == "plan":
            counts["plans"] += 1
        else:
            counts["missing_type"] += 1
            if fix:
                # Decide based on _id pattern: user|YYYY-MM-DD likely plan, else registry
                if isinstance(did, str) and re.search(r"\|\d{4}-\d{2}-\d{2}$", did):
                    col.update_one({"_id": did}, {"$set": {"type": "plan", "updated_at": datetime.utcnow()}})
                else:
                    col.update_one({"_id": did}, {"$set": {"type": "registry", "updated_at": datetime.utcnow()}})

        # schema_version
        if d.get("schema_version") != 2 and fix:
            col.update_one({"_id": did}, {"$set": {"schema_version": 2, "updated_at": datetime.utcnow()}})
            counts["set_schema_v2"] += 1

        if typ == "plan":
            ws = d.get("week_start")
            we = d.get("week_end")
            cap = (d.get("capacity") or {})
            weekday = int(cap.get("weekday", 0))
            weekend = int(cap.get("weekend", 0))
            total = int(cap.get("total", 0))

            # Week date validity
            ok_dates = True
            if not (isinstance(ws, str) and ISO_DATE_RE.match(ws) and isinstance(we, str) and ISO_DATE_RE.match(we)):
                ok_dates = False
            else:
                try:
                    ws_d = date.fromisoformat(ws)
                    we_d = date.fromisoformat(we)
                    ws_calc, we_calc = week_bounds(ws_d)
                    if we_d != we_calc:
                        counts["bad_week_dates"] += 1
                        ok_dates = False
                        if fix:
                            col.update_one({"_id": did}, {"$set": {"week_end": we_calc.isoformat(), "updated_at": datetime.utcnow()}})
                            counts["fixed_week_end"] += 1
                except Exception:
                    ok_dates = False
                    counts["bad_week_dates"] += 1

            # Capacity math: total == weekday*wdays + weekend*wendays
            if ok_dates:
                wd_cnt = 5
                we_cnt = 2
                expected_total = weekday * wd_cnt + weekend * we_cnt
                if total != expected_total:
                    counts["capacity_mismatch"] += 1
                    if fix:
                        col.update_one({"_id": did}, {"$set": {"capacity.total": int(expected_total), "updated_at": datetime.utcnow()}})
                        counts["fixed_capacity"] += 1

            # Allocations don’t exceed capacity total
            alloc = (d.get("allocations") or {})
            try:
                alloc_sum = sum(int(v) for v in alloc.values())
            except Exception:
                alloc_sum = 0
            if total and alloc_sum > total:
                counts["alloc_over_capacity"] += 1
    return counts

def collect_registry_goals(col) -> Dict[str, set]:
    """
    Returns mapping user -> set(goal_ids) from registry docs.
    """
    mapping: Dict[str, set] = {}
    for d in col.find({"type": "registry"}, {"user": 1, "goals": 1}):
        user = d.get("user")
        if not user:
            continue
        goals = d.get("goals") or {}
        mapping[user] = set(goals.keys())
    return mapping

def validate_referential(user_days_col, weekly_plans_col) -> Dict[str, int]:
    """
    Verify that any session with gid/linked_gid points to an existing registry goal for that user.
    """
    counts = {"sessions_with_gid": 0, "missing_goal_refs": 0}
    reg = collect_registry_goals(weekly_plans_col)

    cursor = user_days_col.find({}, {"user": 1, "sessions": 1})
    for d in cursor:
        user = d.get("user")
        goals = reg.get(user, set())
        sessions = d.get("sessions", [])
        for s in sessions if isinstance(sessions, list) else []:
            gid = s.get("gid") or s.get("linked_gid")
            if gid:
                counts["sessions_with_gid"] += 1
                if gid not in goals:
                    counts["missing_goal_refs"] += 1
    return counts

def main():
    args = parse_args()
    client = MongoClient(args.uri)
    db = client[args.db]
    user_days = db["user_days"]
    weekly_plans = db["weekly_plans"]

    print("🔎 Index audit")
    ud_ix = ensure_expected_indexes(user_days, EXPECTED_INDEXES["user_days"], create=False)
    wp_ix = ensure_expected_indexes(weekly_plans, EXPECTED_INDEXES["weekly_plans"], create=False)

    print(f"  user_days: present={ud_ix['present']}, missing={ud_ix['missing']}, stray={ud_ix['stray']}")
    print(f"  weekly_plans: present={wp_ix['present']}, missing={wp_ix['missing']}, stray={wp_ix['stray']}")

    if args.drop_stray_indexes:
        if ud_ix["stray"]:
            n = drop_stray_indexes(user_days, ud_ix["stray"])
            print(f"  ✅ Dropped {n} stray indexes from user_days")
        if wp_ix["stray"]:
            n = drop_stray_indexes(weekly_plans, wp_ix["stray"])
            print(f"  ✅ Dropped {n} stray indexes from weekly_plans")

    # Offer to create missing expected indexes (safe)
    if ud_ix["missing"] or wp_ix["missing"]:
        print("  ℹ️ Creating any missing expected indexes…")
        ensure_expected_indexes(user_days, EXPECTED_INDEXES["user_days"], create=True)
        ensure_expected_indexes(weekly_plans, EXPECTED_INDEXES["weekly_plans"], create=True)

    print("\n🧪 Document validation — user_days")
    ud_counts = validate_user_days(user_days, fix=args.fix)
    for k, v in ud_counts.items():
        print(f"  {k}: {v}")

    print("\n🧪 Document validation — weekly_plans")
    wp_counts = validate_weekly_plans(weekly_plans, fix=args.fix)
    for k, v in wp_counts.items():
        print(f"  {k}: {v}")

    print("\n🔗 Referential integrity")
    ref_counts = validate_referential(user_days, weekly_plans)
    for k, v in ref_counts.items():
        print(f"  {k}: {v}")

    print("\n✨ Done.")

if __name__ == "__main__":
    main()
