# scripts/maintenance.py
from datetime import datetime, timedelta
import re
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://prashanth01071995:pradsml%402025@cluster0.fsbic.mongodb.net/"
DB_NAME = "time_tracker_db"

DEFAULTS = {
    "weekday_poms": 3,
    "weekend_poms": 5,
    "auto_break": True,
    "custom_categories": ["Learning","Projects","Research","Planning"],
}

def scrub_bad_id_indexes(col):
    try:
        for ix in col.list_indexes():
            name = ix.get("name","")
            key  = ix.get("key",{})
            if key == {"_id": 1} and name != "_id_":
                try: col.drop_index(name)
                except Exception: pass
    except Exception:
        pass

def recompute_day(doc):
    sessions = doc.get("sessions", [])
    work = [s for s in sessions if s.get("t") == "W"]
    brk  = [s for s in sessions if s.get("t") == "B"]
    deep = [s for s in work if int(s.get("dur",0)) >= 23]

    by_cat = {}
    for s in work:
        cat = s.get("cat","")
        if cat:
            by_cat[cat] = by_cat.get(cat, 0) + int(s.get("dur",0))

    start_mins = []
    for s in sessions:
        t = s.get("time","")
        m = re.match(r"^(\d{1,2}):(\d{2})\s?(AM|PM)$", t or "", flags=re.I)
        if m:
            hh = int(m.group(1)) % 12
            mm = int(m.group(2))
            if m.group(3).upper() == "PM":
                hh += 12
            start_mins.append(hh*60 + mm)

    switches = 0
    prev = None
    for s in sessions:
        key = s.get("gid") or s.get("linked_gid") or (f"CAT::{s.get('cat','')}" if s.get("cat") else "NA")
        if prev is not None and key != prev:
            switches += 1
        prev = key

    return {
        "totals": {
            "work_sessions": len(work),
            "work_minutes": sum(int(s.get("dur",0)) for s in work),
            "break_sessions": len(brk),
            "break_minutes": sum(int(s.get("dur",0)) for s in brk),
            "deep_work_sessions": len(deep)
        },
        "by_category_minutes": by_cat,
        "start_time_mins": start_mins,
        "switches": switches,
        "updated_at": datetime.utcnow()
    }

if __name__ == "__main__":
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    user_days = db["user_days"]
    weekly = db["weekly_plans"]

    # 1) Scrub bad _id indexes
    for col in (user_days, weekly):
        scrub_bad_id_indexes(col)

    # 2) Normalize user_days
    for d in user_days.find({}):
        changed = False
        if d.get("schema_version") != 2:
            d["schema_version"] = 2; changed = True
        if not isinstance(d.get("sessions"), list):
            d["sessions"] = []; changed = True

        new_sessions = []
        for s in d["sessions"]:
            out = dict(s)
            out["t"] = "B" if s.get("t") == "B" else "W"
            out["dur"] = int(s.get("dur", 0))
            if not isinstance(s.get("time"), str):
                out["time"] = ""
            if out["t"] == "W":
                has_gid = bool(s.get("gid") or s.get("linked_gid"))
                if not has_gid:
                    out["unplanned"] = True
                    out["source"] = "custom"
                    out["cat"] = s.get("cat","")
                    out["task"] = s.get("task","")
                    if s.get("reason"): out["reason"] = s["reason"]
                    if s.get("note"): out["note"] = s["note"]
                else:
                    out["source"] = "plan"
            new_sessions.append(out)
        d["sessions"] = new_sessions
        d.update(recompute_day(d))
        if changed:
            user_days.replace_one({"_id": d["_id"]}, d)
        else:
            user_days.update_one({"_id": d["_id"]}, {"$set": {
                "sessions": d["sessions"],
                **recompute_day(d)
            }})

    # 3) Normalize weekly_plans (registry)
    for r in weekly.find({ "_id": { "$regex": "\\|registry$" } }):
        set_doc = {"updated_at": datetime.utcnow()}
        if r.get("type") != "registry":
            set_doc["type"] = "registry"
        if r.get("schema_version") != 2:
            set_doc["schema_version"] = 2
        if not r.get("user_defaults"):
            set_doc["user_defaults"] = DEFAULTS
        goals = r.get("goals") or {}
        changed = False
        for gid, g in goals.items():
            pw = max(1, min(3, int(g.get("priority_weight", 2))))
            tp = int(g.get("target_poms", 0))
            if pw != g.get("priority_weight") or tp != g.get("target_poms"):
                goals[gid]["priority_weight"] = pw
                goals[gid]["target_poms"] = tp
                goals[gid]["updated_at"] = datetime.utcnow()
                changed = True
        if changed:
            set_doc["goals"] = goals
        if len(set_doc) > 1:
            weekly.update_one({"_id": r["_id"]}, {"$set": set_doc})

    # 4) Normalize weekly_plans (plan docs)
    for p in weekly.find({ "type": "plan" }):
        set_doc = {"updated_at": datetime.utcnow(), "schema_version": 2}
        if p.get("week_start"):
            # recompute week_end = week_start + 6 days
            ws = p["week_start"]
            from datetime import date
            y,m,d = map(int, ws.split("-"))
            wk_start = date(y,m,d)
            wk_end = wk_start + timedelta(days=6)
            if p.get("week_end") != wk_end.isoformat():
                set_doc["week_end"] = wk_end.isoformat()

        allocations = p.get("allocations") or {}
        goals = list(allocations.keys())
        if goals != (p.get("goals") or []):
            set_doc["goals"] = goals

        reg = weekly.find_one({"_id": p["user"] + "|registry"}) or {"goals":{},"user_defaults":DEFAULTS}
        embedded = []
        for gid in goals:
            g = (reg.get("goals") or {}).get(gid) or {}
            embedded.append({
                "goal_id": gid,
                "title": g.get("title","(missing)"),
                "priority_weight": int(g.get("priority_weight",2)),
                "status_at_plan": g.get("status","In Progress"),
                "planned": int(allocations.get(gid,0)),
                "carryover_in": 0,
                "carryover_out": 0,
            })
        set_doc["goals_embedded"] = embedded
        if not p.get("capacity") or "total" not in p["capacity"]:
            set_doc["capacity"] = {
                "weekday": reg.get("user_defaults",{}).get("weekday_poms", DEFAULTS["weekday_poms"]),
                "weekend": reg.get("user_defaults",{}).get("weekend_poms", DEFAULTS["weekend_poms"]),
                "total": sum(int(allocations.get(gid,0)) for gid in goals)
            }
        weekly.update_one({"_id": p["_id"]}, {"$set": set_doc})

    print("✅ Maintenance complete")
