# services/planner_service.py
from typing import Any, Dict, List, Tuple
import numpy as np
from data_access.users_repo import get_user
from data_access.goals_repo import get_goals
from data_access.sessions_repo import aggregate_pe_by_goal_bucket
from core.time_utils import monday_from_week_key, prev_week_key as _prev_week_key

def get_user_capacity_defaults(uid: str) -> Tuple[int, int]:
    u = get_user(uid) or {}
    prefs = (u.get("prefs") or {})
    wkday_default = int(prefs.get("weekday_poms", 3))
    wkend_default = int(prefs.get("weekend_poms", 6))
    return wkday_default, wkend_default

def get_rank_weight_map(uid: str) -> Dict[str, int]:
    u = get_user(uid) or {}
    rwm = ((u.get("prefs") or {}).get("rank_weight_map") or {"1":5,"2":3,"3":2,"4":1,"5":1})
    return {str(k): int(v) for k, v in rwm.items()}

def derive_auto_plan_from_active(uid: str, week_key: str) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    wkday, wkend = get_user_capacity_defaults(uid)
    total_capacity = wkday * 5 + wkend * 2
    rwm = get_rank_weight_map(uid)
    goals = [g for g in get_goals(uid) if g.get("status") == "In Progress"]
    if not goals or total_capacity <= 0:
        return ({"weekday": wkday, "weekend": wkend, "total": total_capacity}, [])
    def w_for(g):
        rank = int(g.get("priority", 3))
        return int(rwm.get(str(rank), 1))
    weights = [w_for(g) for g in goals]
    wsum = sum(weights) or 1
    shares = [(w / wsum) * total_capacity for w in weights]
    base = [int(np.floor(s)) for s in shares]
    left = total_capacity - sum(base)
    frac = [s - b for s, b in zip(shares, base)]
    order = np.argsort(-np.array(frac))
    for i in range(int(left)):
        base[order[i]] += 1
    items = []
    for g, pc in zip(goals, base):
        rank = int(g.get("priority", 3))
        items.append({
            "goal_id": g["_id"], "priority_rank": rank,
            "weight": int(rwm.get(str(rank), 1)),
            "planned_current": int(pc),
            "backlog_in": 0,
            "total_target": int(pc),
            "status_at_plan": "In Progress",
            "close_action": None,
            "notes": None
        })
    return ({"weekday": wkday, "weekend": wkend, "total": total_capacity}, items)

def week_dates_list(week_key: str):
    mon = monday_from_week_key(week_key).date()
    return [(mon + np.timedelta64(i, 'D')).astype('datetime64[D]').astype(object).isoformat() for i in range(7)]

def determine_alloc_bucket(uid: str, week_key: str, goal_id: str, planned_current: int) -> str:
    pe_map = aggregate_pe_by_goal_bucket(uid, week_key)
    done_current = pe_map.get(goal_id, {}).get("current", 0.0)
    return "current" if done_current + 1e-6 < float(planned_current) else "backlog"

def prev_week_key(week_key: str) -> str:
    return _prev_week_key(week_key)
