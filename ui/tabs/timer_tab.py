# ui/tabs/timer_tab.py
import time
import pandas as pd
import streamlit as st
from datetime import timedelta

from core.time_utils import now_ist, today_iso, to_ist_display
from core.constants import ALLOWED_ACTIVITY_TYPES
from data_access.sessions_repo import total_day_pe, list_today_sessions, delete_last_today_session
from data_access.plans_repo import get_week_plan
from data_access.goals_repo import get_goals_map
from services.planner_service import determine_alloc_bucket
from services.sessions_service import insert_session, update_session_post_checkin
from ui.components.sound import play_finish_sound

def _should_render_heavy() -> bool:
    running = bool(st.session_state.get("timer", {}).get("running"))
    if not running:
        return True
    now = time.time()
    last = st.session_state.get("last_heavy", 0.0)
    if now - last >= 10.0:
        st.session_state["last_heavy"] = now
        return True
    return False

def render_timer_tab(USER_ID: str, default_week_key: str, goals_map, default_plan):
    st.header("⏱️ Focus Lab")
    st.caption(f"IST Date: **{today_iso()}** • ISO Week: **{default_week_key}**")

    st.toggle("🔊 Sound", value=st.session_state.get("sound_on", True), key="sound_on",
              help="Play a sound when the timer completes.")

    # Today target + quick stats
    st.subheader("🎯 Today’s Target")
    # keep simple here: show only actual pe (you can re-add target inputs if needed)
    actual_pe = total_day_pe(USER_ID, today_iso())
    st.metric("Today's PE", f"{actual_pe:.1f}")

    st.divider()
    left, right = st.columns([1.15, 0.85])

    with right:
        if _should_render_heavy():
            st.subheader("📊 Current Week Allocation (read-only)")
            plan_cur = default_plan
            if not plan_cur or not plan_cur.get("items"):
                st.info("No allocations yet.")
            else:
                from data_access.sessions_repo import aggregate_pe_by_goal_bucket
                pe_map = aggregate_pe_by_goal_bucket(USER_ID, default_week_key)
                rows = []
                for it in sorted(plan_cur.get("items", []), key=lambda x: x.get("priority_rank", 99)):
                    gid = it["goal_id"]
                    g = goals_map.get(gid, {})
                    planned = int(it["planned_current"])
                    cur_pe = pe_map.get(gid, {}).get("current", 0.0)
                    back_pe = pe_map.get(gid, {}).get("backlog", 0.0)
                    rows.append({
                        "Priority": it["priority_rank"],
                        "Goal": g.get("title", gid),
                        "Category": g.get("category", "—"),
                        "Planned": planned,
                        "Backlog In": int(it["backlog_in"]),
                        "Total Target": int(it["total_target"]),
                        "Done Current (pe)": round(cur_pe,1),
                        "Done Backlog (pe)": round(back_pe,1),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("⏳ Timer running — heavy panel will refresh every ~10s.")

    # LEFT: Session Template Logger + Live Timer
    with left:
        st.subheader("🧾 Session Template Logger")

        with st.form("template_logger", clear_on_submit=False):
            date_ist = st.date_input("Date (IST)", value=now_ist().date())
            end_time = st.time_input("End time (IST)", value=now_ist().time().replace(second=0, microsecond=0))
            dur_min = st.number_input("Duration (minutes)", min_value=1, max_value=180, value=25, step=1)

            ttype = st.radio("Type", ["Work (focus)", "Activity", "Break"], horizontal=True)

            goal_id, alloc_bucket, cat, kind, activity_type, deep_work, task_text = (None, None, None, None, None, None, None)

            if ttype == "Work (focus)":
                kind = "focus"
                # current week goals
                plan_src = default_plan or get_week_plan(USER_ID, default_week_key)
                items_for_pick = (plan_src or {}).get("items", [])
                labels, choices = [], []

                pe_map = None
                if items_for_pick:
                    for it in sorted(items_for_pick, key=lambda x: x.get("priority_rank", 99)):
                        gid = it["goal_id"]; g = goals_map.get(gid, {})
                        labels.append(f"[P{it.get('priority_rank')}] {g.get('title', gid)} · {g.get('category','—')}")
                        choices.append((gid, int(it.get("planned_current", 0))))
                    sel = st.selectbox("Goal", labels) if labels else None
                    if sel:
                        idx = labels.index(sel)
                        goal_id, planned_current = choices[idx]
                        alloc_bucket = determine_alloc_bucket(USER_ID, default_week_key, goal_id, planned_current) if planned_current > 0 else None
                        cat = goals_map.get(goal_id, {}).get("category")
                deep_work = st.checkbox("Deep work", value=(dur_min >= 23))
                task_text = st.text_input("Task / Note (optional)")

            elif ttype == "Activity":
                kind = "activity"
                activity_type = st.selectbox("Activity type", list(ALLOWED_ACTIVITY_TYPES), index=1)
                task_text = st.text_input("Activity note (optional)")
                cat = "Wellbeing"

            else:  # Break
                kind = None

            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                save = st.form_submit_button("✅ Log Session", use_container_width=True)
            with sub_col2:
                save_break = st.form_submit_button("✅ Log + Start Break", use_container_width=True, disabled=(ttype != "Work (focus)"))

        # Handle template save
        if save or save_break:
            # build ended_at_ist from date + time
            from datetime import datetime
            ended_at_ist = datetime.combine(date_ist, end_time)
            sid = insert_session(
                USER_ID,
                "W" if ttype in {"Work (focus)", "Activity"} else "B",
                int(dur_min),
                ended_at_ist,
                kind=kind,
                activity_type=activity_type,
                deep_work=deep_work if ttype == "Work (focus)" else None,
                goal_mode=("weekly" if (ttype=="Work (focus)" and goal_id) else "custom" if ttype=="Work (focus)" else None),
                goal_id=goal_id,
                task=task_text if ttype != "Break" else None,
                cat=(cat if ttype != "Break" else None),
                alloc_bucket=(alloc_bucket if goal_id else None),
                break_autostart=(save_break and ttype=="Work (focus)"),
                skipped=False if ttype=="Break" else None,
                post_checkin=None,
                device="web-template"
            )
            st.success(f"Session saved. id={sid}")
            if save_break:
                st.toast("Auto-break started.", icon="⏱️")
            st.rerun()

        # Live Timer
        st.divider()
        st.subheader("⏳ Live Timer")

        if "timer" not in st.session_state:
            st.session_state.timer = {
                "running": False, "end_ts": None, "started_at": None, "completed": False,
                "t": "W", "dur_min": 25, "kind": "focus",
                "activity_type": None,
                "deep_work": True, "goal_id": None, "task": None, "cat": None,
                "alloc_bucket": None, "auto_break": True, "break_min": 5
            }
        timer = st.session_state.timer

        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("▶️ Focus 25m", use_container_width=True):
                timer.update({"dur_min": 25, "kind": "focus"})
                _start_timer(timer)
        with qc2:
            if st.button("▶️ Focus 50m", use_container_width=True):
                timer.update({"dur_min": 50, "kind": "focus"})
                _start_timer(timer)
        with qc3:
            if st.button("▶️ Activity 10m", use_container_width=True):
                timer.update({"dur_min": 10, "kind": "activity"})
                _start_timer(timer)

        if timer["running"]:
            _render_countdown(timer, USER_ID, default_week_key, goals_map, default_plan)

    # play once if flagged
    if st.session_state.get("beep_once"):
        play_finish_sound()
        st.session_state["beep_once"] = False

    # Today’s sessions
    st.divider()
    st.subheader("📝 Today’s Sessions")
    todays = list_today_sessions(USER_ID, today_iso())
    if not todays:
        st.info("No sessions logged yet.")
    else:
        def fmt_row(s):
            kindlab = "Work" if s.get("t") == "W" else "Break"
            if s.get("kind") == "activity": kindlab = "Activity"
            when = to_ist_display(s.get("started_at_ist")).strftime("%H:%M")
            goal_title = goals_map.get(s.get("goal_id"), {}).get("title") if s.get("goal_id") else (s.get("task") or "—")
            return {"When (IST)": when, "Type": kindlab, "Dur (min)": s.get("dur_min"),
                    "Goal/Task": goal_title, "Bucket": s.get("alloc_bucket") or "—",
                    "Deep": "✓" if s.get("deep_work") else "—"}
        st.dataframe([fmt_row(s) for s in todays], use_container_width=True, hide_index=True)
        if st.button("↩️ Undo last entry", use_container_width=True):
            deleted = delete_last_today_session(USER_ID, today_iso())
            st.warning(f"Deleted last session: {deleted}" if deleted else "Nothing to undo.")
            st.rerun()

def _start_timer(timer):
    timer.update({
        "running": True, "completed": False,
        "started_at": now_ist(), "end_ts": now_ist() + timedelta(minutes=int(timer["dur_min"]))
    })
    st.rerun()

def _render_countdown(timer, USER_ID, default_week_key, goals_map, default_plan):
    total_secs = max(int(timer["dur_min"]) * 60, 1)
    remaining_secs = max(int((timer["end_ts"] - now_ist()).total_seconds()), 0)
    elapsed_secs = total_secs - remaining_secs
    pct_done = min(max(elapsed_secs / total_secs, 0.0), 1.0)

    rem_m = remaining_secs // 60
    rem_s = remaining_secs % 60
    started_lbl = timer["started_at"].strftime("%H:%M")
    ends_lbl    = timer["end_ts"].strftime("%H:%M")
    tlabel = "Work (focus)" if (timer["kind"] == "focus") else "Activity"

    st.markdown(
        f"""
        <div style="font-size:1.2rem;margin-bottom:0.25rem;">
          ⏳ <b>{tlabel}</b> — {timer['dur_min']} min
        </div>
        <div style="font-size:2.6rem;font-weight:700;letter-spacing:1px;">
          {rem_m:02d}:{rem_s:02d}
        </div>
        """, unsafe_allow_html=True
    )
    # 1) One-liner fix (keep your variables)
    st.progress(
        pct_done,
        text=f"Elapsed {elapsed_secs//60:02d}:{elapsed_secs%60:02d} • Remaining {rem_m:02d}:{rem_s:02d}"
    )

    meta1, meta2, meta3 = st.columns([1, 1, 1])
    with meta1: st.caption(f"Started: **{started_lbl}**")
    with meta2: st.caption(f"Ends: **{ends_lbl}**")
    with meta3: st.caption(f"Last tick: **{now_ist().strftime('%H:%M:%S')}**")

    colL, colM, colR = st.columns(3)
    stop_now = colL.button("⏹️ Stop / Cancel", use_container_width=True, key="btn_stop_live")
    refresh  = colM.button("🔄 Update now", use_container_width=True, key="btn_refresh_live")
    complete_early = colR.button("✅ Complete now", use_container_width=True, key="btn_complete_live")

    if stop_now:
        timer["running"] = False
        st.warning("Timer canceled.")
        st.rerun()

    if complete_early:
        timer["end_ts"] = now_ist()
        remaining_secs = 0

    if remaining_secs <= 0 and not timer["completed"]:
        ended_at = timer["end_ts"]
        started_at = timer["started_at"]
        dur_min_done = max(1, int(round((ended_at - started_at).total_seconds() / 60.0)))

        sid = insert_session(
            USER_ID, "W", int(dur_min_done), ended_at,
            kind=("focus" if timer["kind"] == "focus" else "activity"),
            activity_type=(timer["activity_type"] if timer["kind"] == "activity" else None),
            deep_work=(timer["deep_work"] if (timer["kind"] == "focus") else None),
            goal_mode=("custom"),
            goal_id=None,
            task=(timer.get("task") if True else None),
            cat=("Wellbeing" if timer["kind"] == "activity" else None),
            alloc_bucket=None,
            break_autostart=(timer["kind"] == "focus" and timer.get("auto_break", False)),
            skipped=None, post_checkin=None, device="web-live"
        )

        st.session_state["beep_once"] = True
        st.session_state["pending_sid"] = sid
        st.session_state["pending_kind"] = timer["kind"]

        timer["completed"] = True
        timer["running"] = False
        st.success(f"Session saved. id={sid}")

        if timer["kind"] != "activity" and timer["auto_break"] and timer["break_min"] > 0:
            timer.update({
                "running": True, "completed": False,
                "t": "B", "dur_min": timer["break_min"], "kind": None,
                "activity_type": None, "deep_work": None,
                "goal_id": None, "task": None, "cat": None, "alloc_bucket": None,
                "auto_break": False,
                "started_at": now_ist(), "end_ts": now_ist() + timedelta(minutes=timer["break_min"])
            })
            st.toast("Auto-break started.", icon="⏱️")
        st.rerun()

    if refresh:
        st.rerun()

    time.sleep(1)
    st.rerun()
