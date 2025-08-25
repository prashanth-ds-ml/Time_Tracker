# analytics.py
import streamlit as st
import pandas as pd
import math
from db import now_ist, week_bounds, get_sessions_df, get_or_create_week_plan

def _safe_div(n, d, default=0.0):
    try:
        if d is None or d == 0:
            return default
        return float(n)/float(d)
    except Exception:
        return default

def _pct(n, d):
    if d is None or d <= 0:
        return "—"
    return f"{100.0*_safe_div(n,d):.0f}%"

def _gini(counts):
    arr = [c for c in counts if c is not None and c >= 0]
    if not arr: return 0.0
    arr = sorted(arr)
    n = len(arr); s = sum(arr)
    if s == 0: return 0.0
    cum = 0.0
    for i,x in enumerate(arr,1):
        cum += i*x
    return (2.0*cum)/(n*s) - (n+1.0)/n

def _entropy_norm(counts):
    arr = [c for c in counts if c is not None and c > 0]
    k = len(arr)
    if k <= 1: return 0.0
    s = float(sum(arr))
    H = -sum((c/s)*math.log((c/s),2) for c in arr)
    return H / math.log(k,2)

def _time_to_minutes(tstr):
    try:
        import datetime as _dt
        dt = _dt.datetime.strptime(tstr, "%I:%M %p")
        return dt.hour*60 + dt.minute
    except Exception:
        return None

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
        with c1: st.metric("Plan Adherence", _pct(len(work_goal), planned))
        with c2: st.metric("Capacity Utilization", _pct(len(dfw), planned))
        with c3: st.metric("Deep-work %", _pct(deep, len(dfw)))
        with c4: st.metric("Balance (Entropy)", f"{_entropy_norm(goal_counts):.2f}")

        c5,c6,c7,c8 = st.columns(4)
        with c5: st.metric("Gini (Goals)", f"{_gini(goal_counts):.2f}")
        with c6: st.metric("Custom Share", _pct(len(work_custom), len(dfw)))
        with c7:
            exp_breaks = len(dfw)
            skip = max(0, exp_breaks - len(dfb))
            st.metric("Break Skip", _pct(skip, exp_breaks))
        with c8:
            extend = max(0, len(dfb) - len(dfw))
            st.metric("Break Extend", _pct(extend, len(dfw)))

        # Run-rate vs Expected
        if planned > 0 and not dfw.empty:
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
            starts = [m for m in df30["time"].dropna().map(_time_to_minutes) if m is not None]
            if starts:
                top_hour = pd.Series([s//60 for s in starts]).mode().iloc[0]
                ampm = "AM" if top_hour < 12 else "PM"
                focus_hour = f"{(top_hour if 1<=top_hour<=12 else (12 if top_hour%12==0 else top_hour%12))}{ampm}"
            else:
                focus_hour = "—"
            by_cat = df30.groupby("category")["duration"].sum().sort_values(ascending=False)
            top_cat = by_cat.index[0] if len(by_cat) else "—"
            top_share = 100.0 * _safe_div(by_cat.iloc[0], by_cat.sum()) if len(by_cat) else 0
            df30_break = df_all[(df_all["pomodoro_type"]=="Break") & (df_all["date_only"] >= (today - pd.Timedelta(days=30)))]
            skip_rate = _pct(max(0, len(df30) - len(df30_break)), len(df30))
            extend_rate = _pct(max(0, len(df30_break) - len(df30)), len(df30))
            i1,i2,i3,i4 = st.columns(4)
            with i1: st.metric("Best day (mins)", f"{best_day_val}", best_day_lbl)
            with i2: st.metric("Focus window", focus_hour)
            with i3: st.metric("Top category share", f"{top_share:.0f}%")
            with i4: st.metric("Break skip / extend", f"{skip_rate} / {extend_rate}")

        st.divider()
        st.subheader("🎯 Category mix (select period)")
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
            # Small insight
            total_time = view["Minutes"].sum()
            top = view.iloc[0]
            share = 100.0*_safe_div(top["Minutes"], total_time)
            if share > 50:
                st.warning("⚖️ One task dominates your time. Consider splitting or capping it.")
            elif share > 25:
                st.info("🎯 Clear primary task emerging this period.")
            else:
                st.success("✅ Time is well distributed across tasks.")
