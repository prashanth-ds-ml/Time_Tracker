# analytics.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

from user_management import (
    now_ist, week_bounds_ist, get_user_sessions, get_user_settings, goal_title_map,
    pct_or_dash, gini_from_counts, entropy_norm_from_counts, time_to_minutes
)
from weekly_planner import get_or_create_weekly_plan

def render_analytics_review(user: str, review_week_date):
    st.header("📊 Analytics & Review")

    try:
        mode = st.segmented_control("Mode", options=["Week Review", "Trends"], default="Week Review", key="analytics_mode")
    except Exception:
        mode = st.radio("Mode", ["Week Review", "Trends"], horizontal=True, index=0, key="analytics_mode_radio")

    df_all = get_user_sessions(user)
    if df_all.empty:
        st.info("No sessions yet. Start a Pomodoro to populate analytics.")
        return

    df_all["date_only"] = df_all["date"].dt.date
    df_work = df_all[df_all["pomodoro_type"] == "Work"].copy()
    df_break = df_all[df_all["pomodoro_type"] == "Break"].copy()

    if mode == "Week Review":
        pick_date = st.date_input("Review week of", value=review_week_date, key="review_week_picker")
        week_start, week_end = week_bounds_ist(pick_date)
        plan = get_or_create_weekly_plan(user, week_start)
        planned_alloc = plan.get("allocations", {}) or {}
        total_planned = int(sum(planned_alloc.values())) if planned_alloc else 0

        mask_week = (df_all["date_only"] >= week_start) & (df_all["date_only"] <= week_end)
        dfw = df_work[mask_week].copy()
        dfb = df_break[mask_week].copy()

        work_goal = dfw[dfw["goal_id"].notna()].copy()
        work_custom = dfw[dfw["goal_id"].isna()].copy()
        deep = len(dfw[dfw["duration"] >= 23])
        goal_counts = work_goal.groupby("goal_id").size().values.tolist()

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Plan Adherence", pct_or_dash(len(work_goal), total_planned))
        with c2: st.metric("Capacity Utilization", pct_or_dash(len(dfw), total_planned))
        with c3: st.metric("Deep-work %", pct_or_dash(deep, len(dfw)))
        with c4: st.metric("Balance (Entropy)", f"{entropy_norm_from_counts(goal_counts):.2f}")

        c5, c6, c7, c8 = st.columns(4)
        with c5: st.metric("Gini (Goals)", f"{gini_from_counts(goal_counts):.2f}")
        with c6: st.metric("Custom Share", pct_or_dash(len(work_custom), len(dfw)))
        with c7:
            expected_breaks = len(dfw)
            skip = max(0, expected_breaks - len(dfb))
            st.metric("Break Skip", pct_or_dash(skip, expected_breaks))
        with c8:
            extend = max(0, len(dfb) - len(dfw))
            st.metric("Break Extend", pct_or_dash(extend, expected_breaks))

        st.divider()
        st.subheader("Per-Goal Adherence")
        titles = goal_title_map(user)

        def title_of(gid):
            if gid is None: return "Custom (Unplanned)"
            return titles.get(gid, "(missing)")

        planned_df = pd.DataFrame([{"goal_id": gid, "planned": int(v), "title": title_of(gid)} for gid, v in planned_alloc.items()])
        actual_df = dfw.groupby(dfw["goal_id"]).size().rename("actual").reset_index()
        actual_df["title"] = actual_df["goal_id"].apply(title_of)

        merged = pd.merge(planned_df, actual_df, on=["goal_id","title"], how="outer").fillna(0)
        merged["planned"] = merged["planned"].astype(int)
        merged["actual"] = merged["actual"].astype(int)
        if merged["goal_id"].isna().any():
            merged.loc[merged["goal_id"].isna(), "goal_id"] = None

        cA, cB = st.columns([3,2])
        with cA:
            if not merged.empty:
                fig = px.bar(merged.sort_values("planned", ascending=False), x="title", y=["planned","actual"],
                             barmode="group", title="Planned vs Actual Pomodoros")
                fig.update_layout(height=360, xaxis_title="", legend_title="")
                st.plotly_chart(fig, use_container_width=True)
        with cB:
            total_actual_goals = int(work_goal.shape[0])
            carry = max(0, total_planned - total_actual_goals)
            st.metric("Planned (week)", total_planned if total_planned>0 else 0)
            st.metric("Actual (goals)", total_actual_goals)
            st.metric("Carryover", carry)
            st.metric("Carryover Rate", pct_or_dash(carry, total_planned))

        # Run-rate vs Expected
        if total_planned > 0:
            days = pd.date_range(start=pd.to_datetime(week_start), end=pd.to_datetime(min(week_end, now_ist().date())))
            dfw_goal = dfw[dfw["goal_id"].notna()].copy()
            dfw_goal["date_only"] = dfw_goal["date"].dt.date
            actual_cum, exp_cum = [], []
            for i, ts in enumerate(days):
                cutoff = ts.date()
                actual_to_d = int((dfw_goal["date_only"] <= cutoff).sum())
                expected_to_d = int(round(total_planned * ((i + 1) / len(days))))
                actual_cum.append(actual_to_d); exp_cum.append(expected_to_d)
            rr_df = pd.DataFrame({"day": [ts.strftime("%a %d") for ts in days], "Expected": exp_cum, "Actual": actual_cum})
            fig_rr = px.line(rr_df, x="day", y=["Expected","Actual"], markers=True, title="Run-Rate vs Expected (Goals only)")
            fig_rr.update_layout(height=330, legend_title="")
            st.plotly_chart(fig_rr, use_container_width=True)

    else:
        # Trends
        today = now_ist().date()

        st.subheader("Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("🎯 Total Sessions", len(df_work))
        with col2: st.metric("⏱️ Total Hours", int(df_work["duration"].sum() // 60))
        with col3: st.metric("📅 Active Days", int(df_work.groupby("date_only").size().shape[0]))
        with col4:
            avg_daily = df_work.groupby("date_only").size().mean() if len(df_work) else 0
            st.metric("📊 Avg Daily", f"{avg_daily:.1f}")

        st.divider()
        st.subheader("📈 Daily Performance (Last 30 Days)")
        daily_data = []
        for i in range(30):
            d = today - timedelta(days=29 - i)
            dwork = df_work[df_work["date_only"] == d]
            daily_data.append({"date": d.strftime("%m/%d"), "minutes": int(dwork["duration"].sum())})
        daily_df = pd.DataFrame(daily_data)
        if daily_df["minutes"].sum() > 0:
            fig = px.bar(daily_df, x="date", y="minutes", title="Daily Focus Minutes",
                         color="minutes", color_continuous_scale="Blues")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔍 Insights (Last 30 days)")
        df30 = df_work[df_work["date_only"] >= (today - timedelta(days=30))].copy()
        if not df30.empty:
            by_day = df30.groupby("date_only")["duration"].sum().sort_values(ascending=False)
            best_day = by_day.index[0] if len(by_day) else None
            best_day_min = int(by_day.iloc[0]) if len(by_day) else 0

            starts = [time_to_minutes(x) for x in df30["time"].tolist() if isinstance(x, str)]
            starts = [s for s in starts if s is not None]
            if starts:
                import pandas as pd
                top_hour = pd.Series([s//60 for s in starts]).mode().iloc[0]
                ampm = "AM" if top_hour < 12 else "PM"
                hour_disp = f"{(top_hour if 1 <= top_hour <= 12 else (12 if top_hour%12==0 else top_hour%12))}{ampm}"
            else:
                hour_disp = "—"

            by_cat = df30.groupby("category")["duration"].sum().sort_values(ascending=False)
            if len(by_cat) > 0:
                top_share = (by_cat.iloc[0] / by_cat.sum()) * 100.0
            else:
                top_share = 0.0

            df30_break = df_all[(df_all["pomodoro_type"]=="Break") & (df_all["date_only"] >= (today - timedelta(days=30)))]
            skip_rate = pct_or_dash(max(0, len(df30) - len(df30_break)), len(df30))
            extend_rate = pct_or_dash(max(0, len(df30_break) - len(df30)), len(df30))

            cI1, cI2, cI3, cI4 = st.columns(4)
            with cI1: st.metric("Best day (mins)", f"{best_day_min}", f"{best_day.strftime('%a %d %b') if best_day else '—'}")
            with cI2: st.metric("Focus window", hour_disp)
            with cI3: st.metric("Top category share", f"{top_share:.0f}%")
            with cI4: st.metric("Break skip / extend", f"{skip_rate} / {extend_rate}")

        st.divider()
        st.subheader("🎯 Category Deep Dive")
        time_filter = st.selectbox("Time Period", ["Last 7 days", "Last 30 days", "All time"], index=1, key="trend_time_filter")
        if time_filter == "Last 7 days":
            cutoff = today - timedelta(days=7); fw = df_work[df_work["date_only"] >= cutoff]
        elif time_filter == "Last 30 days":
            cutoff = today - timedelta(days=30); fw = df_work[df_work["date_only"] >= cutoff]
        else:
            fw = df_work

        if fw.empty:
            st.info(f"No data available for {time_filter.lower()}.")
            return

        cat_stats = fw.groupby("category").agg(duration=("duration","sum"),
                                               sessions=("duration","count")).sort_values("duration", ascending=False)
        colA, colB = st.columns([3,2])
        with colA:
            total_time = cat_stats["duration"].sum()
            fig_donut = px.pie(values=cat_stats["duration"], names=cat_stats.index,
                               title=f"📊 Time Distribution by Category ({time_filter})",
                               hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
            total_hours = int(total_time) // 60; total_mins = int(total_time) % 60
            center_text = f"<b>Total</b><br>{total_hours}h {total_mins}m" if total_hours > 0 else f"<b>Total</b><br>{total_mins}m"
            fig_donut.add_annotation(text=center_text, x=0.5, y=0.5, showarrow=False)
            fig_donut.update_layout(height=400, showlegend=True, title_x=0.5)
            st.plotly_chart(fig_donut, use_container_width=True)
        with colB:
            st.markdown("#### Category Performance")
            view = cat_stats.copy()
            view["Time"] = view["duration"].apply(lambda m: f"{int(m//60)}h {int(m%60)}m" if m >= 60 else f"{int(m)}m")
            view["Avg/Session"] = (view["duration"] / view["sessions"]).round(1).astype(str) + "m"
            st.dataframe(view[["Time","sessions","Avg/Session"]], use_container_width=True, hide_index=False,
                         height=min(len(view)*35+38, 300))

        st.subheader("🎯 Task Performance")
        tstats = fw.groupby(["category","task"]).agg(total_minutes=("duration","sum"),
                                                    sessions=("duration","count")).reset_index()
        tstats = tstats.sort_values("total_minutes", ascending=False)
        colC, colD = st.columns([3,2])
        with colC:
            top_tasks = tstats.head(12)
            if not top_tasks.empty:
                fig_tasks = px.bar(top_tasks, x="total_minutes", y="task", color="category",
                                   title=f"Top Tasks by Time Investment ({time_filter})",
                                   color_discrete_sequence=px.colors.qualitative.Set3)
                fig_tasks.update_layout(height=max(400, len(top_tasks)*30),
                                        yaxis={"categoryorder": "total ascending"},
                                        title_x=0.5, showlegend=True)
                st.plotly_chart(fig_tasks, use_container_width=True)
        with colD:
            st.markdown("#### Insights")
            if not tstats.empty:
                total_time = float(tstats["total_minutes"].sum())
                top = tstats.iloc[0]
                share_top = (top["total_minutes"] / total_time) * 100.0 if total_time>0 else 0.0
                top3 = float(tstats.head(3)["total_minutes"].sum())
                top3_share = (top3 / total_time) * 100.0 if total_time>0 else 0.0
                frag_count = int((tstats["sessions"] == 1).sum())
                frag_pct = (frag_count / max(1, len(tstats))) * 100.0
                avg_sess_per_task = float(tstats["sessions"].mean())
                sess_std = float(fw["duration"].std()) if len(fw) >= 2 else 0.0

                st.metric("Top task share", f"{share_top:.0f}%")
                st.metric("Top 3 tasks coverage", f"{top3_share:.0f}%")
                st.metric("Task fragmentation", f"{frag_pct:.0f}% with 1 session")
                st.metric("Avg sessions / task", f"{avg_sess_per_task:.1f}")
                st.metric("Session length σ", f"{sess_std:.1f} min")

                if top3_share > 70:
                    st.warning("⚠️ A few tasks dominate your time. Consider diversifying or batching intentionally.")
                elif frag_pct > 50:
                    st.warning("🧩 Many single-session tasks — try batching similar items to reduce context switching.")
                elif sess_std > 6:
                    st.info("⏱️ Session lengths vary a lot — aim for steadier 25±5m blocks.")
                else:
                    st.success("✅ Good balance across tasks with steady session lengths.")
