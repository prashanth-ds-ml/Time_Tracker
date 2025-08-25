# journal.py
import streamlit as st
import pandas as pd
from db import now_ist, get_or_create_user_day, save_reflection, set_daily_target, get_daily_target, add_note

def render_journal(user: str):
    st.header("🧾 Journal")
    tab1, tab2, tab3 = st.tabs(["Reflection", "Daily Target", "Notes"])

    today_iso = now_ist().date().isoformat()

    with tab1:
        st.subheader("End-of-Day Reflection")
        with st.form("reflection_form", clear_on_submit=True):
            aligned = st.selectbox("Aligned with weekly plan?", ["Yes","Partly","No"])
            rating = st.slider("Focus quality (1-5)", 1, 5, 3)
            blockers = st.text_area("Blockers / distractions")
            notes = st.text_area("Insights / anything to note")
            submitted = st.form_submit_button("💾 Save Reflection")
            if submitted:
                save_reflection(user, aligned, rating, blockers.strip(), notes.strip(), today_iso)
                st.success("Saved ✨")

        # recent (last 14 days)
        # read via user_day docs
        st.subheader("Recent Reflections")
        # pull 14 days
        rows = []
        for i in range(14):
            d = (now_ist().date() - pd.Timedelta(days=i)).isoformat()
            doc = get_or_create_user_day(user, d)
            if doc.get("reflection"):
                r = doc["reflection"]
                rows.append({"date": d, "aligned": r.get("aligned"), "focus_rating": r.get("focus_rating"),
                             "blockers": r.get("blockers",""), "notes": r.get("notes","")})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No recent reflections yet.")

    with tab2:
        st.subheader("Daily Target")
        cur = get_daily_target(user, today_iso)
        if cur is not None:
            st.info(f"Today's target: **{cur}**")
            new = st.number_input("Update target", 1, 12, value=int(cur))
            if st.button("💾 Update target"):
                set_daily_target(user, int(new), today_iso)
                st.success("Updated target.")
                st.rerun()
        else:
            val = st.number_input("Set target (pomodoros)", 1, 12, value=1)
            if st.button("Set"):
                set_daily_target(user, int(val), today_iso)
                st.success("Saved target.")
                st.rerun()

    with tab3:
        st.subheader("Notes")
        with st.form("note_form", clear_on_submit=True):
            content = st.text_area("Your thoughts...", height=140)
            sub = st.form_submit_button("💾 Save Note")
            if sub:
                if content.strip():
                    add_note(user, content.strip(), today_iso)
                    st.success("Saved")
                else:
                    st.warning("Add some content")
