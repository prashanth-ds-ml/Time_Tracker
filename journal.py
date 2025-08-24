# journal.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from user_management import now_ist, collection_reflections, collection_logs

def render_journal(user: str):
    st.header("🧾 Journal")
    tab1, tab2, tab3 = st.tabs(["Reflection", "Add Note", "Browse Notes"])

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
                collection_reflections.update_one(
                    {"user": user, "date": today_iso},
                    {"$set": {
                        "user": user, "date": today_iso, "aligned": aligned,
                        "focus_rating": int(rating), "blockers": blockers.strip(),
                        "notes": notes.strip(), "created_at": datetime.utcnow()
                    }},
                    upsert=True
                )
                st.success("Saved ✨")

        recs = list(collection_reflections.find({"user": user}).sort("date", -1).limit(14))
        if recs:
            st.subheader("Recent Reflections")
            df = pd.DataFrame(recs)
            st.dataframe(df[["date","aligned","focus_rating","blockers","notes"]], use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Add Note")
        with st.form("note_form", clear_on_submit=True):
            c1, c2 = st.columns([1,3])
            with c1:
                d = st.date_input("Date", now_ist())
            with c2:
                content = st.text_area("Your thoughts...", height=140)
            if st.form_submit_button("💾 Save Note"):
                if content.strip():
                    import hashlib
                    nid = hashlib.sha256(f"{d.date().isoformat()}_{content}_{user}".encode()).hexdigest()
                    doc = {"_id": nid, "type":"Note", "date": d.date().isoformat(),
                           "content": content.strip(), "user": user, "created_at": datetime.utcnow()}
                    collection_logs.update_one({"_id": nid}, {"$set": doc}, upsert=True)
                    st.success("Saved")
                else:
                    st.warning("Add some content")

    with tab3:
        st.subheader("Browse Notes")
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("From", now_ist().date()-timedelta(days=7))
        with c2:
            end = st.date_input("To", now_ist().date())
        q = {"type":"Note","user": user, "date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}
        notes = list(collection_logs.find(q).sort("date", -1))
        if notes:
            for n in notes:
                st.subheader(f"📅 {n['date']}")
                st.write(n['content'])
                st.divider()
        else:
            st.info("No notes in this range")
