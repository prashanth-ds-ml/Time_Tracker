# ui/components/sound.py
import streamlit as st
from core.config import FINISH_SOUND_URL

def play_finish_sound():
    if not st.session_state.get("sound_on", True):
        return
    st.markdown(
        f"""
        <audio autoplay>
          <source src="{FINISH_SOUND_URL}" type="audio/mpeg">
        </audio>
        """,
        unsafe_allow_html=True,
    )
