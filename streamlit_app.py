import streamlit as st
import os
from src.match_players import match_embeddings

st.set_page_config(page_title="MatchVision", layout="wide")
st.title("🎯 MatchVision: Player Re-Identification Viewer")

if st.button("🔍 Run Re-ID Matching"):
    matches = match_embeddings("features/tacticam.npy", "features/broadcast.npy")

    st.success(f"✅ {len(matches)} player matches found.")

    for pid, (t_file, b_file) in enumerate(matches):
        t_path = os.path.join("outputs", "crops", "tacticam", t_file)
        b_path = os.path.join("outputs", "crops", "broadcast", b_file)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**🎽 Player ID: P{pid}**")
            st.markdown(f"`{t_file}`")
            if os.path.exists(t_path):
                st.image(t_path, width=250)

        with col2:
            st.markdown("**📺 Matched With**")
            st.markdown(f"`{b_file}`")
            if os.path.exists(b_path):
                st.image(b_path, width=250)

        st.markdown("---")
