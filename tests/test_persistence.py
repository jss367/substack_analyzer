import io

import pandas as pd
import streamlit as st

from substack_analyzer.persistence import apply_session_bundle, collect_session_bundle


def test_collect_and_apply_session_bundle_roundtrip():
    st.session_state.clear()
    # Minimal session content
    st.session_state["start_free"] = 100
    st.session_state["start_premium"] = 10
    idx = pd.period_range("2024-01", periods=3, freq="M").to_timestamp("M")
    st.session_state["import_total"] = pd.Series([100, 110, 120], index=idx)
    st.session_state["events_df"] = pd.DataFrame({"date": [idx[1]], "type": ["Ad spend"], "cost": [100.0]})

    bundle = collect_session_bundle(include_fit=False, include_sim=False)
    assert isinstance(bundle, (bytes, bytearray)) and len(bundle) > 0

    bio = io.BytesIO(bundle)
    apply_session_bundle(bio)
    # After applying, keys should still be present or restored
    assert "import_total" in st.session_state
