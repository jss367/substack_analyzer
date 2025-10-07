import io
import json
import zipfile
from contextlib import suppress
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# Stabilize Streamlit session_state when running headless (pytest or plain Python)
try:
    from streamlit import runtime as _st_runtime

    if not _st_runtime.exists():
        import streamlit.runtime.state.session_state_proxy as _ssp
        from streamlit.runtime.state.safe_session_state import SafeSessionState as _SafeSS
        from streamlit.runtime.state.session_state import SessionState as _SS

        if not hasattr(st, "_sa_headless_state"):
            st._sa_headless_state = _SafeSS(_SS(), lambda: None)
        _ssp.get_session_state = lambda: st._sa_headless_state  # type: ignore[assignment]
except Exception:
    pass


def collect_session_bundle(include_fit: bool, include_sim: bool) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        meta = {
            "schema_version": 1,
            "app_name": "Substack Ads ROI Simulator",
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        zf.writestr("metadata.json", json.dumps(meta, indent=2))

        # state: only include scalar configuration keys (avoid DataFrames/Series)
        allowed_keys = [
            "start_free",
            "start_premium",
            "horizon_months",
            "organic_growth",
            "churn_free",
            "churn_prem",
            "conv_new",
            "conv_ongoing",
            "cac",
            "ad_manager_fee",
            "price_monthly",
            "price_annual",
            "substack_pct",
            "stripe_pct",
            "stripe_flat",
            "annual_share",
            "spend_mode_index",
            "ad_stage1",
            "ad_stage2",
            "ad_const",
            "est_window",
            "max_changes_detect",
        ]
        state: dict[str, object] = {}
        for k in allowed_keys:
            if k in st.session_state:
                v = st.session_state.get(k)
                if hasattr(v, "item"):
                    with suppress(Exception):
                        v = v.item()
                if isinstance(v, (int, float, str, bool)) or v is None:
                    state[k] = v
        zf.writestr("state.json", json.dumps(state, indent=2))

        # series
        total = st.session_state.get("import_total")
        if isinstance(total, pd.Series) and not total.empty:
            df_t = total.rename("count").to_frame()
            df_t.index.name = "date"
            zf.writestr("series_total.csv", df_t.to_csv(index=True))

        paid = st.session_state.get("import_paid")
        if isinstance(paid, pd.Series) and not paid.empty:
            df_p = paid.rename("count").to_frame()
            df_p.index.name = "date"
            zf.writestr("series_paid.csv", df_p.to_csv(index=True))

        # events
        ev = st.session_state.get("events_df")
        if isinstance(ev, pd.DataFrame) and not ev.empty:
            ev_out = ev.copy()
            with suppress(Exception):
                ev_out["date"] = pd.to_datetime(ev_out["date"]).dt.date.astype(str)
            zf.writestr("events.csv", ev_out.to_csv(index=False))

        # covariates
        covariates = st.session_state.get("covariates_df")
        if isinstance(covariates, pd.DataFrame) and not covariates.empty:
            cov_out = covariates.copy()
            cov_out = cov_out.reset_index().rename(columns={cov_out.index.name or "index": "date"})
            with suppress(Exception):
                cov_out["date"] = pd.to_datetime(cov_out["date"]).dt.date.astype(str)
            zf.writestr("covariates.csv", cov_out.to_csv(index=False))

        # features
        features = st.session_state.get("features_df")
        if isinstance(features, pd.DataFrame) and not features.empty:
            feat_out = features.copy()
            feat_out = feat_out.reset_index().rename(columns={feat_out.index.name or "index": "date"})
            with suppress(Exception):
                feat_out["date"] = pd.to_datetime(feat_out["date"]).dt.date.astype(str)
            zf.writestr("features.csv", feat_out.to_csv(index=False))

        # fit
        if include_fit and (fit := st.session_state.get("pwlog_fit")) is not None:
            with suppress(Exception):
                fit_dict = {
                    "carrying_capacity": float(getattr(fit, "carrying_capacity", 0.0)),
                    "segment_growth_rates": [float(x) for x in getattr(fit, "segment_growth_rates", [])],
                    "breakpoints": list(getattr(fit, "breakpoints", [])),
                    "gamma_pulse": float(getattr(fit, "gamma_pulse", 0.0)),
                    "gamma_step": float(getattr(fit, "gamma_step", 0.0)),
                    "r2_on_deltas": float(getattr(fit, "r2_on_deltas", 0.0)),
                }
                zf.writestr("fit.json", json.dumps(fit_dict, indent=2))
                fitted = getattr(fit, "fitted_series", None)
                if isinstance(fitted, pd.Series) and not fitted.empty:
                    df_f = fitted.rename("fitted").to_frame()
                    df_f.index.name = "date"
                    zf.writestr("fit_fitted_series.csv", df_f.to_csv(index=True))

        # simulation
        if include_sim and (sim := st.session_state.get("sim_df")) is not None:
            with suppress(Exception):
                zf.writestr("sim.csv", sim.to_csv(index=False))

    buf.seek(0)
    return buf.getvalue()


def apply_session_bundle(file_like) -> None:
    # In headless/tests, Streamlit may not maintain a persistent SessionState.
    # Create a stable SafeSessionState and monkeypatch the getter so subsequent
    # accesses within this process see the same backing state.
    try:
        from streamlit import runtime as _st_runtime

        if not _st_runtime.exists():
            import streamlit.runtime.state.session_state_proxy as _ssp
            from streamlit.runtime.state.safe_session_state import SafeSessionState as _SafeSS
            from streamlit.runtime.state.session_state import SessionState as _SS

            if not hasattr(st, "_sa_headless_state"):
                st._sa_headless_state = _SafeSS(_SS(), lambda: None)
            _ssp.get_session_state = lambda: st._sa_headless_state  # type: ignore[assignment]
    except Exception:
        pass

    with zipfile.ZipFile(file_like, mode="r") as zf:
        # Ensure keys exist up-front in headless/test environments
        if "import_total" not in st.session_state:
            st.session_state["import_total"] = pd.Series(dtype=float)
        if "import_paid" not in st.session_state:
            st.session_state["import_paid"] = pd.Series(dtype=float)
        # metadata
        with suppress(KeyError, Exception):
            meta = json.loads(zf.read("metadata.json"))
            if int(meta.get("schema_version", 0)) != 1:
                raise ValueError("Unsupported bundle version. Please update the app.")

        # state
        with suppress(KeyError, Exception):
            state = json.loads(zf.read("state.json"))
            if isinstance(state, dict):
                # Apply scalar config state directly to session state in headless/tests
                for k, v in state.items():
                    st.session_state[k] = v

        # series: total
        with suppress(KeyError, Exception):
            df_t = pd.read_csv(io.BytesIO(zf.read("series_total.csv")))
            if {"date", "count"}.issubset(df_t.columns):
                s_t = (
                    df_t.assign(date=lambda d: pd.to_datetime(d["date"]))
                    .dropna(subset=["date"])
                    .set_index("date")["count"]
                )
                s_t = pd.to_numeric(s_t, errors="coerce").dropna()
                if not s_t.empty:
                    s_t.index = s_t.index.to_period("M").to_timestamp("M")
                    st.session_state.update({"import_total": s_t.sort_index()})

        # series: paid
        with suppress(KeyError, Exception):
            df_p = pd.read_csv(io.BytesIO(zf.read("series_paid.csv")))
            if {"date", "count"}.issubset(df_p.columns):
                s_p = (
                    df_p.assign(date=lambda d: pd.to_datetime(d["date"]))
                    .dropna(subset=["date"])
                    .set_index("date")["count"]
                )
                s_p = pd.to_numeric(s_p, errors="coerce").dropna()
                if not s_p.empty:
                    s_p.index = s_p.index.to_period("M").to_timestamp("M")
                    st.session_state.update({"import_paid": s_p.sort_index()})

        # events
        with suppress(KeyError, Exception):
            ev = pd.read_csv(io.BytesIO(zf.read("events.csv")))
            if not ev.empty:
                with suppress(Exception):
                    ev["date"] = pd.to_datetime(ev["date"]).dt.date
                st.session_state.update({"events_df": ev})

        # covariates
        with suppress(KeyError, Exception):
            cov = pd.read_csv(io.BytesIO(zf.read("covariates.csv")))
            if {"date", "ad_spend"}.issubset(cov.columns):
                cov["date"] = pd.to_datetime(cov["date"]).dt.to_period("M").dt.to_timestamp("M")
                st.session_state.update({"covariates_df": cov.set_index("date").sort_index()})

        # features
        with suppress(KeyError, Exception):
            feat = pd.read_csv(io.BytesIO(zf.read("features.csv")))
            need = {"date", "pulse", "step", "adstock", "ad_effect_log"}
            if need.issubset(feat.columns):
                feat["date"] = pd.to_datetime(feat["date"]).dt.to_period("M").to_timestamp("M")
                st.session_state.update({"features_df": feat.set_index("date").sort_index()})

        # Ensure required keys exist even if the bundle lacks certain artifacts
        # This helps in headless/test environments where session_state may not persist
        if "import_total" not in st.session_state:
            st.session_state.update({"import_total": pd.Series(dtype=float)})
        if "import_paid" not in st.session_state:
            st.session_state.update({"import_paid": pd.Series(dtype=float)})
