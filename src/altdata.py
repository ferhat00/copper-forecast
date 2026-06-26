"""
altdata.py
==========
Alternative-data loaders for recent copper drivers, plus a single
``attach_altdata_features`` helper that joins the enabled ones onto the raw data
frame before ``build_features``.  Every loader degrades gracefully (returns an
empty frame) when its network/API/file is unavailable, matching the rest of the
data layer — so toggling a feature on never breaks an offline run.

Sources (config_monthly.yaml flags in brackets):
  * LME cash-3M basis        [sources.lme_cash_3m_basis]  -> copper_basis_pct
        reuses src.data_ingestion.fetch_lme_cash_3m (CSV -> real COMEX -> synthetic).
        The one credible *level* beater (Cortazar et al. 2024).
  * GPR + EPU                [features.add_gpr_epu]       -> gpr, epu
        geopolitical-risk (Caldara-Iacoviello) + economic-policy-uncertainty;
        copper long-run *volatility* drivers (Jia et al. 2023).  EPU via FRED;
        GPR via a user CSV (the GPR index is published as a spreadsheet, not FRED).
  * News sentiment           [features.add_news_sentiment] -> news_sentiment
        precomputed (e.g. LLM-scored) headline sentiment CSV; a *direction/density*
        signal in volatile regimes, not a level signal.

Honest note: GPR/EPU/news help volatility, intervals and direction — NOT the
monthly price level.  Only the LME basis is a (modest, regime-conditional) level
play.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily/irregular frame to month-end (last obs), tz-naive."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    try:
        from src.data_ingestion import _resample_to_month_end
        return _resample_to_month_end(out)
    except Exception:
        return out.resample("ME").last()


def load_lme_basis(
    start: str,
    end: Optional[str] = None,
    csv_path: Optional[str] = None,
    try_comex: bool = False,
    freq: str = "M",
) -> pd.DataFrame:
    """LME cash-3M basis as ``copper_basis_pct`` (+ spread). Reuses the data layer.

    ``try_comex=False`` by default to keep offline runs network-free; pass
    ``csv_path`` (a real licensed cash/3M feed) for a genuine carry signal.
    """
    try:
        from src.data_ingestion import fetch_lme_cash_3m
    except Exception as exc:  # pragma: no cover
        logger.warning("altdata: data_ingestion.fetch_lme_cash_3m unavailable (%s)", exc)
        return pd.DataFrame()
    basis = fetch_lme_cash_3m(start=start, end=end, csv_path=csv_path, try_comex=try_comex)
    if basis is None or basis.empty:
        return pd.DataFrame()
    return _month_end(basis) if freq == "M" else basis


def load_gpr_epu(
    start: str,
    end: Optional[str] = None,
    fred_api_key: Optional[str] = None,
    gpr_csv_path: Optional[str] = None,
    epu_series: str = "GEPUCURRENT",
    freq: str = "M",
) -> pd.DataFrame:
    """Geopolitical-risk (CSV) + Economic-Policy-Uncertainty (FRED) indices.

    EPU defaults to the global EPU index ``GEPUCURRENT`` on FRED.  GPR (Caldara &
    Iacoviello) is read from ``gpr_csv_path`` (a date-indexed CSV with a column
    named ``gpr``/``GPR``/``GPRC``).  Either piece is skipped gracefully if its
    source is missing; the function returns whatever it could assemble.
    """
    if end is None:
        end = date.today().isoformat()
    frames = []

    # EPU via FRED (optional dependency + key).
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_api_key) if fred_api_key else Fred()
        s = fred.get_series(epu_series, observation_start=start, observation_end=end)
        frames.append(pd.DataFrame({"epu": s}))
    except Exception as exc:
        logger.warning("altdata: EPU (%s) skipped (%s)", epu_series, exc)

    # GPR via user CSV.
    if gpr_csv_path:
        try:
            raw = pd.read_csv(gpr_csv_path, parse_dates=[0], index_col=0).sort_index()
            lc = {str(c).lower(): c for c in raw.columns}
            col = next((lc[k] for k in ("gpr", "gprc", "gpr_index") if k in lc), None)
            if col is not None:
                frames.append(pd.DataFrame({"gpr": raw[col].astype(float)}))
            else:
                logger.warning("altdata: GPR CSV %s has no gpr column", gpr_csv_path)
        except Exception as exc:
            logger.warning("altdata: GPR CSV load failed (%s)", exc)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[str(start):str(end)] if len(out) else out
    return _month_end(out) if freq == "M" else out


def load_news_sentiment(csv_path: Optional[str], freq: str = "M") -> pd.DataFrame:
    """Precomputed (e.g. LLM-scored) copper news sentiment from a CSV.

    Expects a date-indexed CSV with a sentiment column (``sentiment`` /
    ``news_sentiment`` / ``score``).  Returns an empty frame if no path/file.
    """
    if not csv_path:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(csv_path, parse_dates=[0], index_col=0).sort_index()
        lc = {str(c).lower(): c for c in raw.columns}
        col = next((lc[k] for k in ("news_sentiment", "sentiment", "score") if k in lc), None)
        if col is None:
            logger.warning("altdata: news CSV %s has no sentiment column", csv_path)
            return pd.DataFrame()
        out = pd.DataFrame({"news_sentiment": raw[col].astype(float)})
        return _month_end(out) if freq == "M" else out
    except Exception as exc:
        logger.warning("altdata: news sentiment CSV load failed (%s)", exc)
        return pd.DataFrame()


def attach_altdata_features(
    df: pd.DataFrame,
    cfg: dict,
    start: str,
    end: Optional[str] = None,
    fred_api_key: Optional[str] = None,
    freq: str = "M",
) -> pd.DataFrame:
    """Join the alt-data columns enabled in ``cfg`` onto ``df`` (left join on index).

    Reads ``cfg['features']`` flags (``add_gpr_epu``, ``add_news_sentiment``) and
    ``cfg['sources']`` (``lme_cash_3m_basis`` + optional ``lme_basis_csv``,
    ``gpr_csv_path``, ``news_sentiment_csv``).  Never raises — a failing source is
    logged and skipped, returning ``df`` unchanged for that source.
    """
    feats = (cfg or {}).get("features", {}) or {}
    sources = (cfg or {}).get("sources", {}) or {}
    out = df.copy()

    if sources.get("lme_cash_3m_basis") and "copper_basis_pct" not in out.columns:
        basis = load_lme_basis(start, end, csv_path=sources.get("lme_basis_csv"),
                               try_comex=bool(sources.get("lme_basis_try_comex", False)),
                               freq=freq)
        if not basis.empty:
            out = out.join(basis, how="left")
            logger.info("altdata: attached LME basis (%d rows)", basis.shape[0])

    if feats.get("add_gpr_epu"):
        ge = load_gpr_epu(start, end, fred_api_key=fred_api_key,
                          gpr_csv_path=sources.get("gpr_csv_path"), freq=freq)
        if not ge.empty:
            out = out.join(ge, how="left")
            logger.info("altdata: attached GPR/EPU (%s)", list(ge.columns))

    if feats.get("add_news_sentiment"):
        ns = load_news_sentiment(sources.get("news_sentiment_csv"), freq=freq)
        if not ns.empty:
            out = out.join(ns, how="left")
            logger.info("altdata: attached news sentiment (%d rows)", ns.shape[0])

    return out
