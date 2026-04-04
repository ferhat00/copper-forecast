"""
data_ingestion.py
=================
Downloads and harmonises raw price / macro data from:
  - yfinance  (copper futures & spot ETF, base metals, copper miners, steel
               producers, energy, currencies, equity indices, interest rates,
               broad commodities, construction/industrial demand proxies,
               energy-transition plays, and shipping / dry-bulk freight)
  - FRED API  (industrial production, capacity utilisation, construction,
               manufacturing orders, monetary policy, credit conditions,
               inflation, trade, labour market, and commodity spot prices)

All series are aligned to a common daily date-index (business days),
forward-filled for non-trading gaps, and returned as a single tidy
DataFrame.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker map
# ---------------------------------------------------------------------------

YFINANCE_TICKERS: dict[str, str] = {
    # ── Target ───────────────────────────────────────────────────────────────
    "copper_price":        "HG=F",        # COMEX copper front-month ($/lb → $/t)

    # ── Precious metals (futures) ─────────────────────────────────────────────
    "gold":                "GC=F",        # Gold futures ($/oz) — safe-haven / USD proxy
    "silver":              "SI=F",        # Silver futures — industrial + safe-haven
    "platinum":            "PL=F",        # Platinum futures — auto catalysts / industrial
    "palladium":           "PA=F",        # Palladium futures — auto / supply-constrained

    # ── Base metals (futures & ETFs) ─────────────────────────────────────────
    "aluminium":           "ALI=F",       # CME Aluminium futures ($/t) — substitute metal
    "base_metals_etf":     "DBB",         # Invesco DB Base Metals ETF (Al + Zn + Cu basket)
    "copper_etf":          "CPER",        # US Copper Index Fund (spot Cu exposure)
    "rare_earth":          "REMX",        # VanEck Rare Earth/Strategic Metals ETF

    # ── Copper miners & diversified miners ───────────────────────────────────
    "copper_miners_etf":   "COPX",        # Global X Copper Miners ETF (pure-play)
    "metals_mining_etf":   "XME",         # SPDR S&P Metals & Mining ETF (broad)
    "global_miners_etf":   "PICK",        # iShares MSCI Global Metals & Mining ETF
    "fcx":                 "FCX",         # Freeport-McMoRan — world's largest public Cu miner
    "scco":                "SCCO",        # Southern Copper Corp (Peru / Mexico operations)
    "bhp":                 "BHP",         # BHP Group — Olympic Dam + Escondida
    "rio":                 "RIO",         # Rio Tinto — Kennecott + Oyu Tolgoi
    "vale":                "VALE",        # Vale S.A. — Cu + Ni (Brazil / global)
    "teck":                "TECK",        # Teck Resources — QB2 (Canada/Chile)
    "glencore":            "GLNCY",       # Glencore ADR — major Cu trader & miner
    "anglo_american":      "NGLOY",       # Anglo American ADR — Los Bronces, Collahuasi
    "antofagasta":         "ANFGF",       # Antofagasta plc OTC — pure-play Cu, Chile
    "ivanhoe":             "IVN.TO",      # Ivanhoe Mines — Kamoa-Kakula (DRC)
    "first_quantum":       "FM.TO",       # First Quantum Minerals — Cobre Panama
    "lundin":              "LUNMF",       # Lundin Mining OTC — Chapada, Candelaria
    "ero_copper":          "ERO",         # Ero Copper Corp — Brazil

    # ── Steel & iron-ore complex (downstream Cu demand signal) ───────────────
    "us_steel":            "X",           # US Steel Corp
    "nucor":               "NUE",         # Nucor — largest US electric-arc steelmaker
    "cleveland_cliffs":    "CLF",         # Cleveland-Cliffs — US steel + iron ore
    "arcelormittal":       "MT",          # ArcelorMittal — global steel benchmark
    "ternium":             "TX",          # Ternium — Latin America steel (overlaps Cu regions)

    # ── Aluminium producers (substitute metal / energy cost signal) ──────────
    "alcoa":               "AA",          # Alcoa Corporation

    # ── Gold miners (risk-off / miner sentiment) ─────────────────────────────
    "gold_miners_etf":     "GDX",         # VanEck Gold Miners ETF
    "silver_etf":          "SLV",         # iShares Silver Trust

    # ── Energy (input costs for mining & smelting) ───────────────────────────
    "oil_wti":             "CL=F",        # WTI crude front-month ($/bbl)
    "oil_brent":           "BZ=F",        # Brent crude ($/bbl) — global benchmark
    "nat_gas":             "NG=F",        # Henry Hub natural gas — smelter energy cost
    "heating_oil":         "HO=F",        # Heating oil / diesel — mining fleet fuel
    "coal_arch":           "ARCH",        # Arch Resources — metallurgical coal
    "coal_btu":            "BTU",         # Peabody Energy — thermal + met coal
    "coal_amr":            "AMR",         # Alpha Metallurgical — hard coking coal (steel)

    # ── Currencies — major ────────────────────────────────────────────────────
    "dxy":                 "DX-Y.NYB",   # US Dollar Index (ICE) — primary Cu price driver
    "eur_usd":             "EURUSD=X",   # Euro / USD
    "gbp_usd":             "GBPUSD=X",   # British Pound / USD
    "usd_jpy":             "USDJPY=X",   # USD / Japanese Yen — risk-on/off sentiment
    "usd_chf":             "USDCHF=X",   # USD / Swiss Franc — safe-haven proxy

    # ── Currencies — commodity & EM (supply-country & consumer-country FX) ───
    "aud_usd":             "AUDUSD=X",   # Australian Dollar — major Cu & Au exporter
    "nzd_usd":             "NZDUSD=X",   # New Zealand Dollar — commodity currency
    "usd_cad":             "USDCAD=X",   # USD / Canadian Dollar — Teck, resources
    "usd_cny":             "USDCNY=X",   # USD / Chinese Renminbi — #1 Cu consumer
    "usd_cnh":             "USDCNH=X",   # USD / Offshore CNH — market-rate CNY
    "usd_clp":             "USDCLP=X",   # USD / Chilean Peso — #1 Cu-producing nation
    "usd_pen":             "USDPEN=X",   # USD / Peruvian Sol — #2 Cu-producing nation
    "usd_zar":             "USDZAR=X",   # USD / South African Rand — mining economy
    "usd_brl":             "USDBRL=X",   # USD / Brazilian Real — Vale, mining
    "usd_mxn":             "USDMXN=X",  # USD / Mexican Peso — SCCO operations
    "usd_nok":             "USDNOK=X",  # USD / Norwegian Krone — oil/commodity proxy
    "usd_rub":             "USDRUB=X",  # USD / Russian Ruble — Norilsk Nickel signal

    # ── Equity indices — global demand ────────────────────────────────────────
    "sp500":               "^GSPC",       # S&P 500 — US economic health
    "nasdaq":              "^IXIC",       # Nasdaq Composite — risk appetite
    "shanghai":            "000001.SS",   # Shanghai Composite — China demand signal
    "hang_seng":           "^HSI",        # Hang Seng — HK/China financial proxy
    "ftse":                "^FTSE",       # FTSE 100 — UK + major miners (BHP, RIO, GLEN)
    "dax":                 "^GDAXI",      # DAX — German industrial demand
    "nikkei":              "^N225",       # Nikkei 225 — Japan industrial demand
    "em_etf":              "EEM",         # iShares MSCI Emerging Markets ETF
    "china_etf":           "FXI",         # iShares China Large-Cap ETF
    "latam_etf":           "ILF",         # iShares Latin America 40 ETF (CL, PE, BR)

    # ── Volatility / risk sentiment ───────────────────────────────────────────
    "vix":                 "^VIX",        # CBOE VIX — near-term market fear gauge
    "vix_3m":              "^VIX3M",      # 3-Month VIX — term structure of fear

    # ── Interest rates (Treasury yields) ─────────────────────────────────────
    "t3m_yield":           "^IRX",        # 3-Month T-Bill yield
    "t5y_yield":           "^FVX",        # 5-Year Treasury yield
    "t10y_yield":          "^TNX",        # 10-Year Treasury yield — global benchmark
    "t30y_yield":          "^TYX",        # 30-Year Treasury yield

    # ── Broad commodity indices ───────────────────────────────────────────────
    "commodity_idx":       "DBC",         # Invesco DB Commodity Index
    "gsci_etf":            "GSG",         # iShares S&P GSCI Commodity Index ETF
    "lumber":              "LBS=F",       # Lumber futures — construction activity
    "wheat":               "ZW=F",        # Wheat futures — agri / energy cost linkage
    "corn":                "ZC=F",        # Corn futures — ethanol / energy linkage

    # ── Construction & infrastructure demand ─────────────────────────────────
    "homebuilders_etf":    "XHB",         # SPDR S&P Homebuilders ETF — wiring & plumbing
    "construction_etf":    "PKB",         # Invesco Building & Construction ETF
    "caterpillar":         "CAT",         # Caterpillar — mining & construction equipment

    # ── Industrials (electrical & manufacturing Cu demand) ───────────────────
    "industrials_etf":     "XLI",         # Industrial Select Sector SPDR ETF

    # ── Energy transition (structural long-run Cu demand driver) ─────────────
    "clean_energy_etf":    "ICLN",        # iShares Global Clean Energy ETF
    "lithium_etf":         "LIT",         # Global X Lithium & Battery Tech ETF
    "solar_etf":           "TAN",         # Invesco Solar ETF
    "tesla":               "TSLA",        # Tesla — EV Cu intensity bellwether

    # ── Shipping & dry-bulk freight ───────────────────────────────────────────
    "dry_bulk_etf":        "BDRY",        # Breakwave Dry Bulk Shipping ETF (BDI proxy)
    "star_bulk":           "SBLK",        # Star Bulk Carriers — Capesize dry bulk
    "golden_ocean":        "GOGL",        # Golden Ocean Group — Capesize dry bulk
    "genco":               "GNK",         # Genco Shipping & Trading — dry bulk
    "eagle_bulk":          "EGLE",        # Eagle Bulk Shipping — Supramax dry bulk
    "diana_shipping":      "DSX",         # Diana Shipping — dry bulk
    "safe_bulkers":        "SB",          # Safe Bulkers — Panamax dry bulk
    "zim":                 "ZIM",         # ZIM Integrated Shipping — container
    "matson":              "MATX",        # Matson — Trans-Pacific container (US↔China)

    # ── Additional miners, royalty & streaming ────────────────────────────────
    "newmont":             "NEM",         # Newmont Corp — world's largest gold miner, Cu by-product
    "wheaton":             "WPM",         # Wheaton Precious Metals — Cu/Au/Ag streaming royalties
    "franco_nevada":       "FNV",         # Franco-Nevada — diversified royalty/streaming
    "norilsk":             "NILSY",       # Norilsk Nickel ADR — world's #1 Ni + major Cu producer
    "kinross":             "KGC",         # Kinross Gold — Cu+Au operations (Americas/Africa)
    "agnico":              "AEM",         # Agnico Eagle — gold+copper miner

    # ── Steel ─────────────────────────────────────────────────────────────────
    "steel_etf":           "SLX",         # VanEck Steel ETF — pure-play steel industry

    # ── Uranium / nuclear (long-run Cu grid demand) ───────────────────────────
    "uranium_etf":         "URA",         # Global X Uranium ETF — nuclear energy transition
    "uranium_miners":      "URNM",        # Sprott Uranium Miners ETF

    # ── Broad commodity ───────────────────────────────────────────────────────
    "commodity_broad2":    "PDBC",        # Invesco Optimum Yield Diversified Commodity ETF
}

FRED_SERIES: dict[str, str] = {
    # ── US Industrial Activity ────────────────────────────────────────────────
    "indpro":              "INDPRO",          # Industrial Production Index (total)
    "indpro_mfg":          "IPMAN",           # Industrial Production: Manufacturing
    "capacity_util":       "TCU",             # Total Industry Capacity Utilization (%)
    "capacity_util_mfg":   "MCUMFNS",         # Manufacturing Capacity Utilization (%)

    # ── Construction & Real Estate (wiring, plumbing, HVAC = Cu demand) ──────
    "housing_starts":      "HOUST",           # Total Housing Starts (1000s of units)
    "building_permits":    "PERMIT",          # New Private Housing Permits
    "construction_spend":  "TTLCONS",         # Total Construction Spending ($mn)
    "res_construction":    "TLRESCONS",       # Residential Construction Spending ($mn)

    # ── Manufacturing / Orders ────────────────────────────────────────────────
    "mfg_employment":      "MANEMP",          # Manufacturing Employees (000s)
    "mfg_new_orders":      "AMTMNO",          # Manufacturers' New Orders: Total ($mn)
    "mfg_unfilled_orders": "AMTMUO",          # Manufacturers' Unfilled Orders ($mn)
    "durable_goods":       "DGORDER",         # Durable Goods New Orders ($mn)

    # ── Automotive (major Cu consumer — ~1.5 kg/ICE, ~4 kg/EV) ──────────────
    "auto_sales":          "TOTALSA",         # Total Vehicle Sales (million SAAR)

    # ── Monetary Policy & Money Supply ───────────────────────────────────────
    "fed_funds_rate":      "FEDFUNDS",        # Federal Funds Effective Rate (%)
    "us_m2":               "M2SL",            # M2 Money Supply ($ billions)

    # ── Yields & Real Rates ───────────────────────────────────────────────────
    "real_yield_10y":      "DFII10",          # 10Y TIPS Real Yield (%)
    "real_yield_5y":       "DFII5",           # 5Y TIPS Real Yield (%)
    "inflation_be_10y":    "T10YIE",          # 10Y Inflation Breakeven (%)
    "inflation_be_5y":     "T5YIE",           # 5Y Inflation Breakeven (%)
    "yield_spread_10y2y":  "T10Y2Y",          # 10Y–2Y Treasury Spread (yield curve)
    "yield_spread_10y3m":  "T10Y3M",          # 10Y–3M Treasury Spread

    # ── Credit Conditions ─────────────────────────────────────────────────────
    "hy_oas":              "BAMLH0A0HYM2",    # ICE BofA High Yield OAS (credit stress)
    "ig_oas":              "BAMLC0A0CM",      # ICE BofA Investment Grade OAS
    "fin_conditions":      "NFCI",            # Chicago Fed National Financial Conditions Index

    # ── Inflation / Prices ────────────────────────────────────────────────────
    "cpi":                 "CPIAUCSL",        # CPI All Urban Consumers (SA)
    "core_cpi":            "CPILFESL",        # Core CPI ex Food & Energy
    "ppi_all":             "PPIACO",          # PPI All Commodities
    "ppi_metals":          "PPICMM",          # PPI Metals & Metal Products
    "core_pce":            "PCEPILFE",        # Core PCE Deflator

    # ── Trade & USD ───────────────────────────────────────────────────────────
    "trade_balance":       "BOPGSTB",         # Trade Balance: Goods & Services ($bn)
    "usd_broad_index":     "DTWEXBGS",        # Trade Weighted USD Broad Index

    # ── Labour Market ─────────────────────────────────────────────────────────
    "unemployment":        "UNRATE",          # US Unemployment Rate (%)
    "nonfarm_payrolls":    "PAYEMS",          # Nonfarm Payrolls (000s)
    "labor_force_part":    "CIVPART",         # Civilian Labor Force Participation Rate (%)

    # ── Consumer Demand ───────────────────────────────────────────────────────
    "retail_sales":        "RSXFS",           # Advance Retail Sales ex Food Services ($mn)
    "consumer_sentiment":  "UMCSENT",         # U of Michigan Consumer Sentiment Index

    # ── Inventories ───────────────────────────────────────────────────────────
    "inventory_sales":     "ISRATIO",         # Total Business Inventory-to-Sales Ratio

    # ── Energy Spot Prices (FRED daily) ──────────────────────────────────────
    "wti_spot":            "DCOILWTICO",      # WTI Crude Oil Spot Price ($/bbl)
    "nat_gas_spot":        "DHHNGSP",         # Henry Hub Natural Gas Spot ($/MMBtu)

    # ── Copper Price Reference (independent series) ───────────────────────────
    "copper_lme_monthly":  "PCOPPUSDM",       # World Bank LME Copper Price (monthly, $/mt)

    # ── China Activity (OECD via FRED) ────────────────────────────────────────
    "china_mfg_prod":      "MANMM101CNM657S", # China Manufacturing Production Index (OECD)
}

# ---------------------------------------------------------------------------
# Alpha Vantage — monthly LME / World Bank commodity prices
# (free tier: 25 req/day; function names match AV commodity API)
# ---------------------------------------------------------------------------

ALPHA_VANTAGE_COMMODITIES: dict[str, str] = {
    # ── Base metals (LME spot, monthly, World Bank source) ────────────────────
    "av_copper":           "COPPER",          # LME copper $/metric ton
    "av_aluminum":         "ALUMINUM",        # LME aluminum $/metric ton — substitute metal
    "av_zinc":             "ZINC",            # LME zinc $/metric ton — galvanising / construction
    "av_nickel":           "NICKEL",          # LME nickel $/metric ton — battery metals / co-mined
    "av_lead":             "LEAD",            # LME lead $/metric ton — battery / construction
    "av_tin":              "TIN",             # LME tin $/metric ton — solder / electronics
    # ── Bulk commodities ─────────────────────────────────────────────────────
    "av_iron_ore":         "IRON_ORE",        # Iron ore $/dry metric ton — steel/construction demand
    "av_coal":             "COAL",            # Australian thermal coal $/metric ton — smelter energy
    # ── Energy (monthly cross-check with FRED/yfinance) ──────────────────────
    "av_wti":              "CRUDE_OIL_WTI",   # WTI crude $/bbl
    "av_brent":            "CRUDE_OIL_BRENT", # Brent crude $/bbl
    "av_natural_gas":      "NATURAL_GAS",     # Henry Hub $/MMBtu
}

# ---------------------------------------------------------------------------
# EIA — electricity & energy data relevant to copper demand / cost
# (free API key; series IDs use EIA v1 dot-notation format)
# ---------------------------------------------------------------------------

EIA_COPPER_SERIES: dict[str, str] = {
    # ── Industrial electricity consumption (primary Cu demand proxy) ──────────
    "eia_indl_elec_sales":  "ELEC.SALES.US-IND.M",   # US industrial electricity retail sales (M kWh, monthly)
    "eia_comm_elec_sales":  "ELEC.SALES.US-COM.M",   # US commercial electricity retail sales (M kWh)
    "eia_total_elec_sales": "ELEC.SALES.US-ALL.M",   # US total electricity retail sales (M kWh)
    "eia_total_elec_gen":   "ELEC.GEN.ALL-US-99.M",  # US total net electricity generation (k MWh)
    # ── Natural gas storage (weekly; energy cost / smelting signal) ──────────
    "eia_gas_storage":      "NG.NW2_EPG0_SWO_R48_BCF.W",  # Lower-48 nat gas storage (BCF, weekly)
    # ── Crude oil / petroleum supply signals ─────────────────────────────────
    "eia_crude_stocks":     "PET.WTTSTUS1.W",         # Weekly US total crude oil stocks (Mbbl)
    "eia_crude_prod":       "PET.WCRFPUS2.W",         # Weekly US crude oil production (Mb/d)
    "eia_refinery_util":    "PET.WPULEUS3.W",         # US refinery utilisation % (weekly)
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def fetch_yfinance(
    tickers: Optional[dict[str, str]] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download adjusted close prices from yfinance.

    Parameters
    ----------
    tickers:
        Mapping of {column_name: yfinance_ticker}.  Defaults to
        ``YFINANCE_TICKERS``.
    start:
        ISO date string for the start of the download window.
    end:
        ISO date string for the end of the window.  Defaults to today.

    Returns
    -------
    pd.DataFrame
        Daily close prices with column names from the ``tickers`` mapping.
    """
    if tickers is None:
        tickers = YFINANCE_TICKERS
    if end is None:
        end = date.today().isoformat()

    raw = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]

    # Rename columns
    rev = {v: k for k, v in tickers.items()}
    prices = prices.rename(columns=rev)

    # Copper is quoted in $/lb — convert to $/t (1 short ton = 2 204.62 lb)
    if "copper_price" in prices.columns:
        prices["copper_price"] = prices["copper_price"] * 2204.62

    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    logger.info("yfinance: downloaded %d rows for %d tickers", len(prices), len(tickers))
    return prices


def fetch_fred(
    series: Optional[dict[str, str]] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    fred_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Download macro series from FRED.

    Parameters
    ----------
    series:
        Mapping of {column_name: fred_series_id}.  Defaults to
        ``FRED_SERIES``.
    start:
        ISO date string.
    end:
        ISO date string.  Defaults to today.
    fred_api_key:
        FRED API key.  If *None* the function tries the environment variable
        ``FRED_API_KEY``; if that is also absent it falls back to synthetic
        random-walk placeholders (useful for offline testing).

    Returns
    -------
    pd.DataFrame
        Daily FRED observations, forward-filled to daily frequency.
    """
    import os

    if series is None:
        series = FRED_SERIES
    if end is None:
        end = date.today().isoformat()

    key = fred_api_key or os.environ.get("FRED_API_KEY")

    frames: dict[str, pd.Series] = {}
    if key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=key)
            for col, sid in series.items():
                try:
                    s = fred.get_series(sid, observation_start=start, observation_end=end)
                    s.name = col
                    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
                    frames[col] = s
                    logger.info("FRED: fetched %s (%d obs)", sid, len(s))
                except Exception as exc:
                    logger.warning("FRED: could not fetch %s — %s", sid, exc)
        except ImportError:
            logger.warning("fredapi not installed; FRED data will be synthetic")
            key = None

    if not key or not frames:
        logger.warning(
            "No FRED API key supplied or fredapi unavailable. "
            "Generating synthetic placeholder series."
        )
        idx = pd.date_range(start, end, freq="D")
        rng = __import__("numpy").random.default_rng(42)
        for col in series:
            frames[col] = pd.Series(
                rng.standard_normal(len(idx)).cumsum() + 100,
                index=idx,
                name=col,
            )

    df = pd.concat(frames.values(), axis=1)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def fetch_alpha_vantage(
    commodities: Optional[dict[str, str]] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Download monthly commodity spot prices from Alpha Vantage.

    Uses the Alpha Vantage Physical Commodity API (COPPER, ALUMINUM, ZINC, …).
    Free tier allows 25 requests/day; all series are monthly and forward-filled
    to daily frequency.

    Parameters
    ----------
    commodities:
        Mapping of {column_name: av_function_name}.  Defaults to
        ``ALPHA_VANTAGE_COMMODITIES``.
    start:
        ISO date string for the earliest observation to keep.
    end:
        ISO date string for the latest observation.  Defaults to today.
    api_key:
        Alpha Vantage API key.  Falls back to ``ALPHA_VANTAGE_API_KEY``
        environment variable.  If absent, the function is skipped and an
        empty DataFrame is returned.

    Returns
    -------
    pd.DataFrame
        Monthly observations forward-filled to daily frequency.
    """
    import os
    import time

    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; skipping Alpha Vantage fetch")
        return pd.DataFrame()

    if commodities is None:
        commodities = ALPHA_VANTAGE_COMMODITIES
    if end is None:
        end = date.today().isoformat()

    key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not key:
        logger.warning("No Alpha Vantage API key — skipping AV commodity fetch")
        return pd.DataFrame()

    base_url = "https://www.alphavantage.co/query"
    frames: dict[str, pd.Series] = {}

    for col, function in commodities.items():
        try:
            resp = requests.get(
                base_url,
                params={"function": function, "interval": "monthly", "apikey": key},
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()

            if "data" not in payload:
                logger.warning("AV %s: unexpected response — %s", function, list(payload.keys()))
                continue

            records = [
                (row["date"], float(row["value"]))
                for row in payload["data"]
                if row.get("value") not in (None, ".", "")
            ]
            if not records:
                logger.warning("AV %s: no usable records", function)
                continue

            idx, vals = zip(*records)
            s = pd.Series(vals, index=pd.DatetimeIndex(idx), name=col)
            s = s.sort_index()
            s = s.loc[start:end]
            frames[col] = s
            logger.info("AV: fetched %s (%d obs)", function, len(s))

            # Respect free-tier rate limit (5 req/min → 12 s between calls)
            time.sleep(12)

        except Exception as exc:
            logger.warning("AV: could not fetch %s — %s", function, exc)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def fetch_eia(
    series: Optional[dict[str, str]] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Download energy / electricity series from the EIA API.

    Uses the EIA v1 series endpoint (``/series/?series_id=…``).  Weekly and
    monthly series are both accepted; all are forward-filled to daily.

    Parameters
    ----------
    series:
        Mapping of {column_name: eia_series_id}.  Defaults to
        ``EIA_COPPER_SERIES``.
    start:
        ISO date string for the earliest observation to keep.
    end:
        ISO date string for the latest observation.  Defaults to today.
    api_key:
        EIA API key (register free at https://www.eia.gov/opendata/).
        Falls back to ``EIA_API_KEY`` environment variable.  If absent,
        returns an empty DataFrame.

    Returns
    -------
    pd.DataFrame
        Forward-filled daily observations.
    """
    import os

    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; skipping EIA fetch")
        return pd.DataFrame()

    if series is None:
        series = EIA_COPPER_SERIES
    if end is None:
        end = date.today().isoformat()

    key = api_key or os.environ.get("EIA_API_KEY")
    if not key:
        logger.warning("No EIA API key — skipping EIA fetch")
        return pd.DataFrame()

    base_url = "https://api.eia.gov/series/"
    frames: dict[str, pd.Series] = {}

    for col, sid in series.items():
        try:
            resp = requests.get(
                base_url,
                params={"api_key": key, "series_id": sid},
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()

            if "series" not in payload or not payload["series"]:
                logger.warning("EIA %s: no series in response", sid)
                continue

            raw_data = payload["series"][0]["data"]  # [[period, value], …]
            records = []
            for period, value in raw_data:
                if value is None:
                    continue
                # EIA periods: monthly "YYYYMM", weekly "YYYYMMDD" or "YYYY-MM-DD"
                p = str(period)
                if len(p) == 6:          # monthly: "202401"
                    dt = pd.to_datetime(p, format="%Y%m")
                elif len(p) == 8:        # weekly/daily compact: "20240101"
                    dt = pd.to_datetime(p, format="%Y%m%d")
                else:
                    dt = pd.to_datetime(p)
                records.append((dt, float(value)))

            if not records:
                logger.warning("EIA %s: no usable records", sid)
                continue

            idx, vals = zip(*sorted(records))
            s = pd.Series(list(vals), index=pd.DatetimeIndex(list(idx)), name=col)
            s = s.loc[start:end]
            frames[col] = s
            logger.info("EIA: fetched %s (%d obs)", sid, len(s))

        except Exception as exc:
            logger.warning("EIA: could not fetch %s — %s", sid, exc)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def load_data(
    start: str = "2010-01-01",
    end: Optional[str] = None,
    fred_api_key: Optional[str] = None,
    include_cot: bool = True,
    nasdaq_api_key: Optional[str] = None,
    alpha_vantage_api_key: Optional[str] = None,
    eia_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch all data sources and return a single aligned DataFrame.

    Steps:
    1. Download yfinance price data.
    2. Download FRED macro data.
    3. Optionally download Alpha Vantage LME metals / commodity prices.
    4. Optionally download EIA electricity & energy supply data.
    5. Optionally download COT positioning data.
    6. Outer-join on date index; forward-fill up to 5 days; drop remaining NaN rows.
    7. Ensure index is sorted ascending.

    Parameters
    ----------
    start:
        Training window start date (ISO format).
    end:
        Training window end date.  Defaults to today.
    fred_api_key:
        Optional FRED API key.
    include_cot:
        If True, attempt to download COT positioning data.
    nasdaq_api_key:
        Optional Nasdaq Data Link API key for COT data.
    alpha_vantage_api_key:
        Optional Alpha Vantage API key for LME base-metals monthly prices.
        Falls back to ``ALPHA_VANTAGE_API_KEY`` env var.
    eia_api_key:
        Optional EIA API key for electricity and energy supply series.
        Falls back to ``EIA_API_KEY`` env var.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with daily frequency and all available columns.
    """
    yf_df = fetch_yfinance(start=start, end=end)
    fred_df = fetch_fred(start=start, end=end, fred_api_key=fred_api_key)

    # Reindex FRED to the yfinance business-day calendar
    fred_daily = fred_df.reindex(yf_df.index, method="ffill")

    df = pd.concat([yf_df, fred_daily], axis=1)

    # Alpha Vantage: LME / World Bank monthly commodity prices
    if alpha_vantage_api_key:
        av_df = fetch_alpha_vantage(start=start, end=end, api_key=alpha_vantage_api_key)
        if not av_df.empty:
            av_daily = av_df.reindex(df.index, method="ffill")
            df = pd.concat([df, av_daily], axis=1)
            logger.info("Alpha Vantage data integrated: %d columns added", av_daily.shape[1])
    else:
        av_df = pd.DataFrame()

    # EIA: electricity consumption & energy supply series
    if eia_api_key:
        eia_df = fetch_eia(start=start, end=end, api_key=eia_api_key)
        if not eia_df.empty:
            eia_daily = eia_df.reindex(df.index, method="ffill")
            df = pd.concat([df, eia_daily], axis=1)
            logger.info("EIA data integrated: %d columns added", eia_daily.shape[1])
    else:
        eia_df = pd.DataFrame()

    # COT positioning data
    if include_cot:
        try:
            from src.cot_data import align_cot_to_daily, fetch_cot_data
            cot = fetch_cot_data(start=start, end=end, api_key=nasdaq_api_key)
            cot_daily = align_cot_to_daily(cot, df.index)
            df = pd.concat([df, cot_daily], axis=1)
            logger.info("COT data integrated: %d columns added", cot_daily.shape[1])
        except Exception as exc:
            logger.warning("COT data unavailable: %s", exc)

    df = df.sort_index()
    df = df.ffill(limit=5)

    # Drop rows where the target is still NaN
    df = df.dropna(subset=["copper_price"])

    logger.info("Combined dataset: %d rows × %d columns", *df.shape)
    return df
