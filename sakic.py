"""
SAKIC — NHL power ratings (fakeronjan/sakic)

Data sources:
  - Historical: hockey-reference.com (one HTML page per season, two tables on
    page: id="games" for RS, id="games_playoffs" for playoffs). Rate-limited
    (sports-reference family, 20 req / 10 sec).
  - Current season: NHL API (api-web.nhle.com). No rate limits. Pulled daily.

Model:
  - WLS solver with per-connected-component zero-sum anchor (handles potential
    schedule disruptions cleanly — e.g., a future COVID-style regional split).
  - Single league (no AL/NL-style components), so no per-league anchor needed
    unlike GRIFFEY.
  - Variable rolling window = WINDOW_MULTIPLIER × games-per-team-per-season.
  - Linear recency decay across window.

Author note: Named after Joe Sakic (Avalanche/Nordiques, 1988-2009).
"""

from urllib.request import urlopen, Request
import io, json, time, zipfile

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


# =========================================================
# CONSTANTS
# =========================================================

MIN_SEASON = 1980  # 1979-80 (END year of season, matches hockey-ref URL convention)

WINDOW_MULTIPLIER = 1.5

# Modern-era home advantage in goals — empirical ~0.15-0.20. NHL has shrunk
# from ~0.3 in the 1980s as travel/scheduling normalized.
HOME_COURT_ADJUSTMENT = 0.15

# Goal margins cluster tightly around 2-3. Cap at 5 prevents 7-1 blowouts from
# dominating the regression.
MARGIN_TRANSFORM = "cap"
MARGIN_CAP = 5

WEIGHTING_MODE = "wls"

# Re-process the most recent N game-days each run so late-arriving box scores
# (overturned reviews, suspended games) get re-absorbed automatically.
RECOMPUTE_TAIL_DAYS = 7

# Regular-season game count per season. Modern era: 82 games. Exceptions for
# lockouts/COVID. Used for season-aware window sizing.
REGULAR_SEASON_GAMES = {
    **{y: 82  for y in range(1996, 2030)},  # 82-game era (1995-96 onward; exceptions overridden below)
    1980: 80, 1981: 80, 1982: 80, 1983: 80, 1984: 80, 1985: 80,
    1986: 80, 1987: 80, 1988: 80, 1989: 80, 1990: 80, 1991: 80, 1992: 80,
    1993: 84, 1994: 84,                 # 84-game seasons before the 1994-95 lockout
    1995: 48,                           # 1994-95 lockout
    2013: 48,                           # 2012-13 lockout
    2020: 70,                           # 2019-20 stopped early at COVID stoppage (~70 games/team)
    2021: 56,                           # 2020-21 COVID-shortened
}


# =========================================================
# TEAM CODE → CANONICAL NAME
# =========================================================
# hockey-reference uses full team names already, but the NHL API uses 3-letter
# abbreviations. This dict maps API codes → canonical names so the two sources
# can merge cleanly.
#
# Franchise rule (matches DUNCAN / GRIFFEY): same-market rebrands collapse to
# the modern canonical name; relocations between metros are kept SEPARATE.
NHL_API_TEAM = {
    # Modern Atlantic Division
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "DET": "Detroit Red Wings",
    "FLA": "Florida Panthers",
    "MTL": "Montreal Canadiens",
    "OTT": "Ottawa Senators",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    # Modern Metropolitan Division
    "CAR": "Carolina Hurricanes",       # 1997+ (separate from Hartford Whalers)
    "CBJ": "Columbus Blue Jackets",
    "NJD": "New Jersey Devils",         # 1982+ (separate from Colorado Rockies / Kansas City Scouts)
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "WSH": "Washington Capitals",
    # Modern Central Division
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",        # 1995+ (separate from Quebec Nordiques)
    "DAL": "Dallas Stars",              # 1993+ (separate from Minnesota North Stars)
    "MIN": "Minnesota Wild",            # 2000+ expansion (NOT a continuation of North Stars)
    "NSH": "Nashville Predators",
    "STL": "St. Louis Blues",
    "UTA": "Utah Mammoth",              # 2024+ (separate from Arizona Coyotes; UHC 2024-25 only, rebranded Mammoth 2025-26+)
    "WPG": "Winnipeg Jets",             # 2011+ (resumes original 1979-1996 Jets lineage)
    # Modern Pacific Division
    "ANA": "Anaheim Ducks",             # Same franchise as Mighty Ducks of Anaheim (era display)
    "CGY": "Calgary Flames",
    "EDM": "Edmonton Oilers",
    "LAK": "Los Angeles Kings",
    "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    # Historical codes still emitted by API for legacy game records
    "PHX": "Arizona Coyotes",           # 1996-2024 (Phoenix → Arizona same-market; defunct after 2024 move to Utah)
    "ARI": "Arizona Coyotes",
    "ATL": "Atlanta Thrashers",         # 1999-2011 (separate from modern Winnipeg Jets)
}

# Era-aware display: hockey-reference is good about emitting the era-appropriate
# name in the game data (e.g., "Mighty Ducks of Anaheim" 1993-2006, then
# "Anaheim Ducks"). prepare_game_data() runs source names through TEAM_ALIASES
# to collapse same-market rebrands to the modern canonical name. generate_data.py
# re-applies era-correct display labels per row.
TEAM_ALIASES = {
    "Mighty Ducks of Anaheim": "Anaheim Ducks",
    "Phoenix Coyotes":         "Arizona Coyotes",   # Same metro; collapse to modern Arizona name
    "Utah Hockey Club":        "Utah Mammoth",      # Same-market rebrand: UHC 2024-25 only → Mammoth 2025-26+
    # The 2024 Utah move is a relocation: stays separate (mapped above).
}


# =========================================================
# DATA ACQUISITION — hockey-reference (historical)
# =========================================================

USER_AGENT = "Mozilla/5.0 (compatible; SAKIC NHL ratings bot)"


def _http_get(url, timeout=20):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def scrape_season(year):
    """Scrape one season's RS + playoff games from hockey-reference.
    `year` is the END year (1980 = 1979-80 season). Returns a DataFrame with
    columns matching prepare_game_data's expected schema."""
    url = f"https://www.hockey-reference.com/leagues/NHL_{year}_games.html"
    html = _http_get(url)
    soup = BeautifulSoup(html, "lxml")

    frames = []
    for table_id, is_playoff in [("games", 0), ("games_playoffs", 1)]:
        table = soup.find("table", id=table_id)
        if table is None:
            continue
        rows = []
        for tr in table.select("tbody > tr"):
            if "thead" in (tr.get("class") or []):
                continue
            cells = {c["data-stat"]: c.get_text(strip=True) for c in tr.select("th, td")}
            if not cells.get("date_game"):
                continue
            rows.append(cells)
        if rows:
            df = pd.DataFrame(rows)
            df["is_playoff_game_flag"] = is_playoff
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    combined["season"] = year
    return combined


def scrape_history(min_season, max_season, existing_df, sleep_seconds=1.2):
    """Scrape every season in [min_season, max_season] that isn't already
    fully captured in existing_df. Hockey-reference is rate-limited (20/10s
    family), so sleep gently between requests."""
    if len(existing_df):
        captured = set(existing_df["season"].unique())
    else:
        captured = set()
    # Always re-scrape the most recent season (it's in-progress).
    captured.discard(max_season)

    new_frames = []
    for year in range(min_season, max_season + 1):
        if year in captured:
            continue
        print(f"  Scraping {year - 1}-{str(year)[-2:]}...", end=" ", flush=True)
        try:
            df = scrape_season(year)
            print(f"{len(df)} games")
            new_frames.append(df)
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(sleep_seconds)

    if not new_frames:
        return existing_df

    if len(existing_df):
        # Drop existing rows for any season we just re-scraped (handles in-progress
        # current season being re-fetched each run).
        re_scraped = {f["season"].iloc[0] for f in new_frames if len(f)}
        keep = existing_df[~existing_df["season"].isin(re_scraped)].copy()
    else:
        keep = pd.DataFrame()

    combined = pd.concat([keep] + new_frames, axis=0, sort=False).reset_index(drop=True)
    combined.to_csv("loaded_nhl_games.csv", index=False)
    return combined


# =========================================================
# DATA ACQUISITION — NHL API (current-season tail)
# =========================================================

def fetch_nhl_api_season(season_year):
    """Pull current-season games from the NHL API. `season_year` is the END
    year (2026 = 2025-26 season). NHL API uses a single concat 20252026 format
    for season identifiers. We page through dates from the season start to
    today via the schedule endpoint.

    Returns a DataFrame with the same schema scrape_season produces (modulo
    column names — normalized in the merge step).
    """
    nhl_season = int(f"{season_year - 1}{season_year}")
    # Start at the season's regular-season start date (queried once)
    # Use a known-safe pre-RS date and let the schedule cursor walk forward.
    start_date = f"{season_year - 1}-09-15"

    rows = []
    cursor = start_date
    seen_dates = set()
    while True:
        url = f"https://api-web.nhle.com/v1/schedule/{cursor}"
        try:
            data = json.loads(_http_get(url))
        except Exception as e:
            print(f"  NHL API fetch failed at {cursor}: {e}")
            break
        for week in data.get("gameWeek", []):
            d = week.get("date")
            if d in seen_dates:
                continue
            seen_dates.add(d)
            for g in week.get("games", []):
                if g.get("season") != nhl_season:
                    continue
                if g.get("gameType") not in (2, 3):
                    continue  # 2 = regular, 3 = playoffs; skip preseason/exhibition
                if g.get("gameState") not in ("OFF", "FINAL"):
                    continue  # only completed games
                home = g.get("homeTeam", {})
                away = g.get("awayTeam", {})
                hs = home.get("score")
                as_ = away.get("score")
                if hs is None or as_ is None:
                    continue
                home_abbr = home.get("abbrev")
                away_abbr = away.get("abbrev")
                home_name = NHL_API_TEAM.get(home_abbr)
                away_name = NHL_API_TEAM.get(away_abbr)
                if not home_name or not away_name:
                    print(f"  WARN: unmapped NHL API code {home_abbr!r} or {away_abbr!r}")
                    continue
                period_type = g.get("gameOutcome", {}).get("lastPeriodType", "REG")
                rows.append({
                    "date_game":         d,
                    "visitor_team_name": away_name,
                    "home_team_name":    home_name,
                    "visitor_goals":     int(as_),
                    "home_goals":        int(hs),
                    "overtimes":         period_type if period_type != "REG" else "",
                    "is_playoff_game_flag": 1 if g.get("gameType") == 3 else 0,
                    "season":            season_year,
                })
        # Advance cursor
        next_d = data.get("nextStartDate")
        if not next_d or next_d <= cursor or next_d > pd.Timestamp.utcnow().strftime("%Y-%m-%d"):
            break
        cursor = next_d

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def merge_game_sources(historical_df, current_df):
    """Drop historical rows for the current season; replace with API-pulled
    games. Identical to GRIFFEY's hybrid merge approach."""
    if current_df is None or current_df.empty:
        return historical_df
    current_seasons = set(current_df["season"].unique())
    keep = historical_df[~historical_df["season"].isin(current_seasons)].copy()
    return pd.concat([keep, current_df], axis=0, ignore_index=True, sort=False)


# =========================================================
# GAME DATA PREPARATION
# =========================================================

def prepare_game_data(raw_df):
    """Clean and enrich the raw games DataFrame. Output schema matches the
    DUNCAN convention with NHL-specific extras: `overtimes` (REG/OT/SO),
    `is_playoff_game_flag`, `is_tie` (REG ties — pre-2005 only)."""
    df = raw_df.copy()

    # Normalize column names — hockey-ref uses visitor_goals/home_goals;
    # we'll alias to visitor_pts/home_pts to match the fleet pattern.
    if "visitor_goals" in df.columns:
        df["visitor_pts"] = pd.to_numeric(df["visitor_goals"], errors="coerce")
        df["home_pts"]    = pd.to_numeric(df["home_goals"], errors="coerce")
    else:
        df["visitor_pts"] = pd.to_numeric(df["visitor_pts"], errors="coerce")
        df["home_pts"]    = pd.to_numeric(df["home_pts"], errors="coerce")
    df = df.dropna(subset=["visitor_pts", "home_pts", "date_game"]).copy()
    df["visitor_pts"] = df["visitor_pts"].astype(int)
    df["home_pts"]    = df["home_pts"].astype(int)

    # Apply same-market rebrand consolidation
    df["visitor_team_name"] = df["visitor_team_name"].replace(TEAM_ALIASES)
    df["home_team_name"]    = df["home_team_name"].replace(TEAM_ALIASES)

    # overtimes column — hockey-ref text is 'OT', 'SO', or empty. Normalize.
    if "overtimes" not in df.columns:
        df["overtimes"] = ""
    df["overtimes"] = df["overtimes"].fillna("").astype(str).str.upper().str.strip()
    # Map OT (2), OT (3), etc. all to 'OT'
    df.loc[df["overtimes"].str.startswith("OT"), "overtimes"] = "OT"

    # Tie = equal scores at end of game. Pre-2005 era: many games went to OT
    # (overtimes='OT') and still ended tied. Modern era: shootouts produce a
    # winner so this only fires for pre-2005 data.
    df["is_tie"] = (df["home_pts"] == df["visitor_pts"]).astype(int)

    # Margins (home-team perspective)
    df["home_margin"]    = df["home_pts"] - df["visitor_pts"]
    df["visitor_margin"] = -df["home_margin"]

    # Win flags. Ties count as 0 for both — Massey treats raw margin (0) cleanly.
    df["home_win"]    = (df["home_margin"] > 0).astype(int)
    df["visitor_win"] = (df["home_margin"] < 0).astype(int)

    # Date parsing
    df["date_game"] = pd.to_datetime(df["date_game"], errors="coerce")
    df = df.dropna(subset=["date_game"])
    df = df.sort_values("date_game")

    # Drop dupes (gid-equivalent via the natural game key)
    df = df.drop_duplicates(
        subset=["season", "date_game", "home_team_name", "visitor_team_name"], keep="first"
    )

    # Date and game IDs
    df["grouped_date_id"] = df.groupby("date_game").ngroup() + 1
    df["unique_game_id"]  = np.arange(1, len(df) + 1)

    # Result strings (score-first canonical format)
    df["home_wl"]    = np.where(df["home_win"] == 1, "W",
                       np.where(df["is_tie"] == 1, "T", "L"))
    df["visitor_wl"] = np.where(df["visitor_win"] == 1, "W",
                       np.where(df["is_tie"] == 1, "T", "L"))
    # Append " (OT)" / " (SO)" suffix for non-regulation results
    suffix = np.where(df["overtimes"] != "", " (" + df["overtimes"] + ")", "")
    df["home_result"] = (
        df["home_wl"] + " " + df["home_pts"].astype(str) + "-" + df["visitor_pts"].astype(str)
        + " vs. " + df["visitor_team_name"] + suffix
    )
    df["visitor_result"] = (
        df["visitor_wl"] + " " + df["visitor_pts"].astype(str) + "-" + df["home_pts"].astype(str)
        + " @ " + df["home_team_name"] + suffix
    )

    # Output columns (match DUNCAN's all_nba_games.csv shape with NHL extras)
    out_cols = [
        "season", "date_game", "visitor_team_name", "home_team_name",
        "visitor_pts", "home_pts",
        "visitor_margin", "home_margin",
        "visitor_win", "home_win", "is_tie", "overtimes",
        "grouped_date_id", "unique_game_id",
        "home_wl", "visitor_wl",
        "home_result", "visitor_result",
        "is_playoff_game_flag",
    ]
    df = df[out_cols]
    df.to_csv("all_nhl_games.csv", index=False)
    print(f"  Prepared {len(df):,} games, {df['season'].min()}-{df['season'].max()}.")
    return df


# =========================================================
# WLS MASSEY SOLVER (per-connected-component zero-sum)
# =========================================================

def _apply_margin_transform(margin, transform, cap):
    m = np.asarray(margin, dtype=float)
    if transform == "raw":
        return m
    if transform == "cap":
        return np.clip(m, -cap, cap)
    raise ValueError(f"Unknown MARGIN_TRANSFORM: {transform}")


def _connected_components(teams, edges):
    """Union-find. Returns dict team_name -> 0-indexed component id."""
    parent = {t: t for t in teams}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for h, v in edges:
        if h in parent and v in parent:
            union(h, v)
    roots = {}
    comp_map = {}
    for t in teams:
        r = find(t)
        if r not in roots:
            roots[r] = len(roots)
        comp_map[t] = roots[r]
    return comp_map


def _solve_massey(window_df, hca, weighting_mode, margin_transform, margin_cap):
    """Solve Massey ratings on one rolling window with per-component zero-sum
    constraint. Single league (no AL/NL-style split), so this matches DUNCAN
    rather than GRIFFEY's per-(component, league) variant."""
    teams = sorted(set(window_df["home_team_name"]) | set(window_df["visitor_team_name"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    home_pts = window_df["home_pts"].to_numpy(dtype=float)
    visitor_pts = window_df["visitor_pts"].to_numpy(dtype=float)
    weights = window_df["date_weight"].to_numpy(dtype=float)
    home_names = window_df["home_team_name"].to_numpy()
    visitor_names = window_df["visitor_team_name"].to_numpy()

    comp_map = _connected_components(teams, zip(home_names, visitor_names))
    n_components = max(comp_map.values()) + 1 if comp_map else 1
    teams_by_comp = [[] for _ in range(n_components)]
    for t, c in comp_map.items():
        teams_by_comp[c].append(t)

    n_rows = n_games + n_components
    X = np.zeros((n_rows, n_teams))
    y = np.zeros(n_rows)
    w = np.zeros(n_rows)

    raw_margin = home_pts - visitor_pts - hca
    transformed = _apply_margin_transform(raw_margin, margin_transform, margin_cap)

    for i in range(n_games):
        X[i, team_idx[home_names[i]]]    =  1.0
        X[i, team_idx[visitor_names[i]]] = -1.0

    if weighting_mode == "wls":
        y[:n_games] = transformed
        w[:n_games] = weights
    else:
        raise ValueError(f"Unknown WEIGHTING_MODE: {weighting_mode}")

    for c, comp_teams in enumerate(teams_by_comp):
        row = n_games + c
        for t in comp_teams:
            X[row, team_idx[t]] = 1.0
        y[row] = 0.0
        w[row] = 1.0e8

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = pd.DataFrame({"name": teams, "rating": r, "component": [comp_map[t] for t in teams]})
    out["rank"] = out["rating"].rank(ascending=False, method="min").astype(int)
    return out


def _window_for_season(season):
    """Fixed rolling window across all seasons (82 * WINDOW_MULTIPLIER).

    Why fixed not variable: short seasons (1995 lockout 48g, 2013 lockout 48g,
    2020 COVID-shortened 69-71g, 2021 COVID 56g) used to get a proportionally
    shrunk window, which inflated tiny-sample ratings for whoever ran hot in
    those years. A constant 123-game-day window pulls extra lookback from the
    prior season for short years and keeps full seasons unchanged.
    """
    return int(round(82 * WINDOW_MULTIPLIER))


_MIN_WINDOW = min(_window_for_season(s) for s in REGULAR_SEASON_GAMES)


# =========================================================
# RATING LOOP
# =========================================================

def compute_ratings(master_df, existing_ratings_df):
    """Per-game-day ratings using a season-aware rolling window. Skips dates
    already in existing_ratings_df; re-processes the most recent
    RECOMPUTE_TAIL_DAYS to absorb late-arriving data."""
    max_date_id = int(master_df["grouped_date_id"].max())

    if len(existing_ratings_df):
        all_ids = sorted(existing_ratings_df["ranking_id"].unique())
        if len(all_ids) > RECOMPUTE_TAIL_DAYS:
            tail_threshold = all_ids[-RECOMPUTE_TAIL_DAYS]
            n_dropped = int((existing_ratings_df["ranking_id"] >= tail_threshold).sum())
            existing_ratings_df = existing_ratings_df[
                existing_ratings_df["ranking_id"] < tail_threshold
            ].copy()
            print(f"  Re-processing tail {RECOMPUTE_TAIL_DAYS} game-days "
                  f"({n_dropped:,} rows dropped from ratings cache).")
        max_ranked = int(existing_ratings_df["ranking_id"].max()) if len(existing_ratings_df) else -1
        min_ranked = int(existing_ratings_df["ranking_id"].min()) if len(existing_ratings_df) else -1
    else:
        max_ranked, min_ranked = -1, -1

    print("Running SAKIC ratings for new data...")
    new_frames = []

    rid_to_season = (
        master_df.sort_values("grouped_date_id")
                 .drop_duplicates("grouped_date_id", keep="last")
                 .set_index("grouped_date_id")["season"]
                 .to_dict()
    )

    last_printed_ym = None
    for i in range(_MIN_WINDOW, max_date_id + 1):
        if min_ranked <= i <= max_ranked:
            continue

        season = int(rid_to_season.get(i, master_df["season"].max()))
        window_size = _window_for_season(season)
        if i < window_size:
            continue

        window = master_df[
            (master_df["grouped_date_id"] >= i - (window_size - 1))
            & (master_df["grouped_date_id"] <= i)
        ].copy()

        window["date_weight"] = (window["grouped_date_id"] - i + window_size) / window_size

        current_date = window["date_game"].max()

        current_ym = (current_date.year, current_date.month)
        if current_ym != last_printed_ym:
            pct = (i - _MIN_WINDOW) / max(1, max_date_id - _MIN_WINDOW) * 100
            print(f"  Ratings: {current_date.strftime('%B %Y')} ({pct:.0f}% complete)")
            last_printed_ym = current_ym

        try:
            ranked = _solve_massey(
                window, hca=HOME_COURT_ADJUSTMENT, weighting_mode=WEIGHTING_MODE,
                margin_transform=MARGIN_TRANSFORM, margin_cap=MARGIN_CAP,
            )
        except Exception as e:
            print(f"  [skip] grouped_date_id {i} ({current_date.date()}): {e}")
            continue

        ranked["ranking_date"] = current_date
        ranked["ranking_id"]   = i
        ranked["season"]       = season
        new_frames.append(ranked)

    if new_frames:
        ratings_df = pd.concat([existing_ratings_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    else:
        ratings_df = existing_ratings_df
    return ratings_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import os

    # Step 1: load existing scraped CSV cache, scrape any missing seasons
    print("Loading scraped game cache...")
    if os.path.exists("loaded_nhl_games.csv"):
        loaded = pd.read_csv("loaded_nhl_games.csv", low_memory=False)
        print(f"  {len(loaded):,} games cached from {loaded['season'].min()}-{loaded['season'].max()}")
    else:
        loaded = pd.DataFrame()
        print("  (no cache yet — first run will scrape everything)")

    current_season = pd.Timestamp.utcnow().year
    if pd.Timestamp.utcnow().month < 9:
        # Sep is start of NHL season — before that, current season is the
        # one whose end-year matches calendar year.
        max_season = current_season
    else:
        max_season = current_season + 1

    historical = scrape_history(MIN_SEASON, max_season, loaded)

    # Step 2: pull current-season games from NHL API (no rate limit, more up-to-date)
    print(f"\nPulling NHL API for {max_season - 1}-{str(max_season)[-2:]} season...")
    try:
        current = fetch_nhl_api_season(max_season)
        print(f"  {len(current)} current-season games.")
    except Exception as e:
        print(f"  NHL API fetch failed: {e}")
        current = pd.DataFrame()

    # Step 3: merge sources (API overrides hockey-ref for current season)
    merged = merge_game_sources(historical, current)
    print(f"\nMerged: {len(merged):,} total games.")

    # Step 4: prepare master DataFrame
    print("\nPreparing master game frame...")
    master = prepare_game_data(merged)

    # Step 5: load existing ratings cache (incremental)
    if os.path.exists("sakic_ratings.csv.gz"):
        existing = pd.read_csv("sakic_ratings.csv.gz")
        existing["ranking_date"] = pd.to_datetime(existing["ranking_date"])
        print(f"  Loaded {len(existing):,} cached ratings rows.")
    else:
        existing = pd.DataFrame(columns=["ranking_id","ranking_date","season","name","rating","rank","component"])
        print("  No ratings cache yet — will compute from scratch.")

    # Step 6: compute new ratings
    ratings = compute_ratings(master, existing)
    print(f"\nFinal ratings: {len(ratings):,} rows.")

    ratings.to_csv("sakic_ratings.csv.gz", index=False, compression="gzip")
    print("Saved sakic_ratings.csv.gz")

    print("\nLatest standings (top 10):")
    latest_id = ratings["ranking_id"].max()
    latest = ratings[ratings["ranking_id"] == latest_id].sort_values("rank").head(10)
    print(latest[["rank", "name", "rating"]].to_string(index=False))
