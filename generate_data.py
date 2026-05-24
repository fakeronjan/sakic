"""
generate_data.py — reads sakic_ratings.csv.gz + all_nhl_games.csv and writes
JSON files for the SAKIC web frontend. Run after sakic.py. Outputs to docs/data/.

Structure clones DUNCAN/GRIFFEY:
  - current_standings.json  (latest snapshot)
  - teams_index.json        (slug → display_name + conference)
  - seasons_index.json      (year list + first/last date)
  - teams/<slug>.json       (per-team per-snapshot history)
  - seasons/<year>.json     (per-season snapshots)
  - champions.json          (Stanley Cup champs + runners-up with ratings)
  - goat_rs.json / goat_ps.json (top 50 by RS-end / PS-end rating)
"""

import json, os, re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

os.makedirs("docs/data/teams",   exist_ok=True)
os.makedirs("docs/data/seasons", exist_ok=True)


# =========================================================
# TEAM CONFERENCE + DIVISION (era-aware, full 1979-80 → present)
# =========================================================
# Major NHL realignments (season-end-year notation, so season=1982 means
# the 1981-82 season):
#   - 1981 (1980-81): last year of pre-realignment Wales/Campbell. Original
#     four divisions Norris/Adams/Patrick/Smythe but NOT geographic.
#   - 1982 (1981-82): geographic realignment of the same four divisions.
#     Detroit/Toronto/Chicago move to Campbell (Norris); Pittsburgh to Wales
#     (Patrick); New Jersey takes Colorado Rockies' Patrick slot (1983).
#   - 1994 (1993-94): renamed to Eastern/Western conferences with
#     Atlantic/Northeast/Central/Pacific. PIT moved from Patrick to Northeast.
#     Florida Panthers + Anaheim Mighty Ducks expansion.
#   - 1999 (1998-99): expanded to 6 divisions (Southeast + Northwest added).
#     PIT/TOR shuffle. Nashville Predators expansion.
#   - 2014 (2013-14): consolidated to 4 divisions (Atlantic/Metropolitan/
#     Central/Pacific). Detroit + Columbus moved East. Winnipeg/Dallas to West.
#   - 2021 (2020-21): COVID-temporary 4 divisions (East/Central/West/North,
#     Canadian teams pooled into North). No Eastern/Western conferences that
#     year — encoded as conf=div=temp-name.
#   - 2022 (2021-22): back to 4-division layout. Seattle Kraken joins Pacific;
#     Arizona moves from Pacific to Central to make room.
#   - 2025 (2024-25): Arizona Coyotes franchise moves to Utah (Utah Hockey
#     Club → Utah Mammoth rebrand same market). Stays in Central.
#
# Format: team → list of (start_season, end_season_inclusive, conference,
# division) ranges. 9999 = ongoing.
TEAM_DIVISION_HISTORY = {
    # ─── Eastern (originally Adams / now Atlantic-Northeast lineage) ──
    "Boston Bruins": [
        (1980, 1993, "Wales",   "Adams"),
        (1994, 2013, "Eastern", "Northeast"),
        (2014, 2020, "Eastern", "Atlantic"),
        (2021, 2021, "East",    "East"),         # COVID
        (2022, 9999, "Eastern", "Atlantic"),
    ],
    "Buffalo Sabres": [
        (1980, 1993, "Wales",   "Adams"),
        (1994, 2013, "Eastern", "Northeast"),
        (2014, 2020, "Eastern", "Atlantic"),
        (2021, 2021, "East",    "East"),
        (2022, 9999, "Eastern", "Atlantic"),
    ],
    "Detroit Red Wings": [
        (1980, 1981, "Wales",    "Norris"),       # pre-realignment Wales/Norris
        (1982, 1993, "Campbell", "Norris"),       # geographic realignment
        (1994, 2013, "Western",  "Central"),
        (2014, 2020, "Eastern",  "Atlantic"),     # moved East with 2013 realignment
        (2021, 2021, "Central",  "Central"),
        (2022, 9999, "Eastern",  "Atlantic"),
    ],
    "Florida Panthers": [
        (1994, 1998, "Eastern", "Atlantic"),      # joined 1993-94 expansion
        (1999, 2013, "Eastern", "Southeast"),
        (2014, 2020, "Eastern", "Atlantic"),
        (2021, 2021, "Central", "Central"),
        (2022, 9999, "Eastern", "Atlantic"),
    ],
    "Montreal Canadiens": [
        (1980, 1981, "Wales",   "Norris"),
        (1982, 1993, "Wales",   "Adams"),
        (1994, 2013, "Eastern", "Northeast"),
        (2014, 2020, "Eastern", "Atlantic"),
        (2021, 2021, "North",   "North"),
        (2022, 9999, "Eastern", "Atlantic"),
    ],
    "Ottawa Senators": [
        (1993, 1993, "Wales",   "Adams"),         # joined 1992-93
        (1994, 2013, "Eastern", "Northeast"),
        (2014, 2020, "Eastern", "Atlantic"),
        (2021, 2021, "North",   "North"),
        (2022, 9999, "Eastern", "Atlantic"),
    ],
    "Tampa Bay Lightning": [
        (1993, 1993, "Campbell", "Norris"),        # joined 1992-93 in Norris
        (1994, 1998, "Eastern",  "Atlantic"),
        (1999, 2013, "Eastern",  "Southeast"),
        (2014, 2020, "Eastern",  "Atlantic"),
        (2021, 2021, "Central",  "Central"),
        (2022, 9999, "Eastern",  "Atlantic"),
    ],
    "Toronto Maple Leafs": [
        (1980, 1981, "Wales",    "Adams"),
        (1982, 1993, "Campbell", "Norris"),        # moved to Norris in 1982 realignment
        (1994, 1998, "Western",  "Central"),
        (1999, 2013, "Eastern",  "Northeast"),     # moved back to East 1998-99
        (2014, 2020, "Eastern",  "Atlantic"),
        (2021, 2021, "North",    "North"),
        (2022, 9999, "Eastern",  "Atlantic"),
    ],
    # ─── Eastern (originally Patrick / now Metropolitan lineage) ──────
    "Carolina Hurricanes": [
        (1998, 1998, "Eastern", "Northeast"),     # was Hartford until 1996-97; became CAR 1997-98
        (1999, 2013, "Eastern", "Southeast"),
        (2014, 2020, "Eastern", "Metropolitan"),
        (2021, 2021, "Central", "Central"),
        (2022, 9999, "Eastern", "Metropolitan"),
    ],
    "Columbus Blue Jackets": [
        (2001, 2013, "Western", "Central"),       # 2000-01 expansion (Central, Western)
        (2014, 2020, "Eastern", "Metropolitan"),
        (2021, 2021, "Central", "Central"),
        (2022, 9999, "Eastern", "Metropolitan"),
    ],
    "New Jersey Devils": [
        (1983, 1993, "Wales",   "Patrick"),        # franchise (formerly Colorado Rockies) settled in NJ for 1982-83
        (1994, 2013, "Eastern", "Atlantic"),
        (2014, 2020, "Eastern", "Metropolitan"),
        (2021, 2021, "East",    "East"),
        (2022, 9999, "Eastern", "Metropolitan"),
    ],
    "New York Islanders": [
        (1980, 1981, "Campbell", "Patrick"),       # pre-realignment Patrick was a Campbell division
        (1982, 1993, "Wales",    "Patrick"),        # geographic realignment swapped Patrick to Wales
        (1994, 2013, "Eastern",  "Atlantic"),
        (2014, 2020, "Eastern",  "Metropolitan"),
        (2021, 2021, "East",     "East"),
        (2022, 9999, "Eastern",  "Metropolitan"),
    ],
    "New York Rangers": [
        (1980, 1981, "Campbell", "Patrick"),
        (1982, 1993, "Wales",    "Patrick"),
        (1994, 2013, "Eastern",  "Atlantic"),
        (2014, 2020, "Eastern",  "Metropolitan"),
        (2021, 2021, "East",     "East"),
        (2022, 9999, "Eastern",  "Metropolitan"),
    ],
    "Philadelphia Flyers": [
        (1980, 1981, "Campbell", "Patrick"),
        (1982, 1993, "Wales",    "Patrick"),
        (1994, 2013, "Eastern",  "Atlantic"),
        (2014, 2020, "Eastern",  "Metropolitan"),
        (2021, 2021, "East",     "East"),
        (2022, 9999, "Eastern",  "Metropolitan"),
    ],
    "Pittsburgh Penguins": [
        (1980, 1981, "Wales",   "Norris"),         # pre-realignment Wales/Norris
        (1982, 1993, "Wales",   "Patrick"),        # moved to Patrick in 1982
        (1994, 1998, "Eastern", "Northeast"),      # moved Patrick→Northeast in 1994 rename
        (1999, 2013, "Eastern", "Atlantic"),       # moved back to Atlantic in 1998-99
        (2014, 2020, "Eastern", "Metropolitan"),
        (2021, 2021, "East",    "East"),
        (2022, 9999, "Eastern", "Metropolitan"),
    ],
    "Washington Capitals": [
        (1980, 1981, "Campbell", "Patrick"),
        (1982, 1993, "Wales",    "Patrick"),
        (1994, 1998, "Eastern",  "Atlantic"),
        (1999, 2013, "Eastern",  "Southeast"),
        (2014, 2020, "Eastern",  "Metropolitan"),
        (2021, 2021, "East",     "East"),
        (2022, 9999, "Eastern",  "Metropolitan"),
    ],
    # ─── Western (originally Norris / now Central lineage) ────────────
    "Chicago Black Hawks": [
        (1980, 1981, "Campbell", "Smythe"),        # pre-realignment Smythe (Chicago's pre-realignment home)
        (1982, 1987, "Campbell", "Norris"),        # moved to Norris in 1982; renamed Blackhawks for 1986-87 (= season 1987)
    ],
    "Chicago Blackhawks": [
        (1987, 1993, "Campbell", "Norris"),        # renamed from Black Hawks for 1986-87
        (1994, 2013, "Western",  "Central"),
        (2014, 2020, "Western",  "Central"),
        (2021, 2021, "Central",  "Central"),
        (2022, 9999, "Western",  "Central"),
    ],
    "Colorado Avalanche": [
        (1996, 1998, "Eastern", "Northeast"),     # Quebec relocated to COL for 1995-96; inherited NE slot
        (1999, 2013, "Western", "Northwest"),
        (2014, 2020, "Western", "Central"),
        (2021, 2021, "West",    "West"),
        (2022, 9999, "Western", "Central"),
    ],
    "Dallas Stars": [
        (1994, 1998, "Western", "Central"),       # MIN North Stars relocated to DAL for 1993-94; inherited Central slot
        (1999, 2013, "Western", "Pacific"),       # moved to Pacific in 1998-99 reorg
        (2014, 2020, "Western", "Central"),
        (2021, 2021, "Central", "Central"),
        (2022, 9999, "Western", "Central"),
    ],
    "Minnesota Wild": [
        (2001, 2013, "Western", "Northwest"),     # 2000-01 expansion
        (2014, 2020, "Western", "Central"),
        (2021, 2021, "West",    "West"),
        (2022, 9999, "Western", "Central"),
    ],
    "Nashville Predators": [
        (1999, 2013, "Western", "Central"),       # 1998-99 expansion
        (2014, 2020, "Western", "Central"),
        (2021, 2021, "Central", "Central"),
        (2022, 9999, "Western", "Central"),
    ],
    "St. Louis Blues": [
        (1980, 1981, "Campbell", "Smythe"),
        (1982, 1993, "Campbell", "Norris"),
        (1994, 2013, "Western",  "Central"),
        (2014, 2020, "Western",  "Central"),
        (2021, 2021, "West",     "West"),
        (2022, 9999, "Western",  "Central"),
    ],
    "Utah Mammoth": [
        # Arizona Coyotes relocated to Utah for 2024-25; same franchise per
        # leaving-a-market-breaks-history rule? No — Arizona LEFT, so per fleet
        # rule the franchise breaks. Utah is a NEW canonical franchise.
        # 2024-25 played as Utah Hockey Club; rebranded Utah Mammoth 2025-26.
        (2025, 9999, "Western", "Central"),
    ],
    "Winnipeg Jets": [
        (1980, 1981, "Campbell", "Smythe"),        # original Jets, joined 1979-80 from WHA merger
        (1982, 1982, "Campbell", "Norris"),        # briefly Norris in 1981-82 only
        (1983, 1993, "Campbell", "Smythe"),        # back to Smythe
        (1994, 1996, "Western",  "Central"),       # joined Central in 1993 rename; original Jets relocated to PHX for 1996-97
        # 1997-2011: gap (no NHL team in Winnipeg)
        (2012, 2013, "Eastern",  "Southeast"),     # modern Jets inherit ATL Thrashers slot in 2011-12 (= season 2012)
        (2014, 2020, "Western",  "Central"),       # 2013 realignment moved them west
        (2021, 2021, "North",    "North"),
        (2022, 9999, "Western",  "Central"),
    ],
    # ─── Western (originally Smythe / now Pacific lineage) ────────────
    "Anaheim Ducks": [
        (1994, 2013, "Western", "Pacific"),       # joined 1993-94 expansion as Mighty Ducks of Anaheim
        (2014, 2020, "Western", "Pacific"),
        (2021, 2021, "West",    "West"),
        (2022, 9999, "Western", "Pacific"),
    ],
    "Calgary Flames": [
        (1981, 1981, "Campbell", "Patrick"),       # first year in Calgary (1980-81); inherited Atlanta's Patrick slot
        (1982, 1993, "Campbell", "Smythe"),        # joined Smythe in geographic realignment
        (1994, 2013, "Western",  "Pacific"),
        (2014, 2020, "Western",  "Pacific"),
        (2021, 2021, "North",    "North"),
        (2022, 9999, "Western",  "Pacific"),
    ],
    "Edmonton Oilers": [
        (1980, 1993, "Campbell", "Smythe"),        # joined 1979-80 from WHA merger
        (1994, 2013, "Western",  "Pacific"),
        (2014, 2020, "Western",  "Pacific"),
        (2021, 2021, "North",    "North"),
        (2022, 9999, "Western",  "Pacific"),
    ],
    "Los Angeles Kings": [
        (1980, 1981, "Wales",    "Norris"),         # pre-realignment Wales/Norris (oddly)
        (1982, 1993, "Campbell", "Smythe"),
        (1994, 2013, "Western",  "Pacific"),
        (2014, 2020, "Western",  "Pacific"),
        (2021, 2021, "West",     "West"),
        (2022, 9999, "Western",  "Pacific"),
    ],
    "Seattle Kraken": [
        (2022, 9999, "Western", "Pacific"),       # 2021-22 expansion
    ],
    "San Jose Sharks": [
        (1992, 1993, "Campbell", "Smythe"),        # 1991-92 expansion
        (1994, 2013, "Western",  "Pacific"),
        (2014, 2020, "Western",  "Pacific"),
        (2021, 2021, "West",     "West"),
        (2022, 9999, "Western",  "Pacific"),
    ],
    "Vancouver Canucks": [
        (1980, 1981, "Campbell", "Smythe"),
        (1982, 1993, "Campbell", "Smythe"),
        (1994, 1998, "Western",  "Pacific"),
        (1999, 2013, "Western",  "Northwest"),     # moved to Northwest in 1998-99
        (2014, 2020, "Western",  "Pacific"),
        (2021, 2021, "North",    "North"),
        (2022, 9999, "Western",  "Pacific"),
    ],
    "Vegas Golden Knights": [
        (2018, 2020, "Western", "Pacific"),       # 2017-18 expansion
        (2021, 2021, "West",    "West"),
        (2022, 9999, "Western", "Pacific"),
    ],
    # ─── Defunct franchises (separate per relocation policy) ──────────
    "Quebec Nordiques": [
        (1980, 1981, "Wales", "Adams"),            # joined 1979-80 from WHA merger
        (1982, 1993, "Wales", "Adams"),
        (1994, 1995, "Eastern", "Northeast"),      # last 2 seasons before relocating to COL for 1995-96
    ],
    "Hartford Whalers": [
        (1980, 1981, "Wales",   "Norris"),         # joined 1979-80 from WHA merger
        (1982, 1993, "Wales",   "Adams"),
        (1994, 1997, "Eastern", "Northeast"),      # last 4 seasons before relocating to CAR for 1997-98
    ],
    "Atlanta Thrashers": [
        (2000, 2013, "Eastern", "Southeast"),      # 1999-2000 expansion; left for Winnipeg 2011-12
        # NOTE: ATL's last season was 2010-11; the franchise relocated to
        # become modern Winnipeg Jets for 2011-12. ATL has data up to 2012.
    ],
    "Minnesota North Stars": [
        (1980, 1981, "Wales",    "Adams"),         # pre-realignment, oddly in Wales/Adams
        (1982, 1993, "Campbell", "Norris"),        # 1982+ in Norris; relocated to DAL for 1993-94
        (1994, 1994, "Western",  "Central"),       # ghost year
    ],
    "Colorado Rockies": [
        (1980, 1982, "Campbell", "Smythe"),        # NHL franchise; relocated to NJ for 1982-83
    ],
    "Arizona Coyotes": [
        (1997, 1998, "Western", "Central"),       # WPG Jets relocated to Phoenix for 1996-97; was Central
        (1999, 2013, "Western", "Pacific"),       # moved to Pacific in 1998-99 reorg
        (2014, 2020, "Western", "Pacific"),
        (2021, 2021, "West",    "West"),
        (2022, 2024, "Western", "Central"),        # moved Pacific→Central in 2021-22 to make room for SEA
        # Relocated to Utah for 2024-25
    ],
    "Atlanta Flames": [
        (1980, 1981, "Campbell", "Patrick"),       # original Atlanta; relocated to Calgary for 1980-81 (= season 1981)
    ],
}


# Modern conference lookup (legacy fallback for any team not yet in history).
TEAM_CONFERENCE = {
    team: ranges[-1][2] for team, ranges in TEAM_DIVISION_HISTORY.items()
}


# Era-aware display names (same-market rebrands)
SAKIC_TEAM_DISPLAY_HISTORY = {
    "Anaheim Ducks":         [(1994, 2006, "Mighty Ducks of Anaheim"),
                              (2007, 9999, "Anaheim Ducks")],
    # Phoenix→Arizona was a same-market rebrand (collapsed in sakic.py to
    # 'Arizona Coyotes'); era display reflects the historical name
    "Arizona Coyotes":       [(1997, 2014, "Phoenix Coyotes"),
                              (2015, 2024, "Arizona Coyotes")],
    # Utah: arrived 2024-25 as Utah Hockey Club, rebranded to Utah Mammoth for
    # 2025-26+. Same-market rebrand → one canonical franchise, era display.
    "Utah Mammoth":          [(2025, 2025, "Utah Hockey Club"),
                              (2026, 9999, "Utah Mammoth")],
}


def display_name(canonical, season):
    history = SAKIC_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return canonical
    s = int(season)
    for start, end, name in history:
        if start <= s <= end:
            return name
    return canonical


def current_display_name(canonical):
    history = SAKIC_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return canonical
    return history[-1][2]


def historical_display_names(canonical):
    history = SAKIC_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return []
    current = history[-1][2]
    seen = {current}
    out = []
    for _, _, name in reversed(history[:-1]):
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _conf_div(team, season):
    """Era-aware (conference, division) lookup. Falls back to the team's
    most-recent (conf, div) if the season is outside any encoded range."""
    history = TEAM_DIVISION_HISTORY.get(team)
    if not history:
        return ("Other", "Other")
    s = int(season) if season is not None else history[-1][1]
    for start, end, c, d in history:
        if start <= s <= end:
            return (c, d)
    # Outside any range (typically a ghost snapshot from rolling-window math
    # for a season the team didn't actually play). Fall back to closest era.
    if s < history[0][0]:
        return (history[0][2], history[0][3])
    return (history[-1][2], history[-1][3])


def conference(team, season=None):
    return _conf_div(team, season)[0]


def division(team, season=None):
    return _conf_div(team, season)[1]


# =========================================================
# HELPERS
# =========================================================

def clean(val):
    if pd.isna(val):
        return ""
    return str(val)


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def season_label(season):
    """1980 → '1979-80'"""
    s = int(season)
    return f"{s - 1}-{str(s)[-2:]}"


# =========================================================
# DATA LOAD
# =========================================================

print("Reading ratings + games...")
ratings = pd.read_csv("sakic_ratings.csv.gz")
ratings["ranking_date"] = pd.to_datetime(ratings["ranking_date"])
ratings["season"] = ratings["season"].astype(int)

games = pd.read_csv("all_nhl_games.csv", low_memory=False)
games["date_game"] = pd.to_datetime(games["date_game"])
games["season"] = games["season"].astype(int)

# Per fleet relocation policy (leaving a market breaks history; returning doesn't),
# the original Jets (1979-1996) and modern Jets (2011+) are ONE canonical "Winnipeg
# Jets" franchise. Source data already emits both eras as "Winnipeg Jets" with a
# 1996-2011 gap, so no rename needed here.

print(f"  Ratings: {len(ratings):,} rows, {ratings['season'].min()}-{ratings['season'].max()}")
print(f"  Games:   {len(games):,} rows, {games['season'].min()}-{games['season'].max()}")


# =========================================================
# REGULAR-SEASON END DETECTION (per season)
# =========================================================
# Gate on actual completion: rs_end fires only if postseason games exist OR
# enough teams have played their full RS schedule. Mirrors GRIFFEY's gate.

REGULAR_SEASON_GAMES = {
    **{y: 82 for y in range(1996, 2030)},
    1980: 80, 1981: 80, 1982: 80, 1983: 80, 1984: 80, 1985: 80,
    1986: 80, 1987: 80, 1988: 80, 1989: 80, 1990: 80, 1991: 80, 1992: 80,
    1993: 84, 1994: 84,
    1995: 48, 2013: 48, 2020: 70, 2021: 56,
}

rs_end_by_season = {}
rs_games = games[games["is_playoff_game_flag"] == 0]
for s, sub in rs_games.groupby("season"):
    th = REGULAR_SEASON_GAMES.get(int(s), 82)
    home = sub[["date_game", "home_team_name"]].rename(columns={"home_team_name": "team"})
    away = sub[["date_game", "visitor_team_name"]].rename(columns={"visitor_team_name": "team"})
    g = pd.concat([home, away])
    team_totals = g.groupby("team").size()
    n_teams_done = int((team_totals >= th).sum())
    has_ps = ((games["season"] == s) & (games["is_playoff_game_flag"] == 1)).any()
    if has_ps or n_teams_done >= 16:  # ≥ half the league
        rs_end_by_season[int(s)] = sub["date_game"].max()
print(f"\nRS-end dates detected for {len(rs_end_by_season)} seasons.")


# =========================================================
# FLAG RS-end and PS-end ON SNAPSHOTS
# =========================================================

ratings["is_rs_end"] = 0
ratings["is_ps_end"] = 0
ratings["is_playoff_snapshot"] = 0

for s, rs_end in rs_end_by_season.items():
    season_mask = ratings["season"] == s
    season_subset = ratings[season_mask]
    if season_subset.empty:
        continue
    rs_candidates = season_subset[season_subset["ranking_date"] <= rs_end]
    if not rs_candidates.empty:
        rs_end_id = rs_candidates["ranking_id"].max()
        ratings.loc[season_mask & (ratings["ranking_id"] == rs_end_id), "is_rs_end"] = 1
    ps_end_id = season_subset["ranking_id"].max()
    ratings.loc[season_mask & (ratings["ranking_id"] == ps_end_id), "is_ps_end"] = 1
    ratings.loc[season_mask & (ratings["ranking_date"] > rs_end), "is_playoff_snapshot"] = 1


# =========================================================
# PER-TEAM SEASON RECORDS
# =========================================================
# NHL has ties (pre-2005) and OT/SO losses (post-1999, 2-point system since 2005)

def team_game_view(games_df):
    home = games_df[["season", "date_game", "home_team_name", "home_win", "is_tie", "overtimes", "home_result", "is_playoff_game_flag"]].rename(
        columns={"home_team_name": "team", "home_win": "won", "home_result": "result"})
    home["home"] = 1
    away = games_df[["season", "date_game", "visitor_team_name", "visitor_win", "is_tie", "overtimes", "visitor_result", "is_playoff_game_flag"]].rename(
        columns={"visitor_team_name": "team", "visitor_win": "won", "visitor_result": "result"})
    away["home"] = 0
    return pd.concat([home, away], ignore_index=True)

team_games = team_game_view(games)
team_games["is_playoff_game"] = team_games["is_playoff_game_flag"] == 1

# Loss types (regular season only):
#   ot_so_loss = 1 if team lost AND game ended in OT/SO
#   regular_loss = 1 if team lost in regulation
#   tie = 1 if game ended in REG tie (pre-2005)
team_games["ot_so_loss"] = (
    (team_games["won"] == 0)
    & (team_games["is_tie"] == 0)
    & (team_games["overtimes"].isin(["OT", "SO"]))
).astype(int)


def _build_record(sub):
    w = int(sub["won"].sum())
    t = int(sub["is_tie"].sum())
    otl = int(sub["ot_so_loss"].sum())
    total_l = int(((sub["won"] == 0) & (sub["is_tie"] == 0)).sum())
    reg_l = total_l - otl
    return w, reg_l, otl, t


records = {}
for (season, team, is_po), sub in team_games.groupby(["season", "team", "is_playoff_game"]):
    w, reg_l, otl, t = _build_record(sub)
    records.setdefault((int(season), team), {"rs": (0,0,0,0), "ps": (0,0,0,0)})
    key = "ps" if is_po else "rs"
    records[(int(season), team)][key] = (w, reg_l, otl, t)


def format_record(season, tup, is_ps=False):
    """Era-aware record formatter.
       Regular season: era-dependent 3-column (W-L-T pre-1999, W-L-(T+OTL) 1999-2005, W-L-OTL 2005+).
       Postseason: always 2-column W-L (NHL playoffs have no ties or loser points)."""
    w, reg_l, otl, t = tup
    if (w + reg_l + otl + t) == 0:
        return ""
    if is_ps:
        # NHL playoffs: every game has a winner; no OT loss / tie. 2-column.
        return f"{w}-{reg_l + otl}"
    s = int(season)
    if s >= 2006:
        return f"{w}-{reg_l}-{otl}"            # W-L-OTL (modern shootout era)
    if s >= 2000:
        return f"{w}-{reg_l}-{otl + t}"        # W-L-(T+OTL) (1999-2005 loser-point era)
    return f"{w}-{reg_l + otl}-{t}"             # W-L-T (pre-1999, no OTL)


def format_pts(season, tup):
    w, reg_l, otl, t = tup
    return _pts_for(season, w, reg_l, otl, t)


# =========================================================
# PER-SNAPSHOT RECORDS (rolling)
# =========================================================
# For each (team, snapshot_date), compute running record + last_match.
print("\nComputing per-snapshot rolling records + last-match lookups...")
team_games_sorted = team_games.sort_values(["team", "season", "date_game"]).copy()
team_games_sorted["rs_w_inc"]    = (~team_games_sorted["is_playoff_game"] & (team_games_sorted["won"] == 1)).astype(int)
team_games_sorted["rs_l_inc"]    = (~team_games_sorted["is_playoff_game"] & (team_games_sorted["won"] == 0) & (team_games_sorted["is_tie"] == 0) & (team_games_sorted["ot_so_loss"] == 0)).astype(int)
team_games_sorted["rs_otl_inc"]  = (~team_games_sorted["is_playoff_game"] & (team_games_sorted["ot_so_loss"] == 1)).astype(int)
team_games_sorted["rs_t_inc"]    = (~team_games_sorted["is_playoff_game"] & (team_games_sorted["is_tie"] == 1)).astype(int)
team_games_sorted["ps_w_inc"]    = ( team_games_sorted["is_playoff_game"] & (team_games_sorted["won"] == 1)).astype(int)
team_games_sorted["ps_l_inc"]    = ( team_games_sorted["is_playoff_game"] & (team_games_sorted["won"] == 0) & (team_games_sorted["is_tie"] == 0)).astype(int)
# Playoff OT losses still count as just "L" in playoff record (no loser point)

for col in ["rs_w", "rs_l", "rs_otl", "rs_t", "ps_w", "ps_l"]:
    team_games_sorted[f"cum_{col}"] = (
        team_games_sorted.groupby(["team", "season"])[f"{col}_inc"].cumsum().astype(int)
    )

unique_snaps = ratings[["season", "name", "ranking_date"]].drop_duplicates().copy()
unique_snaps = unique_snaps.rename(columns={"name": "team", "ranking_date": "date_game"})

unique_snaps_sorted = unique_snaps.sort_values("date_game").reset_index(drop=True)
team_games_for_merge = team_games_sorted[
    ["team", "season", "date_game", "result", "cum_rs_w", "cum_rs_l", "cum_rs_otl", "cum_rs_t", "cum_ps_w", "cum_ps_l"]
].copy()
team_games_for_merge["actual_game_date"] = team_games_for_merge["date_game"]
team_games_for_merge = team_games_for_merge.sort_values("date_game").reset_index(drop=True)

snap_records = pd.merge_asof(
    unique_snaps_sorted, team_games_for_merge,
    on="date_game", by=["team", "season"],
    direction="backward",
)


def _fmt_rs(r, season):
    w   = int(r["cum_rs_w"])   if pd.notna(r["cum_rs_w"])   else 0
    l   = int(r["cum_rs_l"])   if pd.notna(r["cum_rs_l"])   else 0
    otl = int(r["cum_rs_otl"]) if pd.notna(r["cum_rs_otl"]) else 0
    t   = int(r["cum_rs_t"])   if pd.notna(r["cum_rs_t"])   else 0
    if (w + l + otl + t) == 0:
        return "0-0-0"
    s = int(season)
    if s >= 2006:
        # Modern shootout era: W-L-OTL, T is always 0
        return f"{w}-{l}-{otl}"
    if s >= 2000:
        # 1999-2005: loser-point era. Combine OTL and T into one "loser points" column
        # (both gave 1 pt). Display as W-L-(T+OTL).
        return f"{w}-{l}-{otl + t}"
    # Pre-1999: no loser point. OTL didn't exist as a separate category — folded into L.
    return f"{w}-{l + otl}-{t}"


def _fmt_ps(r):
    pw = int(r["cum_ps_w"]) if pd.notna(r["cum_ps_w"]) else 0
    pl = int(r["cum_ps_l"]) if pd.notna(r["cum_ps_l"]) else 0
    if (pw + pl) == 0:
        return ""
    return f"{pw}-{pl}"   # NHL playoffs: 2-col W-L (every game has a winner)


def _pts_for(season, w, l, otl, t):
    """NHL points: W*2 + (T or OTL = 1 pt). Era rules:
       1979-80 to 1998-99: W*2 + T*1 (no OT-loss point)
       1999-00 to 2004-05: W*2 + OTL*1 + T*1 (loser point introduced 1999)
       2005-06 onward:     W*2 + OTL*1 (no ties; OTL is OT/SO loss)"""
    s = int(season)
    if s >= 2000:  # 1999-00 introduced loser point; modern era continues it
        return w * 2 + otl + t
    return w * 2 + t  # pre-1999


snap_records["rs_record"] = [
    _fmt_rs(r, s) for r, s in zip(snap_records.to_dict("records"), snap_records["season"])
]
snap_records["ps_record"]       = snap_records.apply(_fmt_ps, axis=1)
snap_records["last_match"]      = snap_records["result"].fillna("")
snap_records["last_match_date"] = snap_records["actual_game_date"].dt.strftime("%Y-%m-%d").fillna("")
# Era-aware points for the standings (NHL ranks by points, not wins).
snap_records["rs_pts"] = [
    _pts_for(
        s,
        int(r["cum_rs_w"])   if pd.notna(r["cum_rs_w"])   else 0,
        int(r["cum_rs_l"])   if pd.notna(r["cum_rs_l"])   else 0,
        int(r["cum_rs_otl"]) if pd.notna(r["cum_rs_otl"]) else 0,
        int(r["cum_rs_t"])   if pd.notna(r["cum_rs_t"])   else 0,
    )
    for r, s in zip(snap_records.to_dict("records"), snap_records["season"])
]

snap_record_lookup = {
    (int(r["season"]), r["team"], r["date_game"]): {
        "rs_record":       r["rs_record"],
        "ps_record":       r["ps_record"],
        "rs_pts":          int(r["rs_pts"]),
        "last_match":      r["last_match"],
        "last_match_date": r["last_match_date"],
    }
    for _, r in snap_records.iterrows()
}
print(f"  {len(snap_record_lookup):,} (season, team, snapshot) lookups built.")


# =========================================================
# STANLEY CUP CHAMPIONS (derived from playoff games)
# =========================================================
# Like GRIFFEY's WS walker: the last team to play in the playoff bracket is
# the Cup champion; their opponent in that final series is the runner-up.

print("\nDeriving Stanley Cup champions from playoff games...")
# Gate on the postseason being COMPLETE (not just started). Two checks:
#   (a) The bracket has resolved to a single 2-team series at the end (last-date games == 1 series)
#   (b) The winner of that final series has clinched best-of-7 (>= 4 wins)
#   (c) Today's date is past the typical Cup Final window (Jul 1 of end year)
# All three must hold. Without these, in-progress 2025-26 incorrectly emits a
# conference-final leader as Cup champ. Same in-progress-season bug we've hit
# repeatedly on other sites.
import datetime as _dt
_today = _dt.date.today()
ws_results = {}  # season -> {champion, runner_up, series}
for s, ps_games in games[games["is_playoff_game_flag"] == 1].groupby("season"):
    s_int = int(s)
    # Gate (c): season's Cup Final window must be in the past.
    if _today < _dt.date(s_int, 7, 1):
        continue
    last_date = ps_games["date_game"].max()
    finals_teams = set(ps_games[ps_games["date_game"] == last_date]["home_team_name"]) | \
                   set(ps_games[ps_games["date_game"] == last_date]["visitor_team_name"])
    # Gate (a): exactly two teams on the latest date (a clinching game).
    if len(finals_teams) != 2:
        continue
    a, b = sorted(finals_teams)
    finals_games = ps_games[
        ((ps_games["home_team_name"] == a) & (ps_games["visitor_team_name"] == b)) |
        ((ps_games["home_team_name"] == b) & (ps_games["visitor_team_name"] == a))
    ]
    a_wins = int(((finals_games["home_team_name"] == a) & (finals_games["home_win"] == 1)).sum()
              + ((finals_games["visitor_team_name"] == a) & (finals_games["visitor_win"] == 1)).sum())
    b_wins = len(finals_games) - a_wins
    # Gate (b): winner clinched the best-of-7 series (>= 4 wins).
    if max(a_wins, b_wins) < 4:
        continue
    if a_wins == b_wins:
        continue
    if a_wins > b_wins:
        champ, ru, series = a, b, f"{a_wins}-{b_wins}"
    else:
        champ, ru, series = b, a, f"{b_wins}-{a_wins}"
    # Clinching game score (from champion's POV).
    clincher = finals_games.sort_values("date_game").iloc[-1]
    if clincher["home_team_name"] == champ:
        final_score = f"{int(clincher['home_pts'])}-{int(clincher['visitor_pts'])}"
    else:
        final_score = f"{int(clincher['visitor_pts'])}-{int(clincher['home_pts'])}"
    ws_results[s_int] = {"champion": champ, "runner_up": ru, "series": series, "final_score": final_score}

print(f"  Detected {len(ws_results)} Stanley Cup champions.")
for s in sorted(ws_results)[-5:]:
    info = ws_results[s]
    print(f"    {season_label(s)}: {info['champion']} def. {info['runner_up']} ({info['series']})")


# =========================================================
# DIVISION WINNERS (per season, per team)
# =========================================================
# 2004-05: NHL lockout cancelled the entire season — no games played, no
# division titles, no Presidents' Trophy, no Stanley Cup. Mirrors the GRIFFEY
# 1994 strike handling pattern.
NO_TITLES_SEASONS = {2005}

# Short / disrupted seasons — flagged on GOAT entries so the UI can tag them.
# Pattern matches GRIFFEY's COVID tag for 2020. Small samples bias ratings;
# the tag gives readers context without altering the model.
SHORT_SEASONS = {
    1995: "lockout 48g",   # 1994-95 lockout: 48-game season
    2013: "lockout 48g",   # 2012-13 lockout: 48-game season
    2020: "COVID 70g",     # 2019-20 stopped early at COVID stoppage (~70 g)
    2021: "COVID 56g",     # 2020-21 COVID-shortened: 56 games
}

# Compute division winners for ALL completed seasons (1979-80 onward),
# scoped to each season's (conf, div). NHL points = 2*W + 1*OTL + 1*T.
print("\nComputing division winners...")
COMPLETED_SEASONS = set(rs_end_by_season.keys())
division_winners = set()

rs_only_simple = team_games[~team_games["is_playoff_game"]].copy()
rs_only_simple["pts"] = rs_only_simple["won"] * 2 + rs_only_simple["ot_so_loss"] + rs_only_simple["is_tie"]

# Names that mean "no real division" — skip them.
_GENERIC_DIV_LABELS = {"Other", ""}

for s, sub in rs_only_simple.groupby("season"):
    s = int(s)
    if s not in COMPLETED_SEASONS or s in NO_TITLES_SEASONS:
        continue
    by_team = sub.groupby("team").agg(P=("pts", "sum"), G=("pts", "size")).reset_index()
    by_team["ppg"] = by_team["P"] / by_team["G"]
    by_team["conf"] = by_team["team"].apply(lambda t: conference(t, s))
    by_team["div"]  = by_team["team"].apply(lambda t: division(t, s))
    by_team = by_team[~by_team["div"].isin(_GENERIC_DIV_LABELS)
                      & ~by_team["conf"].isin(_GENERIC_DIV_LABELS)]
    if by_team.empty:
        continue
    # Scope by (conf, div) so two divisions sharing a name in different
    # conferences (none in NHL, but defensive) don't collide.
    for (_cf, _dv), grp in by_team.groupby(["conf", "div"]):
        top = grp.sort_values("ppg", ascending=False).head(1)
        if not top.empty:
            division_winners.add((s, top.iloc[0]["team"]))
print(f"  {len(division_winners)} division titles flagged.")


# =========================================================
# OUTPUT: per-season JSON files
# =========================================================
print("\nWriting per-season JSON files...")

teams_played_by_season = {}
for s_, sub in games.groupby("season"):
    teams_played_by_season[int(s_)] = set(sub["home_team_name"]) | set(sub["visitor_team_name"])

for s in sorted(ratings["season"].unique()):
    s = int(s)
    sdf = ratings[ratings["season"] == s].copy()
    ws_info = ws_results.get(s, {})
    cup_champ = ws_info.get("champion")
    cup_ru    = ws_info.get("runner_up")
    snapshots = []
    played_this_season = teams_played_by_season.get(s, set())
    for rid, rdf in sdf.groupby("ranking_id"):
        rdf = rdf[rdf["name"].isin(played_this_season)].copy() if played_this_season else rdf
        rdf = rdf.sort_values("rank")
        snap_date_ts = rdf["ranking_date"].iloc[0]
        snap_date = str(snap_date_ts.date())
        is_rs_end = int(rdf["is_rs_end"].iloc[0])
        is_ps_end = int(rdf["is_ps_end"].iloc[0])
        is_playoff_snap = int(rdf["is_playoff_snapshot"].iloc[0])
        label = None
        if is_rs_end:
            label = "End of regular season"
        elif is_ps_end and is_playoff_snap:
            label = "End of playoffs"
        teams_snap = []
        for _, r in rdf.iterrows():
            rec = snap_record_lookup.get((s, r["name"], snap_date_ts), {})
            finals_status = 2 if r["name"] == cup_champ else (1 if r["name"] == cup_ru else 0)
            div_winner = 1 if (s, r["name"]) in division_winners else 0
            teams_snap.append({
                "rank":             int(r["rank"]) if not pd.isna(r["rank"]) else None,
                "team":             r["name"],
                "display_name":     display_name(r["name"], s),
                "conference":       conference(r["name"], s),
                "division":         division(r["name"], s),
                "rating":           round(float(r["rating"]), 3),
                "regular_record":   rec.get("rs_record", "0-0-0"),
                "regular_pts":      rec.get("rs_pts", 0),
                "playoff_record":   rec.get("ps_record", ""),
                "last_match":       rec.get("last_match", ""),
                "last_match_date":  rec.get("last_match_date", ""),
                "finals_status":    finals_status,
                "division_winner":  div_winner,
            })
        snapshots.append({
            "date":            snap_date,
            "label":           label,
            "is_rs_end":       is_rs_end,
            "is_ps_end":       is_ps_end,
            "is_playoff_snap": is_playoff_snap,
            "teams":           teams_snap,
        })

    snapshots.sort(key=lambda x: x["date"])
    with open(f"docs/data/seasons/{s}.json", "w") as f:
        json.dump({"season": s, "season_label": season_label(s), "snapshots": snapshots}, f, separators=(",", ":"))


# =========================================================
# OUTPUT: per-team JSON files
# =========================================================
print("Writing per-team JSON files...")
teams_index = []
for team in sorted(ratings["name"].unique()):
    tdf = ratings[ratings["name"] == team].copy()
    team_slug = slug(team)
    seasons_dict = {}
    for s, sub in tdf.groupby("season"):
        s = int(s)
        if team not in teams_played_by_season.get(s, set()):
            continue
        ws_info = ws_results.get(s, {})
        finals_status = 2 if ws_info.get("champion") == team else (1 if ws_info.get("runner_up") == team else 0)
        entries = []
        div_winner = 1 if (s, team) in division_winners else 0
        for _, r in sub.sort_values("ranking_date").iterrows():
            snap_date_ts = r["ranking_date"]
            rec = snap_record_lookup.get((s, team, snap_date_ts), {})
            sf = 2 if int(r["is_ps_end"]) == 1 else (1 if int(r["is_rs_end"]) == 1 else 0)
            # Filter to game-days for THIS team (last_match_date == snap date)
            # PLUS season-flag rows (EOR / EOS). Matches DUNCAN's data-layer
            # filter — keeps the file small AND avoids stale-row pollution in
            # the single-season Team Summary view.
            snap_date_str = str(snap_date_ts.date())
            is_game_day = rec.get("last_match_date") == snap_date_str
            if not (is_game_day or sf != 0):
                continue
            entries.append({
                "date":             snap_date_str,
                "rank":             int(r["rank"]) if not pd.isna(r["rank"]) else None,
                "rating":           round(float(r["rating"]), 3),
                "display_name":     display_name(team, s),
                "conference":       conference(team, s),
                "division":         division(team, s),
                "regular_record":   rec.get("rs_record", "0-0-0"),
                "regular_pts":      rec.get("rs_pts", 0),
                "playoff_record":   rec.get("ps_record", ""),
                "last_match":       rec.get("last_match", ""),
                "last_match_date":  rec.get("last_match_date", ""),
                "finals_status":    finals_status,
                "division_winner":  div_winner,
                "season_flag":      sf,
            })
        seasons_dict[str(s)] = entries
    with open(f"docs/data/teams/{team_slug}.json", "w") as f:
        json.dump({
            "team":         team,
            "display_name": current_display_name(team),
            "conference":   conference(team),
            "seasons":      seasons_dict,
        }, f, separators=(",", ":"))

    teams_index.append({
        "name":              team,
        "display_name":      current_display_name(team),
        "conference":        conference(team),
        "historical_names":  historical_display_names(team),
        "slug":              team_slug,
    })

with open("docs/data/teams_index.json", "w") as f:
    json.dump(teams_index, f, separators=(",", ":"))


# =========================================================
# OUTPUT: champions.json
# =========================================================
print("Writing champions.json...")
rec_lookup = {(s, t): records.get((s, t)) for (s, t) in records}

champs_list = []
for s in sorted(ws_results.keys(), reverse=True):
    info = ws_results[s]
    ch_rs = ratings[(ratings["season"] == s) & (ratings["name"] == info["champion"]) & (ratings["is_rs_end"] == 1)]
    ch_ps = ratings[(ratings["season"] == s) & (ratings["name"] == info["champion"]) & (ratings["is_ps_end"] == 1)]
    ru_rs = ratings[(ratings["season"] == s) & (ratings["name"] == info["runner_up"]) & (ratings["is_rs_end"] == 1)]
    ru_ps = ratings[(ratings["season"] == s) & (ratings["name"] == info["runner_up"]) & (ratings["is_ps_end"] == 1)]
    rec_ch = rec_lookup.get((s, info["champion"]))
    rec_ru = rec_lookup.get((s, info["runner_up"]))
    pre_rated = ch_rs.empty and ch_ps.empty
    champs_list.append({
        "season":       s,
        "season_label": season_label(s),
        "series":       info["series"],
        "final_score":  info.get("final_score", ""),
        "pre_rated":    pre_rated,
        "champion": {
            "team":           info["champion"],
            "display_name":   display_name(info["champion"], s),
            "conference":     conference(info["champion"], s),
            "rs_record":      format_record(s, rec_ch["rs"]) if rec_ch else "",
            "rs_pts":         format_pts(s, rec_ch["rs"]) if rec_ch else 0,
            "ps_record":      format_record(s, rec_ch["ps"], is_ps=True) if rec_ch else "",
            "rs_end_rating":  round(float(ch_rs["rating"].iloc[0]), 3) if not ch_rs.empty else None,
            "rs_end_rank":    int(ch_rs["rank"].iloc[0]) if not ch_rs.empty else None,
            "ps_end_rating":  round(float(ch_ps["rating"].iloc[0]), 3) if not ch_ps.empty else None,
            "ps_end_rank":    int(ch_ps["rank"].iloc[0]) if not ch_ps.empty else None,
        },
        "runner_up": {
            "team":           info["runner_up"],
            "display_name":   display_name(info["runner_up"], s),
            "conference":     conference(info["runner_up"], s),
            "rs_record":      format_record(s, rec_ru["rs"]) if rec_ru else "",
            "rs_pts":         format_pts(s, rec_ru["rs"]) if rec_ru else 0,
            "ps_record":      format_record(s, rec_ru["ps"], is_ps=True) if rec_ru else "",
            "rs_end_rating":  round(float(ru_rs["rating"].iloc[0]), 3) if not ru_rs.empty else None,
            "rs_end_rank":    int(ru_rs["rank"].iloc[0]) if not ru_rs.empty else None,
            "ps_end_rating":  round(float(ru_ps["rating"].iloc[0]), 3) if not ru_ps.empty else None,
            "ps_end_rank":    int(ru_ps["rank"].iloc[0]) if not ru_ps.empty else None,
        },
    })

# 2004-05: NHL lockout cancelled the entire season — no Cup awarded. Inject a
# synthetic entry (mirrors GRIFFEY's 1994 strike handling).
if not any(e["season"] == 2005 for e in champs_list):
    champs_list.append({
        "season":       2005,
        "season_label": season_label(2005),
        "series":       "",
        "no_series":    True,
        "pre_rated":    False,
        "champion":     None,
        "runner_up":    None,
    })
    champs_list.sort(key=lambda e: -e["season"])

# Pre-1980 Stanley Cup totals (NHL era 1918-1979 + relevant Cup era 1893+).
# Only counts titles won under the franchise's CURRENT city/name per fleet
# relocation policy. Same-market rebrands (Toronto Arenas → St. Patricks →
# Maple Leafs; Detroit Cougars → Falcons → Red Wings) ARE folded into the
# modern franchise. Defunct franchises (Montreal Maroons, original Ottawa
# Senators 1893-1934, Victoria Cougars, etc.) are omitted entirely.
PRE_1980_CHAMPIONSHIPS = {
    "Toronto Maple Leafs":  13,  # 1918 (Arenas), 1922 (St. Patricks), 1932, 1942, 1945, 1947, 1948, 1949, 1951, 1962, 1963, 1964, 1967
    "Montreal Canadiens":   21,  # 1924, 1930, 1931, 1944, 1946, 1953, 1956, 1957, 1958, 1959, 1960, 1965, 1966, 1968, 1969, 1971, 1973, 1976, 1977, 1978, 1979
    "Detroit Red Wings":     7,  # 1936, 1937, 1943, 1950, 1952, 1954, 1955
    "Chicago Blackhawks":    3,  # 1934, 1938, 1961
    "New York Rangers":      3,  # 1928, 1933, 1940
    "Boston Bruins":         5,  # 1929, 1939, 1941, 1970, 1972
    "Philadelphia Flyers":   2,  # 1974, 1975
}

PRE_1980_RUNNER_UPS = {
    "Detroit Red Wings":    13,  # 1934, 1941, 1942, 1945, 1948, 1949, 1954, 1955, 1956, 1961, 1963, 1964, 1966
    "Boston Bruins":        10,  # 1927, 1930, 1943, 1946, 1953, 1957, 1958, 1974, 1977, 1978
    "Toronto Maple Leafs":   8,  # 1933, 1935, 1936, 1938, 1939, 1940, 1959, 1960
    "Montreal Canadiens":    7,  # 1925, 1947, 1951, 1952, 1954, 1955, 1967
    "Chicago Blackhawks":    6,  # 1931, 1944, 1962, 1965, 1971, 1973
    "New York Rangers":      6,  # 1929, 1932, 1937, 1950, 1972, 1979
    "St. Louis Blues":       3,  # 1968, 1969, 1970
    "Philadelphia Flyers":   1,  # 1976
    "Buffalo Sabres":        1,  # 1975
}

# Running title counts seeded with pre-1980 totals; walk oldest-first.
_champ_count = dict(PRE_1980_CHAMPIONSHIPS)
_ru_count    = dict(PRE_1980_RUNNER_UPS)
for entry in reversed(champs_list):
    if entry.get("no_series"):
        continue  # 2004-05 lockout — no champion/runner-up to count
    ct = entry["champion"]["team"]
    rt = entry["runner_up"]["team"]
    _champ_count[ct] = _champ_count.get(ct, 0) + 1
    _ru_count[rt]    = _ru_count.get(rt, 0) + 1
    entry["champion"]["title_count"]      = _champ_count[ct]
    entry["runner_up"]["runner_up_count"] = _ru_count[rt]

with open("docs/data/champions.json", "w") as f:
    json.dump({"NHL": champs_list}, f, separators=(",", ":"))


# =========================================================
# OUTPUT: goat_rs.json + goat_ps.json
# =========================================================
print("Writing GOAT lists...")
GOAT_TOP_N = 50
cup_seasons = set(ws_results.keys())


def build_goat(flag_col, require_cup=False):
    rows = ratings[(ratings[flag_col] == 1) & (ratings["season"].isin(cup_seasons))].copy()
    rows = rows[rows.apply(
        lambda r: r["name"] in teams_played_by_season.get(int(r["season"]), set()), axis=1
    )]
    if require_cup:
        cup_teams = {(s, info["champion"]) for s, info in ws_results.items()} | \
                    {(s, info["runner_up"]) for s, info in ws_results.items()}
        rows = rows[rows.apply(lambda r: (int(r["season"]), r["name"]) in cup_teams, axis=1)]
    rows = rows.sort_values("rating", ascending=False).reset_index(drop=True)
    top = rows.head(GOAT_TOP_N)
    out = []
    for i, r in top.iterrows():
        s = int(r["season"])
        rec = rec_lookup.get((s, r["name"]))
        info = ws_results.get(s, {})
        finals_status = 2 if info.get("champion") == r["name"] else (1 if info.get("runner_up") == r["name"] else 0)
        out.append({
            "rank":             i + 1,
            "team":             r["name"],
            "display_name":     display_name(r["name"], s),
            "conference":       conference(r["name"], s),
            "division":         division(r["name"], s),
            "division_winner":  1 if (s, r["name"]) in division_winners else 0,
            "season":           s,
            "season_label":     season_label(s),
            "short_season":     s in SHORT_SEASONS,
            "short_season_tag": SHORT_SEASONS.get(s, ""),
            "rating":           round(float(r["rating"]), 3),
            "regular_record":   format_record(s, rec["rs"]) if rec else "",
            "regular_pts":      format_pts(s, rec["rs"]) if rec else 0,
            "playoff_record":   format_record(s, rec["ps"], is_ps=True) if rec else "",
            "finals_status":    finals_status,
        })
    return out


goat_rs = build_goat("is_rs_end", require_cup=False)
goat_ps = build_goat("is_ps_end", require_cup=True)

with open("docs/data/goat_rs.json", "w") as f:
    json.dump(goat_rs, f, separators=(",", ":"))
with open("docs/data/goat_ps.json", "w") as f:
    json.dump(goat_ps, f, separators=(",", ":"))


# =========================================================
# OUTPUT: current_standings.json + seasons_index.json
# =========================================================
latest_id = ratings["ranking_id"].max()
latest_snap = ratings[ratings["ranking_id"] == latest_id].copy()
latest_season = int(latest_snap["season"].iloc[0])
latest_date = str(latest_snap["ranking_date"].iloc[0].date())
latest_played = teams_played_by_season.get(latest_season, set())
latest_snap = latest_snap[latest_snap["name"].isin(latest_played)] if latest_played else latest_snap

current_teams = []
for _, r in latest_snap.sort_values("rank").iterrows():
    current_teams.append({
        "rank":         int(r["rank"]) if not pd.isna(r["rank"]) else None,
        "team":         r["name"],
        "display_name": display_name(r["name"], latest_season),
        "conference":   conference(r["name"], latest_season),
        "division":     division(r["name"], latest_season),
        "rating":       round(float(r["rating"]), 3),
    })

with open("docs/data/current_standings.json", "w") as f:
    json.dump({
        "updated":      latest_date,
        "season":       latest_season,
        "season_label": season_label(latest_season),
        "teams":        current_teams,
    }, f, separators=(",", ":"))

seasons_index = sorted({int(s) for s in ratings["season"].unique()}, reverse=True)
with open("docs/data/seasons_index.json", "w") as f:
    json.dump({
        "seasons":        seasons_index,
        "season_labels":  {s: season_label(s) for s in seasons_index},
        "first_date":     str(games["date_game"].min().date()),
        "last_date":      str(games["date_game"].max().date()),
        "generated_at":   datetime.now(timezone.utc).isoformat(),
    }, f, separators=(",", ":"))


# =========================================================
# DONE
# =========================================================
print(f"\nDone. {len(teams_index)} teams, {len(seasons_index)} seasons, "
      f"{len(ws_results)} Cups detected, {len(goat_rs)} GOAT-RS, {len(goat_ps)} GOAT-PS.")
print(f"\nGOAT-PS top 10:")
for t in goat_ps[:10]:
    marker = " ★" if t["finals_status"] == 2 else " (RU)" if t["finals_status"] == 1 else ""
    print(f"  #{t['rank']:2d}. {t['display_name']} {t['season_label']}  ({t['regular_record']}, {t['rating']:+.2f}){marker}")
print(f"\nGOAT-RS top 10:")
for t in goat_rs[:10]:
    marker = " ★" if t["finals_status"] == 2 else " (RU)" if t["finals_status"] == 1 else ""
    print(f"  #{t['rank']:2d}. {t['display_name']} {t['season_label']}  ({t['regular_record']}, {t['rating']:+.2f}){marker}")
