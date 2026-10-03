"""Final regular-season standings for every season from the NHL API:
conference, division, points and the league's own final order, which the
title-odds sim (playoff_sim.py) uses for playoff formats and to check its
tiebreakers. Past seasons don't change, so rerun only to add a season:

    python fetch_divisions.py              # -> nhl_divisions.csv
    python fetch_divisions.py --current    # the daily run: adds the current
                                           # season once its regular season ends

Once a regular season is over the playoff seeds must come from the league's
own final order, never the sim's tiebreak estimate, so the daily run adds
the season as soon as the regular season ends: a warning for GRACE_DAYS,
then the run fails.
"""
import sys
import time

import requests

import pandas as pd

OUT = 'nhl_divisions.csv'
# API name -> SAKIC's (franchise-continuous) name
NAME_MAP = {'Montréal Canadiens': 'Montreal Canadiens', 'Winnipeg Jets (1979)': 'Winnipeg Jets',
            'Phoenix Coyotes': 'Arizona Coyotes', 'Utah Hockey Club': 'Utah Mammoth'}
UA = {'User-Agent': 'Mozilla/5.0'}   # the API 403s Python's default agent
KEEP = ['conferenceName', 'divisionName', 'points', 'gamesPlayed', 'wins', 'losses', 'otLosses', 'ties',
        'regulationWins', 'regulationPlusOtWins', 'goalFor', 'goalAgainst', 'leagueSequence',
        'conferenceSequence', 'divisionSequence', 'wildcardSequence']


GRACE_DAYS = 2


def _season_rows(season, day):
    r = requests.get(f'https://api-web.nhle.com/v1/standings/{day}', headers=UA, timeout=30)
    r.raise_for_status()
    rows = []
    for t in r.json().get('standings', []):
        name = t['teamName']['default']
        row = {'season': int(season), 'team': NAME_MAP.get(name, name)}
        row.update({k: t.get(k) for k in KEEP})
        rows.append(row)
    return rows


def update_current(today=None):
    """Add the current season's final standings once its regular season is
    over (no regular-season games left in nhl_schedule.csv)."""
    g = pd.read_csv('all_nhl_games.csv', usecols=['season', 'date_game', 'home_team_name',
                                                   'home_pts', 'is_playoff_game_flag'])
    season = int(g['season'].max())
    rs = g[(g['season'] == season) & (g['is_playoff_game_flag'] == 0) & g['home_pts'].notna()]
    sched = pd.read_csv('nhl_schedule.csv')
    if rs.empty or len(sched):
        return                                            # regular season not over
    have = pd.read_csv(OUT)
    if (have['season'] == season).any():
        return
    day = rs['date_game'].max()
    teams = set(g.loc[g['season'] == season, 'home_team_name'])
    try:
        rows = _season_rows(season, day)
        got = {r['team'] for r in rows}
        if got != teams:
            raise RuntimeError(f"standings teams don't match the season's teams: "
                               f"{sorted(got ^ teams)}")
        if any(r['leagueSequence'] is None for r in rows):
            raise RuntimeError('standings are missing leagueSequence')
    except Exception as e:
        days = ((today or pd.Timestamp.now()).normalize() - pd.Timestamp(day).normalize()).days
        msg = f"{season} final standings not usable yet: {e}"
        if days > GRACE_DAYS:
            raise RuntimeError(msg + f' ({days} days after the regular season)')
        print(f'::warning::{msg}')
        return
    pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT, index=False)
    print(f'  {season} final standings ({day}) -> {OUT}')


def main():
    g = pd.read_csv('all_nhl_games.csv', usecols=['season', 'date_game', 'is_playoff_game_flag'])
    rs_end = g[g['is_playoff_game_flag'] == 0].groupby('season')['date_game'].max()
    rows = []
    for season, day in rs_end.items():
        r = requests.get(f'https://api-web.nhle.com/v1/standings/{day}', headers=UA, timeout=30)
        r.raise_for_status()
        st = r.json().get('standings', [])
        for t in st:
            name = t['teamName']['default']
            row = {'season': int(season), 'team': NAME_MAP.get(name, name)}
            row.update({k: t.get(k) for k in KEEP})
            rows.append(row)
        print(season, day, len(st))
        time.sleep(0.3)
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == '__main__':
    update_current() if '--current' in sys.argv else main()
