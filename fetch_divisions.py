"""Final regular-season standings for every season from the NHL API:
conference, division, points and the league's own final order, which the
title-odds sim (playoff_sim.py) uses for playoff formats and to check its
tiebreakers. Past seasons don't change, so rerun only to add a season:

    python fetch_divisions.py              # -> nhl_divisions.csv
"""
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
    main()
