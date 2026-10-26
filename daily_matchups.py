"""Daily Matchups: every game of a day, previewed from the ratings going into
it - win probability, SAKIC's line and O/U (projected total goals), and
the stakes (each team's playoff and Cup odds with a win vs with a loss).
After the games: the final score and whether SAKIC's pick was right. The
NHL port of DUNCAN's Daily Matchups (duncan/daily_matchups.py).

Stakes come from the season sim (playoff_sim.py): one run of 10k simulations
from the last snapshot before the day. A day with no snapshot yet this
season (opening night) previews from last season's final ratings. Playoff
games use the series' next game in the sim (2020: round-robin games too).

Win probability: the sim's chance of winning the game, overtime and shootout
included. Before 2006 a game could end tied; the pick uses the chance of
winning given the game doesn't end tied, and a tie scores neither right nor
wrong. The stakes' "with a loss" counts the loser point an
overtime loss earns (2000 on) at the sim's rate for that game.

Line and O/U (fit on 1980-2026 games, leave-one-season-out; line MAE 2.01
goals vs 2.11 pick'em, O/U MAE 1.90 vs 1.92 league average only):
    margin = LINE_LAM * (home rating - away rating) + LINE_HOME (0 in the
             2020 bubble), rounded to 0.5; Pick'em when it names a different
             favorite than the win probability
    total = 2 * mu + TOTAL_B * (both offenses - both defenses)
            + PS_OFFSET in the playoffs
        mu = league goals per team-game this season so far, shrunk toward
        last season's with MU_PRIOR_GAMES pseudo team-games
        PS_OFFSET: playoff games ran 0.33 goals under (playoff MAE 1.89 -> 1.85)
Start times: the NHL API's (sakic.py).
"""
import hashlib
import multiprocessing as _mp
import os
import pickle

import numpy as np
import pandas as pd
from scipy.special import ndtr

import playoff_sim

N_SIMS = 10_000                   # fleet standard since 2026-10-02
LINE_LAM = 0.695
LINE_HOME = 0.349
TOTAL_B = 0.479
PS_OFFSET = -0.331
MU_PRIOR_GAMES = 300


def _win_value(y, pts, ctrl):
    """(mean of y, change in y per extra standings point) for outcome y
    (per-sim bool) and the team's final points in each sim. As DUNCAN's
    value of one more win: regress on final points plus both teams' drift
    offsets, so the slope is the value of luck alone."""
    y = y.astype(float)
    m = float(y.mean())
    X = np.column_stack([np.ones(len(y)), pts] + ([ctrl] if ctrl is not None else []))
    if np.ptp(pts) == 0:
        return m, 0.0
    return m, max(float(np.linalg.lstsq(X, y, rcond=None)[0][1]), 0.0)


def _raw_split(y, won):
    """Playoff games: no drift left, so split the sims by the game's result."""
    y = y.astype(float)
    return (float(y[won].mean()) if won.any() else None,
            float(y[~won].mean()) if (~won).any() else None)


def _around(m, b, pw):
    """Split mean m into (with a win, with a loss) for a +b win-vs-loss effect."""
    clip = lambda v: min(max(v, 0.0), 1.0)
    return clip(m + b * (1 - pw)), clip(m - b * pw)


def _mu_by_day(season_games, prev_mu):
    """League goals per team-game going into each day: this season so far,
    shrunk toward last season's mean."""
    played = season_games[season_games['home_pts'].notna()]
    per_day = played.groupby('date').apply(
        lambda x: (x['home_pts'].sum() + x['visitor_pts'].sum(), 2 * len(x)))
    out, s, n = {}, 0.0, 0
    for d in sorted(set(season_games['date'])):
        out[d] = (s + MU_PRIOR_GAMES * prev_mu) / (n + MU_PRIOR_GAMES)
        if d in per_day.index:
            s += per_day[d][0]; n += per_day[d][1]
    return out


def build_season(season, games, ratings, fmt_record, prev_mu, n_sims=N_SIMS, log=print):
    """games: the season's games (season, date, home, away, home_pts,
    visitor_pts, overtimes, is_tie, is_playoff, start; NaN pts = scheduled).
    ratings: (season, date, name, rating, rank, rating_o, rating_d) for this
    season and the last. fmt_record(season, (w, reg_l, otl, t)): the site's
    era record format. Returns the season's days (list of dicts)."""
    g = games[games['season'] == season].sort_values(['date', 'home', 'away'], kind='stable').reset_index(drop=True)
    snaps = {d: x.set_index('name') for d, x in ratings.groupby('date')}
    cur = sorted(d for d in snaps if ratings.loc[ratings['date'] == d, 'season'].iloc[0] == season)
    prev = sorted(d for d in snaps if d not in set(cur) and d < (cur[0] if cur else pd.Timestamp.max))
    sim_ratings = {d: dict(zip(snaps[d].index, snaps[d]['rating'])) for d in cur}
    # Scheduled (unplayed) rows are the sim's remaining schedule only in the
    # regular season; an unplayed playoff row would read as a played game.
    sim_g = g[(g['is_playoff'] == 0) | g['home_pts'].notna()]
    sim = playoff_sim.SeasonSim(season, sim_g, sim_ratings)
    rs_keys = set(zip(sim.rs['date'], sim.rs['home'], sim.rs['away']))
    mu = _mu_by_day(g, prev_mu)
    otl_pt = playoff_sim.points_rules(season)[0]

    days = sorted(set(g['date']))
    unplayed_days = sorted(set(g.loc[g['home_pts'].isna(), 'date']))
    preview_day = unplayed_days[0] if unplayed_days else None

    days_out, previewed = [], False
    for day in days:
        if previewed:
            break                                     # live season: preview only the next game day
        before = [d for d in cur if d < day]
        if before:
            snap_d = before[-1]
        elif prev:
            snap_d = prev[-1]                         # opening night: last season's final ratings
            sim.ratings[snap_d] = dict(zip(snaps[snap_d].index, snaps[snap_d]['rating']))
        else:
            continue                                  # 1980: no ratings yet
        rt = snaps[snap_d]
        cap = {}
        sim.odds_at(snap_d, n_sims=n_sims, capture=cap)
        idx = {t: i for i, t in enumerate(cap['teams'])}
        made, champ = cap['made'], cap['champ']
        rest = cap.get('rest')
        E, Psim = cap.get('E'), cap.get('P')
        value = {}                                    # team -> per-point values, once per day

        def per_point(t, opp):
            if t not in value:
                it = idx[t]
                ctrl = None if E is None else E[:, [it, idx[opp]]]
                value[t] = (_win_value(made[:, it], Psim[:, it], ctrl),
                            _win_value(champ == it, Psim[:, it], ctrl))
            return value[t]
        ps_next = {frozenset((a, b)): (a, won, bo, rnd) for a, b, won, bo, rnd in cap['ps_next']}

        # Regular-season records going into the day; series scores for playoff games.
        played = g[(g['date'] < day) & g['home_pts'].notna()]
        rec, series = {}, {}
        for x in played.itertuples(index=False):
            tie = x.is_tie == 1 or x.home_pts == x.visitor_pts
            if x.is_playoff == 0:
                ot = isinstance(x.overtimes, str) and x.overtimes in ('OT', 'SO')
                for t, won in ((x.home, x.home_pts > x.visitor_pts), (x.away, x.visitor_pts > x.home_pts)):
                    w, l, o, tt = rec.get(t, (0, 0, 0, 0))
                    if tie:
                        tt += 1
                    elif won:
                        w += 1
                    elif ot:
                        o += 1
                    else:
                        l += 1
                    rec[t] = (w, l, o, tt)
            elif not tie:
                win = x.home if x.home_pts > x.visitor_pts else x.away
                k = frozenset((x.home, x.away))
                series.setdefault(k, {}).setdefault(win, 0)
                series[k][win] += 1

        def record(t):
            r = rec.get(t, (0, 0, 0, 0))
            return fmt_record(season, r) or fmt_record(season, (0, 0, 0, 0)) or ('0-0-0')

        day_g = g[g['date'] == day]
        seen = {}
        games_out = []
        for x in day_g.itertuples(index=False):
            h, a = x.home, x.away
            k_rep = seen.get((h, a), 0)
            seen[(h, a)] = k_rep + 1
            neutral = season == 2020 and x.is_playoff == 1       # the bubble
            blank = pd.Series({'rating': 0.0, 'rating_o': 0.0, 'rating_d': 0.0, 'rank': np.nan})
            rh = rt.loc[h] if h in rt.index else blank
            ra = rt.loc[a] if a in rt.index else blank
            stakes, ser = None, None
            if x.is_playoff == 0:
                if rest is None or (x.date, h, a) not in rs_keys or h not in idx or a not in idx:
                    continue
                j = rest.index[(rest['home'] == h) & (rest['away'] == a) & (rest['date'] == day)]
                if not len(j):
                    continue
                jj = j[min(k_rep, len(j) - 1)]
                hw, aw = cap['hw'][:, jj], cap['aw'][:, jj]
                p_hw, p_aw = float(hw.mean()), float(aw.mean())
                p_home = p_hw / max(p_hw + p_aw, 1e-9)
                stakes = {}
                for t, opp, won, lost, own_pts in ((h, a, hw, aw, cap['hpts'][:, jj]), (a, h, aw, hw, cap['apts'][:, jj])):
                    (po_m, po_b), (ti_m, ti_b) = per_point(t, opp)
                    # A loss still earns the loser point when it goes to overtime.
                    loss_pts = float(own_pts[lost].mean()) if lost.any() else 0.0
                    gap = 2.0 - loss_pts
                    pw = float(won.mean()) / max(float(won.mean()) + float(lost.mean()), 1e-9)
                    po_w, po_l = _around(po_m, po_b * gap, pw)
                    ti_w, ti_l = _around(ti_m, ti_b * gap, pw)
                    stakes['home' if t == h else 'away'] = {
                        'po_win': po_w, 'po_loss': po_l, 'title_win': ti_w, 'title_loss': ti_l}
            else:
                hit = ps_next.get(frozenset((h, a)))
                if hit is None:
                    if pd.isna(x.home_pts):
                        continue                      # an unplayed game after the series ended
                    # A played game the sim has no next game for: Quality only.
                    p_home = float(ndtr(playoff_sim.PA * (rh['rating'] - ra['rating'] + (0.0 if neutral else playoff_sim.PH))))
                else:
                    ta, won_a, best_of, rnd_name = hit
                    sw = series.get(frozenset((h, a)), {})
                    hw = won_a if ta == h else ~won_a
                    p_home = float(hw.mean())
                    stakes = {}
                    for t, won in ((h, hw), (a, ~hw)):
                        ti_w, ti_l = _raw_split(champ == idx[t], won)
                        stakes['home' if t == h else 'away'] = {
                            'po_win': None, 'po_loss': None, 'title_win': ti_w, 'title_loss': ti_l}
                    ser = {'home_wins': sw.get(h, 0), 'away_wins': sw.get(a, 0), 'best_of': int(best_of),
                           'round': rnd_name}
            line = round(float(LINE_LAM * (rh['rating'] - ra['rating']) + (0.0 if neutral else LINE_HOME)) * 2) / 2
            if line != 0 and (line > 0) != (p_home >= 0.5):
                line = 0.0                                  # Pick'em: the line and the pick disagree
            total = 2 * mu.get(day, prev_mu) + TOTAL_B * (
                (rh['rating_o'] + ra['rating_o']) - (rh['rating_d'] + ra['rating_d'])) + (
                PS_OFFSET if x.is_playoff == 1 else 0.0)
            game = {
                'home': h, 'away': a, 'neutral': bool(neutral),
                'home_record': record(h), 'away_record': record(a),
                'home_rank': None if pd.isna(rh['rank']) else int(rh['rank']),
                'away_rank': None if pd.isna(ra['rank']) else int(ra['rank']),
                'p_home': round(p_home, 4),
                'line': line,                               # home by this many (negative = away)
                'total': round(float(total) * 2) / 2,       # SAKIC O/U: projected combined goals
                'kickoff': x.start if isinstance(x.start, str) else None,
                'stakes': None if stakes is None else
                          {k: {kk: (None if vv is None else round(vv, 4)) for kk, vv in v.items()} for k, v in stakes.items()},
                'result': None,
            }
            if ser is not None:
                game['series'] = ser
            if not pd.isna(x.home_pts):
                game['result'] = {'home': int(x.home_pts), 'away': int(x.visitor_pts)}
                if isinstance(x.overtimes, str) and x.overtimes in ('OT', 'SO') and x.home_pts != x.visitor_pts:
                    game['result']['ot'] = x.overtimes
            game['_q'] = (min(rh['rating'], ra['rating']), max(rh['rating'], ra['rating']))
            games_out.append(game)
        # The next game day with a game left on it: an unplayed game after
        # its series ended doesn't count.
        previewed = preview_day is not None and day >= preview_day and bool(games_out)
        if games_out:
            days_out.append({'date': day.strftime('%Y-%m-%d'),
                             'playoffs': bool((day_g['is_playoff'] == 1).all()),
                             'games': games_out})
        log(f'  {season} {day.date()}: {len(games_out)} games')
    return days_out


# ── Juice: Quality x Stakes ──────────────────────────────────────────────────
# As DUNCAN, after DILLON (weekly_matchups.py): Quality = 35% the worse
# team's rating, 35% the better team's, 30% closeness; Stakes = both teams'
# playoff-odds swing + K x their Cup-odds swing, K ramping 4 -> 8 over the
# regular season (playoff games: Cup swing only, K = 8). Each is ranked
# 0-100 and Juice = sqrt(Quality x Stakes).
#
# The ranking pool is COMPLETED seasons only: ranking against the live
# season too would move every historical score every day and rewrite all 47
# season files nightly. Live-season games are placed within the completed
# seasons' distribution instead, so history only re-ranks once a year, when a
# season finishes.
TITLE_K_START, TITLE_K_END = 4.0, 8.0


def _pct(ref_sorted, x):
    """Percentile (0-1) of x within ref, matching pandas rank(pct=True)'s
    average rank for values already in ref."""
    lo = np.searchsorted(ref_sorted, x, side='left')
    hi = np.searchsorted(ref_sorted, x, side='right')
    return ((lo + hi + 1) / 2) / len(ref_sorted)


def add_juice(seasons, pool_seasons):
    """seasons: {season: days (build_season output)}; pool_seasons: the
    completed seasons to rank against. Adds 'quality', 'stakes_score' and
    'juice' (0-100) to every game and drops the internal '_q'."""
    rows = []
    for season, days in seasons.items():
        rs_days = [d['date'] for d in days if not d['playoffs']]
        first, last = (rs_days[0], rs_days[-1]) if rs_days else (None, None)
        span = (pd.Timestamp(last) - pd.Timestamp(first)).days if rs_days else 1
        for d in days:
            if d['playoffs']:
                k = TITLE_K_END
            else:
                frac = (pd.Timestamp(d['date']) - pd.Timestamp(first)).days / max(span, 1)
                k = TITLE_K_START + (TITLE_K_END - TITLE_K_START) * min(frac, 1.0)
            for g in d['games']:
                qmin, qmax = g.pop('_q')
                close = 1 - abs(2 * g['p_home'] - 1)
                stake = None
                if g['stakes'] is not None:
                    st = list(g['stakes'].values())
                    po = sum((s['po_win'] or 0) - (s['po_loss'] or 0) for s in st)
                    ti = sum((s['title_win'] or 0) - (s['title_loss'] or 0) for s in st)
                    stake = po + k * ti
                rows.append((g, season in pool_seasons, qmin, qmax, close, stake))
    if not rows:
        return
    df = pd.DataFrame([r[1:] for r in rows], columns=['pool', 'qmin', 'qmax', 'close', 'stake'])
    ref = df[df['pool']]
    parts = {c: np.sort(ref[c].to_numpy()) for c in ('qmin', 'qmax', 'close')}
    q_raw = sum(w * _pct(parts[c], df[c].to_numpy()) for c, w in (('qmin', .35), ('qmax', .35), ('close', .3)))
    q = _pct(np.sort(q_raw[df['pool'].to_numpy()]), q_raw) * 100
    st_ref = np.sort(ref['stake'].dropna().to_numpy())
    for (g, *_), qq, ss in zip(rows, q, df['stake']):
        g['quality'] = int(round(min(qq, 100)))
        if pd.isna(ss):
            g['stakes_score'] = g['juice'] = None
        else:
            s_pct = min(_pct(st_ref, ss) * 100, 100)
            g['stakes_score'] = int(round(s_pct))
            g['juice'] = int(round(np.sqrt(min(qq, 100) * s_pct)))


# ── Per-season cache ─────────────────────────────────────────────────────────
# Finished seasons never change unless the engine or their inputs do: cache
# each season under a fingerprint of this file + the sim's engine files + the
# season's games (and last season's, for opening night and the scoring
# baseline) + ratings (rounded to 3dp; the ratings engine isn't
# bit-reproducible).
CACHE_DIR = 'matchups_cache'
_HERE = os.path.dirname(os.path.abspath(__file__))
_JOB = {}


def _fingerprint(season, J):
    h = hashlib.sha256()
    for f in ('daily_matchups.py',) + playoff_sim._ENGINE_FILES:
        p = os.path.join(_HERE, f)
        if os.path.exists(p):
            h.update(open(p, 'rb').read())
    g = J['games'][J['games']['season'].isin([season - 1, season])]
    h.update(g.sort_values(list(g.columns), kind='stable', na_position='last').to_csv(index=False).encode())
    r = J['ratings'][J['ratings']['season'].isin([season - 1, season])].sort_values(['date', 'name']).copy()
    for c in ('rating', 'rating_o', 'rating_d'):
        r[c] = r[c].round(3)
    h.update(r.to_csv(index=False).encode())
    h.update(repr(J['prev_mu'].get(season)).encode())
    return h.hexdigest()


def _one(season):
    J = _JOB
    r = J['ratings']
    days = build_season(season, J['games'], r[r['season'].isin([season - 1, season])],
                        J['fmt_record'], J['prev_mu'][season], log=lambda *_: None)
    return season, days


def build_cached(seasons, games, ratings, fmt_record, workers=None, log=print):
    """{season: days} for every season, reusing cached ones. ratings needs
    season, date, name, rating, rank, rating_o, rating_d."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    played = games[games['home_pts'].notna()]
    team_pts = pd.concat([played[['season', 'home_pts']].rename(columns={'home_pts': 'p'}),
                          played[['season', 'visitor_pts']].rename(columns={'visitor_pts': 'p'})])
    season_mu = team_pts.groupby('season')['p'].mean().to_dict()
    first = min(season_mu)
    prev_mu = {s: float(season_mu.get(s - 1, season_mu.get(s, season_mu[first]))) for s in seasons}
    _JOB.update(games=games, ratings=ratings, fmt_record=fmt_record, prev_mu=prev_mu)
    out, todo, sigs = {}, [], {}
    for s in seasons:
        sigs[s] = _fingerprint(s, _JOB)
        path = os.path.join(CACHE_DIR, f'{s}.pkl')
        if os.path.exists(path):
            try:
                old_sig, days = pickle.load(open(path, 'rb'))
                if old_sig == sigs[s]:
                    out[s] = days
                    continue
            except Exception:
                pass
        todo.append(s)
    log(f'  daily matchups: {len(seasons) - len(todo)} seasons from cache, computing {len(todo)}')
    if todo:
        # Most recent first: the live season isn't stuck behind 40 others.
        with _mp.get_context('fork').Pool(workers or os.cpu_count()) as pool:
            for s, days in pool.imap_unordered(_one, sorted(todo, reverse=True)):
                out[s] = days
                pickle.dump((sigs[s], days), open(os.path.join(CACHE_DIR, f'{s}.pkl'), 'wb'))
    return {s: out[s] for s in seasons}
