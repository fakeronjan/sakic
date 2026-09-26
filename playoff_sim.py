"""SAKIC title odds: Monte Carlo of the rest of the NHL season + playoffs.

For every rating snapshot, simulate the remaining regular-season games,
build that season's standings (points, loser points, ties), seed its playoff
format and play the bracket. Anything already played is fixed.

Game model (fit on 1981-2026 games, pre-game snapshot ratings, per era):
  regular season, by outcome (standings only need the outcome, and
  independent-Poisson goals under-predict regulation ties by ~25%):
    P(tied after regulation) = logistic(T0 + T1 * |d + h|)
    P(home wins | decided in regulation) = Phi(A * (d + h))
    tied: settled in OT/shootout with P = OT_DECIDED (else a tie),
          home wins it with P = Phi(OA * d + OB)
  playoffs (every game decided): P(home) = Phi(PA * (d + PH))
  d = home rating - away rating.
Early ratings are uncertain: each simulation offsets every team's rating for
the rest of the season by a random amount (drift_sd), fit to how the rest
of past seasons actually went given that date's ratings.

Formats: 1980-81 top 16 overall, reseeded every round; 1982-93 top 4 per
division (Bo5 first round through 1986); 1994-2013 conference seeds 1-8
(division winners first), reseeded each round; 2014-19 and 2022+ division
top 3 + two wild cards; 2020 bubble (top-4 round robin + Bo5 qualifiers,
reseeded, neutral site); 2021 realigned divisions, final four seeded by
points.
"""
import hashlib
import multiprocessing as _mp
import os as _os
import pickle

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr

N_SIMS = 10_000
N_SIMS_PLAYOFFS = 100_000
CHUNK = 200_000

# (first season, T0, T1, A, h): regulation outcome model, fit per era.
REG_PARAMS = [
    (1980, -1.486, -0.064, 0.415, 0.862),
    (1984, -1.432, -0.108, 0.359, 0.733),
    (1994, -1.195, -0.130, 0.358, 0.436),
    (2006, -1.104, -0.098, 0.329, 0.481),
    (2016, -1.144, -0.136, 0.344, 0.365),
]
# (first season, share of regulation ties settled, OA, OB): overtime.
OT_PARAMS = [
    (1980, 0.0, 0.0, 0.0),        # no regular-season overtime
    (1984, 0.329, 0.195, 0.098),  # 5-minute OT; a tie otherwise
    (1994, 0.329, 0.228, 0.086),
    (2000, 0.460, 0.228, 0.086),  # 4-on-4 OT (+ loser point)
    (2006, 1.0, 0.065, 0.017),    # shootout: no ties
    (2016, 1.0, 0.087, 0.031),    # 3-on-3 OT
]
PA, PH = 0.287, 0.420             # playoff game model
# Rating uncertainty for the rest of the season, by share of the regular
# season left (per-team SD, goals).
DRIFT_LEFT = [0.0, 0.52, 0.68, 0.83, 0.96, 1.0]
DRIFT_SD = [0.0, 0.0, 0.488, 0.983, 1.666, 1.666]

BUBBLE_R1 = pd.Timestamp('2020-08-11')   # 2020: first round after the round robin/qualifiers
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DIV = pd.read_csv(_os.path.join(_HERE, 'nhl_divisions.csv'))


def _era(table, season):
    row = table[0]
    for r in table:
        if season >= r[0]:
            row = r
    return row[1:]


# Each team's random offset leans toward the league average by this share of
# its distance from it (0 = pure noise).
DRIFT_LEAN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def drift_sd(frac_left):
    return float(np.interp(frac_left, DRIFT_LEFT, DRIFT_SD))


def drift_lean(frac_left):
    return float(np.interp(frac_left, DRIFT_LEFT, DRIFT_LEAN))


def fmt(season):
    if season <= 1981:
        return 'overall16'
    if season <= 1993:
        return 'division'
    if season <= 2013:
        return 'conference'
    if season == 2020:
        return 'bubble'
    if season == 2021:
        return 'div2021'
    return 'wildcard'


def round_names(season):
    if fmt(season) == 'bubble':
        return (['Qualifiers', 'First Round', 'Second Round', 'Conference Finals', 'Stanley Cup Final'],
                ['Qual', 'R1', 'R2', 'Conf Final', 'Final'])
    if fmt(season) == 'division':
        return (['Division Semifinals', 'Division Finals', 'Conference Finals', 'Stanley Cup Final'],
                ['R1', 'R2', 'Conf Final', 'Final'])
    if fmt(season) == 'div2021':
        return (['First Round', 'Second Round', 'Semifinals', 'Stanley Cup Final'],
                ['R1', 'R2', 'Semis', 'Final'])
    return (['First Round', 'Second Round', 'Conference Finals', 'Stanley Cup Final'],
            ['R1', 'R2', 'Conf Final', 'Final'])


def points_rules(season):
    """(loser point for an OT/shootout loss, tiebreak wins column)."""
    otl = 1 if season >= 2000 else 0
    wt = 'RW' if season >= 2020 else ('ROW' if season >= 2011 else 'W')
    return otl, wt


def host_pattern(bo):
    return {7: [1, 1, 0, 0, 1, 0, 1], 5: [1, 1, 0, 0, 1], 1: [1]}[bo]


class SeasonSim:
    def __init__(self, season, games, ratings, schedule=None):
        """games: the season's games (date, home, away, home_pts, visitor_pts,
        overtimes, is_tie, is_playoff). schedule: unplayed regular-season
        games (date, home, away)."""
        self.season = season
        self.fmt = fmt(season)
        g = games.sort_values('date', kind='stable').reset_index(drop=True)
        rs = g[g['is_playoff'] == 0][['date', 'home', 'away', 'home_pts', 'visitor_pts', 'overtimes', 'is_tie']]
        if schedule is not None and len(schedule):
            rs = pd.concat([rs, schedule.assign(home_pts=np.nan, visitor_pts=np.nan, overtimes=np.nan, is_tie=0)],
                           ignore_index=True)
        dv = _DIV[_DIV['season'] == season]
        if dv.empty:                                   # current season: last known alignment
            dv = _DIV[_DIV['season'] == _DIV['season'].max()]
        self.teams = sorted(set(rs['home']) | set(rs['away']))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        conf = dict(zip(dv['team'], dv['conferenceName'].fillna('League')))
        div = dict(zip(dv['team'], dv['divisionName']))
        self.conf = np.array([conf.get(t, 'League') for t in self.teams], dtype=object)
        self.div = np.array([div.get(t, '') for t in self.teams], dtype=object)
        # The league's own final order: breaks exact ties in the sims.
        seq = dict(zip(dv['team'], dv['leagueSequence'])) if season in set(_DIV['season']) else {}
        T = len(self.teams)
        self.static = np.array([T + 1 - seq.get(t, T + 1) for t in self.teams], float)
        rs = rs.assign(h=rs['home'].map(self.idx), a=rs['away'].map(self.idx)).sort_values('date', kind='stable')
        self.rs = rs.reset_index(drop=True)
        self.ratings = ratings
        self.otl, self.wtype = points_rules(season)

        # Real playoff series, by (pair, round).
        ps = g[g['is_playoff'] == 1].copy()
        ps['winner'] = np.where(ps['home_pts'] > ps['visitor_pts'], ps['home'], ps['away'])
        ps['pair'] = [frozenset(x) for x in zip(ps['home'], ps['away'])]
        # Round of each real game: every pair's series is numbered by how many
        # series its teams had already played. 2020: everything before the
        # first round (Aug 11) is round 1 (round robin + qualifiers) - Boston
        # and Tampa Bay met in both the round robin and the second round.
        ps['phase'] = 0
        if self.fmt == 'bubble':
            ps['phase'] = (ps['date'] >= BUBBLE_R1).astype(int)
        first = ps.groupby(['phase', 'pair'])['date'].min().sort_values()
        done_series = {}
        round_of = {}
        for (phase, p), _ in first.items():
            if self.fmt == 'bubble' and phase == 0:
                round_of[(phase, p)] = 1
                continue
            base = 2 if self.fmt == 'bubble' else 1
            rnd = base + max(done_series.get(t, 0) for t in p)
            round_of[(phase, p)] = rnd
            for t in p:
                done_series[t] = done_series.get(t, 0) + 1
        ps['round'] = [round_of[(ph, p)] for ph, p in zip(ps['phase'], ps['pair'])]
        self.ps_games = {}
        for p, rnd, dt, w in zip(ps['pair'], ps['round'], ps['date'], ps['winner']):
            self.ps_games.setdefault((p, rnd), []).append((dt, w))
        self.n_rounds = 5 if self.fmt == 'bubble' else 4

    def rs_over(self, d):
        return not ((self.rs['date'] > d) | self.rs['home_pts'].isna()).any()

    # ── regular season ─────────────────────────────────────────────────
    def _standings(self, d, n, rng, R, E):
        """Points, games, tiebreak wins per sim (n, T)."""
        T = len(self.teams)
        s = self.season
        otl = self.otl
        rs = self.rs
        played = rs['home_pts'].notna() & (rs['date'] <= d)
        done, rest = rs[played], rs[~played]
        P = np.zeros(T); G = np.zeros(T); WT = np.zeros(T)
        for h, a, hp, vp, ot, tie in done[['h', 'a', 'home_pts', 'visitor_pts', 'overtimes', 'is_tie']].itertuples(index=False):
            G[h] += 1; G[a] += 1
            if tie == 1 or hp == vp:
                P[h] += 1; P[a] += 1
                continue
            w, l = (h, a) if hp > vp else (a, h)
            P[w] += 2
            if isinstance(ot, str):
                P[l] += otl
                if self.wtype == 'W' or (self.wtype == 'ROW' and ot != 'SO'):
                    WT[w] += 1
            else:
                WT[w] += 1
        Pn = np.tile(P, (n, 1)); Gn = np.tile(G, (n, 1)); WTn = np.tile(WT, (n, 1))
        if len(rest):
            t0, t1, A, hh = _era(REG_PARAMS, s)
            dec, oa, ob = _era(OT_PARAMS, s)
            h = rest['h'].to_numpy(); a = rest['a'].to_numpy()
            dd = R[h] - R[a] + (0.0 if E is None else E[:, h] - E[:, a])
            dd = np.broadcast_to(dd, (n, len(rest)))
            u = lambda: rng.random((n, len(rest)), dtype=np.float32)
            tied = u() < expit(t0 + t1 * np.abs(dd + hh))
            hreg = ~tied & (u() < ndtr(A * (dd + hh)))
            settled = tied & (u() < dec)
            hot = settled & (u() < ndtr(oa * dd + ob))
            aot = settled & ~hot
            still = tied & ~settled
            areg = ~tied & ~hreg
            hp_ = 2.0 * (hreg | hot) + still + otl * aot
            ap_ = 2.0 * (areg | aot) + still + otl * hot
            hw_ = (hreg | (hot if self.wtype != 'RW' else False)).astype(np.float32)
            aw_ = (areg | (aot if self.wtype != 'RW' else False)).astype(np.float32)
            Hm = np.zeros((len(rest), T), np.float32); Hm[np.arange(len(rest)), h] = 1
            Am = np.zeros((len(rest), T), np.float32); Am[np.arange(len(rest)), a] = 1
            Pn += hp_.astype(np.float32) @ Hm + ap_.astype(np.float32) @ Am
            WTn += hw_ @ Hm + aw_ @ Am
            Gn += (Hm + Am).sum(0)
        return Pn, Gn, WTn, rest.empty

    # ── playoffs ───────────────────────────────────────────────────────
    def odds_at(self, d, n_sims=N_SIMS):
        if n_sims <= CHUNK:
            return self._odds_at(d, n_sims, 0)
        parts, done, k = [], 0, 0
        while done < n_sims:
            n = min(CHUNK, n_sims - done)
            parts.append(self._odds_at(d, n, k) * n)
            done += n; k += 1
        return sum(parts) / n_sims

    def _odds_at(self, d, n, chunk):
        T = len(self.teams)
        rng = np.random.default_rng([int(pd.Timestamp(d).strftime('%Y%m%d')), chunk])
        rt = self.ratings.get(d, {})
        R = np.array([rt.get(t, 0.0) for t in self.teams])
        rest = self.rs[~(self.rs['home_pts'].notna() & (self.rs['date'] <= d))]
        frac_left = len(rest) / max(len(self.rs), 1)
        sd = drift_sd(frac_left)
        lean = -drift_lean(frac_left) * (R - R.mean())
        E = rng.normal(lean, sd, (n, T)) if sd > 0 else (lean[None, :] if lean.any() else None)
        Pn, Gn, WTn, rs_done = self._standings(d, n, rng, R, E)
        S = 1 if rs_done else n                       # one table once the RS is over
        pct = Pn[:S] / np.maximum(2 * Gn[:S], 1)
        noise = rng.random((S, T))
        # pct first; ties -> tiebreak wins, then the league's own order, then a coin flip
        key = pct + WTn[:S] * 1e-7 + self.static[None, :] * 1e-10 + noise * 1e-12
        self.rs_complete = rs_done
        self._key = key
        Rp = R[None, :] + (0.0 if E is None else E)   # playoff ratings per sim
        self._Rp = Rp
        self._d = d
        self._n = n
        self._rng = rng
        self.matchups = []
        self.seeds = {}
        self.used_actual = 0
        self.reach = np.zeros((self.n_rounds + 2, T))
        self.entered = np.zeros((n, T), bool)
        champ = getattr(self, '_play_' + self.fmt)()
        np.add.at(self.reach[-1], champ, 1)
        reach = self.reach / n
        cols = ['playoffs'] + [f'r{k}' for k in range(2, self.n_rounds + 1)] + ['champ']
        rows = np.vstack([reach[0]] + [reach[k] for k in range(2, self.n_rounds + 1)] + [reach[-1]])
        return pd.DataFrame(rows.T, index=self.teams, columns=cols)

    # helpers ---------------------------------------------------------------
    def _bcast(self, x):
        return np.broadcast_to(x, (self._n,) + x.shape[1:]) if x.shape[0] == 1 else x

    def _rank(self, members):
        """(n, len) members ordered best first."""
        m = np.array(members)
        order = np.argsort(-self._key[:, m], axis=1, kind='stable')
        return self._bcast(m[order])

    def _better(self, a, b):
        """a has the better record (hosts)."""
        k = self._bcast(self._key)
        ix = np.arange(self._n)
        return k[ix, a] > k[ix, b]

    def _enter(self, teams, rnd):
        ix = np.arange(self._n)
        for t in teams:
            new = ~self.entered[ix, t]
            self.entered[ix, t] = True
            np.add.at(self.reach[0], t[new], 1)
            for k in range(2, rnd):
                np.add.at(self.reach[k], t[new], 1)
            np.add.at(self.reach[rnd], t, 1)

    def _label(self, arr, fn):
        if self.rs_complete:
            for k, t in enumerate(arr[0]):
                self.seeds[self.teams[t]] = fn(k)

    def series(self, a, b, rnd, bo, neutral=False):
        """Winner array of a best-of-bo series; real games fixed."""
        self._enter((a, b), rnd)
        n = self._n
        ix = np.arange(n)
        fixed = np.all(a == a[0]) and np.all(b == b[0])
        ta, tb = self.teams[a[0]], self.teams[b[0]]
        actual = []
        if fixed:
            actual = [w for dt, w in self.ps_games.get((frozenset((ta, tb)), rnd), []) if dt <= self._d]
        a_hosts = self._better(a, b)
        need = bo // 2 + 1
        wa = np.zeros(n, int); wb = np.zeros(n, int)
        Rp = self._bcast(self._Rp)
        base = Rp[ix, a] - Rp[ix, b]
        for gi, better_hosts in enumerate(host_pattern(bo)):
            if gi < len(actual):
                won = np.full(n, actual[gi] == ta); self.used_actual += 1
            else:
                home = 0.0 if neutral else np.where(a_hosts == bool(better_hosts), PH, -PH)
                won = self._rng.random(n) < ndtr(PA * (base + home))
            live = (wa < need) & (wb < need)
            wa += won & live; wb += ~won & live
        if fixed and self.rs_complete:
            games_ = actual[:bo]
            na, nb = games_.count(ta), games_.count(tb)
            decided = ta if na >= need else (tb if nb >= need else None)
            self.matchups.append((rnd, bo, ta, tb, games_, decided))
        return np.where(wa >= need, a, b)

    def _reseed(self, teams, seedkey):
        """Pair best vs worst among (n, k) teams by seedkey (n, T)."""
        ix = np.arange(self._n)[:, None]
        sk = self._bcast(seedkey)[ix, teams]
        order = np.argsort(-sk, axis=1, kind='stable')
        s = teams[ix, order]
        k = s.shape[1]
        return [(s[:, i], s[:, k - 1 - i]) for i in range(k // 2)]

    def _final(self, a, b, rnd):
        return self.series(a, b, rnd, 7)

    # formats ---------------------------------------------------------------
    def _play_overall16(self):
        top = self._rank(range(len(self.teams)))[:, :16]
        self._label(top, lambda k: str(k + 1))
        sk = self._key
        alive = top
        for rnd, bo in ((1, 5), (2, 7), (3, 7), (4, 7)):
            win = [self.series(a, b, rnd, bo) for a, b in self._reseed(alive, sk)]
            alive = np.stack(win, 1)
        return alive[:, 0]

    def _play_division(self):
        bo1 = 5 if self.season <= 1986 else 7
        confs = {}
        for c in sorted(set(self.conf)):
            winners = []
            for dv in sorted(set(self.div[self.conf == c])):
                m = np.where((self.conf == c) & (self.div == dv))[0]
                top = self._rank(m)[:, :4]
                self._label(top, lambda k, dv=dv: f'{dv[0]}{k + 1}')
                s1 = self.series(top[:, 0], top[:, 3], 1, bo1)
                s2 = self.series(top[:, 1], top[:, 2], 1, bo1)
                winners.append(self.series(s1, s2, 2, 7))
            confs[c] = self.series(winners[0], winners[1], 3, 7)
        a, b = list(confs.values())
        return self._final(a, b, 4)

    def _play_conference(self):
        top_n = 2 if self.season <= 1998 else 3        # division winners seeded first
        champs = []
        for c in sorted(set(self.conf)):
            m = np.where(self.conf == c)[0]
            order = self._rank(m)
            n = self._n
            ix = np.arange(n)[:, None]
            pos = np.empty((n, len(self.teams)), int)
            pos[ix, order] = np.arange(len(m))[None, :]
            bonus = np.zeros((n, len(self.teams)))
            for dv in sorted(set(self.div[m])):
                mem = np.where((self.conf == c) & (self.div == dv))[0]
                win = mem[np.argmin(pos[:, mem], axis=1)]
                bonus[np.arange(n), win] = 1
            sk = self._bcast(self._key) + bonus * 10
            mm = np.array(m)
            seeds = mm[np.argsort(-sk[:, mm], axis=1, kind='stable')][:, :8]
            self._label(seeds, lambda k, c=c: f'{c[0]}{k + 1}')
            seedrank = np.zeros((n, len(self.teams)))
            seedrank[ix, seeds] = 8 - np.arange(8)[None, :]
            alive = seeds
            for rnd in (1, 2, 3):
                win = [self.series(a, b, rnd, 7) for a, b in self._reseed(alive, seedrank)]
                alive = np.stack(win, 1)
            champs.append(alive[:, 0])
        return self._final(champs[0], champs[1], 4)

    def _play_wildcard(self):
        champs = []
        n = self._n
        for c in sorted(set(self.conf)):
            divs = sorted(set(self.div[self.conf == c]))
            tops = {dv: self._rank(np.where((self.conf == c) & (self.div == dv))[0]) for dv in divs}
            ix = np.arange(n)[:, None]
            taken = np.zeros((n, len(self.teams)), bool)
            for dv in divs:
                taken[ix, tops[dv][:, :3]] = True
            m = np.where(self.conf == c)[0]
            k = self._bcast(self._key)[:, m] - taken[:, m] * 10
            wc = m[np.argsort(-k, axis=1, kind='stable')][:, :2]
            d1, d2 = tops[divs[0]], tops[divs[1]]
            self._label(d1[:, :3], lambda k, dv=divs[0]: f'{dv[0]}{k + 1}')
            self._label(d2[:, :3], lambda k, dv=divs[1]: f'{dv[0]}{k + 1}')
            self._label(wc, lambda k, c=c: f'{c[0]}WC{k + 1}')
            d1_best = self._better(d1[:, 0], d2[:, 0])
            wc_for_d1 = np.where(d1_best, wc[:, 1], wc[:, 0])
            wc_for_d2 = np.where(d1_best, wc[:, 0], wc[:, 1])
            finals = []
            for top, wct in ((d1, wc_for_d1), (d2, wc_for_d2)):
                a = self.series(top[:, 0], wct, 1, 7)
                b = self.series(top[:, 1], top[:, 2], 1, 7)
                finals.append(self.series(a, b, 2, 7))
            champs.append(self.series(finals[0], finals[1], 3, 7))
        return self._final(champs[0], champs[1], 4)

    def _play_div2021(self):
        winners = []
        for dv in sorted(set(self.div)):
            m = np.where(self.div == dv)[0]
            top = self._rank(m)[:, :4]
            self._label(top, lambda k, dv=dv: f'{dv.split()[-1][0]}{k + 1}')
            s1 = self.series(top[:, 0], top[:, 3], 1, 7)
            s2 = self.series(top[:, 1], top[:, 2], 1, 7)
            winners.append(self.series(s1, s2, 2, 7))
        four = np.stack(winners, 1)
        pairs = self._reseed(four, self._key)
        a, b = [self.series(x, y, 3, 7) for x, y in pairs]
        return self._final(a, b, 4)

    def _play_bubble(self):
        """2020: top 4 per conference play a round robin for seeds 1-4, seeds
        5-12 play best-of-5 qualifiers; then reseeded Bo7 rounds. Neutral."""
        n = self._n
        ix = np.arange(n)
        champs = []
        for c in sorted(set(self.conf)):
            m = np.where(self.conf == c)[0]
            order = self._rank(m)[:, :12]
            self._label(order, lambda k, c=c: f'{c[0]}{k + 1}')
            rr = order[:, :4]
            self._enter([rr[:, i] for i in range(4)], 1)
            rrw = np.zeros((n, len(self.teams)))
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = rr[:, i], rr[:, j]
                    fixed = np.all(a == a[0]) and np.all(b == b[0])
                    x = [w for dt, w in self.ps_games.get((frozenset((self.teams[a[0]], self.teams[b[0]])), 1), [])
                         if dt <= self._d] if fixed else []
                    if x:
                        won = np.full(n, x[0] == self.teams[a[0]]); self.used_actual += 1
                    else:
                        Rp = self._bcast(self._Rp)
                        won = self._rng.random(n) < ndtr(PA * (Rp[ix, a] - Rp[ix, b]))
                    np.add.at(rrw, (ix, np.where(won, a, b)), 1)
            key = rrw + self._bcast(self._key) * 1e-3          # round-robin wins, then RS points %
            rrs = rr[ix[:, None], np.argsort(-key[ix[:, None], rr], axis=1, kind='stable')]
            quals = [self.series(order[:, 4 + i], order[:, 11 - i], 1, 5, neutral=True) for i in range(4)]
            alive = np.concatenate([rrs, np.stack(quals, 1)], 1)
            seedrank = np.zeros((n, len(self.teams)))
            seedrank[ix[:, None], rrs] = 100 - np.arange(4)[None, :]
            seedrank[ix[:, None], order[:, 4:]] = 50 - np.arange(8)[None, :]
            for rnd in (2, 3, 4):
                win = [self.series(a, b, rnd, 7, neutral=True) for a, b in self._reseed(alive, seedrank)]
                alive = np.stack(win, 1)
            champs.append(alive[:, 0])
        return self.series(champs[0], champs[1], 5, 7, neutral=True)


def compute(games, ratings_df, current_season, schedule=None, seasons=None, log=print):
    """games: all NHL games (season, date, home, away, home_pts, visitor_pts,
    overtimes, is_tie, is_playoff). ratings_df: (season, date, name, rating).
    Returns (odds, brackets) like DUNCAN's."""
    out, brackets = [], {}
    for season, g in games.groupby('season'):
        season = int(season)
        if seasons is not None and season not in seasons:
            continue
        rsub = ratings_df[ratings_df['season'] == season]
        ratings = {d: dict(zip(x['name'], x['rating'])) for d, x in rsub.groupby('date')}
        if not ratings:
            continue
        sim = SeasonSim(season, g, ratings, schedule if season == current_season else None)
        for d in sorted(ratings):
            n = N_SIMS_PLAYOFFS if sim.rs_over(d) else N_SIMS
            o = sim.odds_at(d, n_sims=n)
            if sim.rs_complete:
                brackets.setdefault(season, {})[d] = (dict(sim.seeds), list(sim.matchups), n)
            o.index.name = 'team'
            o = o.reset_index()
            o['season'] = season
            o['date'] = d
            out.append(o)
        log(f'  {season}: {len(ratings)} snapshots')
    return pd.concat(out, ignore_index=True), brackets


# ── Cached, parallel driver (as DUNCAN) ──────────────────────────────────────
_ENGINE_FILES = ('playoff_sim.py', 'nhl_divisions.csv')
_JOB = {}


def _fingerprint(season, games, ratings_df, schedule, current_season):
    h = hashlib.sha256()
    for f in _ENGINE_FILES:
        p = _os.path.join(_HERE, f)
        if _os.path.exists(p):
            h.update(open(p, 'rb').read())
    g = games[games['season'] == season].sort_values(['date', 'home']).copy()
    h.update(g.to_csv(index=False).encode())
    r = ratings_df[ratings_df['season'] == season].sort_values(['date', 'name']).copy()
    r['rating'] = r['rating'].round(3)
    h.update(r.to_csv(index=False).encode())
    if season == current_season and schedule is not None:
        h.update(schedule.sort_values(['date', 'home']).to_csv(index=False).encode())
    return h.hexdigest()


def _one(season):
    j = _JOB
    return season, compute(j['games'], j['ratings'], j['current'], j['schedule'],
                           seasons={season}, log=lambda *_: None)


def compute_cached(games, ratings_df, current_season, schedule=None, cache_dir='title_odds_cache',
                   workers=None, log=print):
    _os.makedirs(cache_dir, exist_ok=True)
    seasons = sorted(int(s) for s in games['season'].unique() if (ratings_df['season'] == s).any())
    results, todo, sigs = {}, [], {}
    for s in seasons:
        sigs[s] = _fingerprint(s, games, ratings_df, schedule, current_season)
        path = _os.path.join(cache_dir, f'{s}.pkl')
        if _os.path.exists(path):
            try:
                sig, payload = pickle.load(open(path, 'rb'))
                if sig == sigs[s]:
                    results[s] = payload
                    continue
            except Exception:
                pass
        todo.append(s)
    log(f'  {len(results)} seasons from cache, computing {len(todo)}: {todo}')
    if todo:
        _JOB.update(games=games, ratings=ratings_df, current=current_season, schedule=schedule)
        ctx = _mp.get_context('fork')
        with ctx.Pool(workers or _os.cpu_count()) as pool:
            for s, payload in pool.imap_unordered(_one, todo):
                results[s] = payload
                pickle.dump((sigs[s], payload), open(_os.path.join(cache_dir, f'{s}.pkl'), 'wb'))
                log(f'  {s} done')
    odds = pd.concat([results[s][0] for s in seasons], ignore_index=True)
    brackets = {}
    for s in seasons:
        brackets.update(results[s][1])
    return odds, brackets
