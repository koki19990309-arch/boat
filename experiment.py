"""
ChatGPT の提案の検証
 A. 人気順ごとのオッズの歪み(⑧) — 2026/1/1〜昨日の全レース、2単/3単/2複/3複
 B. 市場との差分を学習するモデル(①) と 2複・3複(⑦)
 C. 着順を直接予想するモデル(②: 1着が決まった時の2着、1・2着が決まった時の3着)
 信頼区間はレース単位のブートストラップ(⑭)

使い方: python experiment.py → data/experiment.json
"""
import datetime as dt
import json
import math
import os
from itertools import permutations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import backtest as B
import bot

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "experiment.json")
KINDS = {"ex": "2連単", "tri": "3連単", "qn": "2連複", "trio": "3連複"}
PAYKEY = {"ex": "exacta", "tri": "trifecta", "qn": "quinella", "trio": "trio"}
rng = np.random.default_rng(0)


def combos(kind):
    if kind == "ex":
        return list(permutations(range(1, 7), 2))
    if kind == "tri":
        return list(permutations(range(1, 7), 3))
    if kind == "qn":
        return [(a, b) for a in range(1, 7) for b in range(a + 1, 7)]
    return [(a, b, c) for a in range(1, 7) for b in range(a + 1, 7) for c in range(b + 1, 7)]


CMB = {k: combos(k) for k in KINDS}


def cstr(kind, c):
    return ("-" if kind in ("ex", "tri") else "=").join(map(str, c))


def race_market(kind, od, pay, ref):
    """そのレースの (組, オッズ, 払戻, 当たり) の配列。オッズのない組は除く"""
    rows = []
    for c in CMB[kind]:
        o = B.odd(od[kind], c)
        if not o:
            continue
        if set(c) & set(ref):
            p, h = 100, 0
        else:
            amt = pay.get(PAYKEY[kind], {}).get(cstr(kind, c))
            p, h = (amt, 1) if amt else (0, 0)
        rows.append((c, o, p, h))
    return rows


def tri_from(p1, p2=None, p3=None):
    """3連単120通りの確率。p2/p3 が無ければハービル式"""
    q = {}
    for a, b, c in CMB["tri"]:
        if p2 is None:
            q[(a, b, c)] = p1[a - 1] * p1[b - 1] / (1 - p1[a - 1]) * p1[c - 1] / (1 - p1[a - 1] - p1[b - 1])
        else:
            q[(a, b, c)] = p1[a - 1] * p2[a][b] * p3[(a, b)][c]
    s = sum(q.values())
    return {k: v / s for k, v in q.items()}


def agg_from_tri(kind, qt):
    if kind == "tri":
        return qt
    out = {}
    for (a, b, c), v in qt.items():
        k = (a, b) if kind == "ex" else tuple(sorted((a, b))) if kind == "qn" else tuple(sorted((a, b, c)))
        out[k] = out.get(k, 0) + v
    return out


def boot(groups, n=1000):
    """レース単位のブートストラップで回収率の95%区間"""
    if not groups:
        return None
    G = np.array(groups, dtype=float)   # (レース数, 2) = 点数, 払戻
    rr = []
    for _ in range(n):
        s = G[rng.integers(0, len(G), len(G))].sum(0)
        rr.append(s[1] / (100 * s[0]) if s[0] else 0)
    return [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))]


def summary(bets):
    """bets: [(レースID, 払戻, 当たり)]"""
    n = len(bets)
    if not n:
        return {"n": 0}
    pay = sum(b[1] for b in bets); hit = sum(b[2] for b in bets)
    g = {}
    for rid, p, _ in bets:
        x = g.setdefault(rid, [0, 0]); x[0] += 1; x[1] += p
    return {"n": n, "races": len(g), "hit": int(hit), "rr": pay / (100 * n), "pl": int(pay - 100 * n), "ci95": boot(list(g.values()))}


# ---------- A. 人気順ごとの歪み ----------
def part_a():
    T = dt.datetime.now(bot.JST).date()
    d = dt.date(2026, 1, 1)
    bands = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 999)]
    acc = {k: {f"{a}-{b}": [0, 0, 0] for a, b in bands} for k in KINDS}
    days = 0
    while d < T:
        day = B.tm_odds(d)
        d += dt.timedelta(days=1)
        if not day:
            continue
        days += 1
        for (sn, rn), R in day.items():
            if not R["pay"]:
                continue
            for k in KINDS:
                rows = race_market(k, R, R["pay"], R["ref"])
                rows.sort(key=lambda x: x[1])
                for rank, (c, o, p, h) in enumerate(rows, 1):
                    for a, b in bands:
                        if a <= rank <= b:
                            x = acc[k][f"{a}-{b}"]; x[0] += 1; x[1] += p; x[2] += h
    out = {"days": days}
    for k, v in acc.items():
        out[KINDS[k]] = {band: {"n": n, "rr": pay / (100 * n) if n else 0, "hr": h / n if n else 0} for band, (n, pay, h) in v.items()}
    return out


# ---------- C. 着順を直接予想 ----------
def cond_rows(X, w, s=None):
    f = [X[w - 1]] + ([X[s - 1]] if s else [])
    rows, cand = [], []
    for j in range(1, 7):
        if j == w or j == s:
            continue
        rows.append(np.concatenate([X[j - 1]] + f + [[j, w] + ([s] if s else [])]))
        cand.append(j)
    return rows, cand


def train_cond(train):
    X2, y2, X3, y3 = [], [], [], []
    for X, pl in train:
        w, s, t = pl[1], pl[2], pl[3]
        r, c = cond_rows(X, w); X2 += r; y2 += [int(j == s) for j in c]
        r, c = cond_rows(X, w, s); X3 += r; y3 += [int(j == t) for j in c]
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
    return HGB(**kw).fit(np.array(X2), y2), HGB(**kw).fit(np.array(X3), y3)


def cond_probs(m2, m3, X):
    p2, p3 = {}, {}
    R2, idx2 = [], []
    for w in range(1, 7):
        r, c = cond_rows(X, w); R2 += r; idx2 += [(w, j) for j in c]
    pr = m2.predict_proba(np.array(R2))[:, 1]
    for (w, j), v in zip(idx2, pr):
        p2.setdefault(w, {})[j] = v
    for w in p2:
        s = sum(p2[w].values()); p2[w] = {j: v / s for j, v in p2[w].items()}
    R3, idx3 = [], []
    for w in range(1, 7):
        for s_ in range(1, 7):
            if s_ == w:
                continue
            r, c = cond_rows(X, w, s_); R3 += r; idx3 += [((w, s_), j) for j in c]
    pr = m3.predict_proba(np.array(R3))[:, 1]
    for (ws, j), v in zip(idx3, pr):
        p3.setdefault(ws, {})[j] = v
    for ws in p3:
        s = sum(p3[ws].values()); p3[ws] = {j: v / s for j, v in p3[ws].items()}
    return p2, p3


# ---------- B. 市場との差分 ----------
FEATS = ["log市場確率", "log予想確率", "log予想/市場", "log予想PL/市場", "logオッズ", "人気順", "予想順", "順位差", "本命確率", "会場", "1着の艇"]


def main():
    out = {"made": dt.datetime.now(bot.JST).strftime("%Y-%m-%d %H:%M")}
    print("A: 人気順の歪み"); out["A_人気順"] = part_a()
    RACES, ntrain = B.prepare(150, 90)
    print("C: 着順モデルを学習", len(B.prepare.train))
    m2, m3 = train_cond(B.prepare.train)
    days = sorted({R["d"] for R in RACES}); cut = days[-60]
    out["期間"] = {"差分モデルの学習": [days[0].isoformat(), (cut - dt.timedelta(days=1)).isoformat()],
                 "確かめる": [cut.isoformat(), days[-1].isoformat()]}
    rows = {k: [] for k in KINDS}   # (fit?, rid, 特徴..., オッズ, 払戻, 当たり)
    ll = {k: {"市場": [], "予想(ハービル)": [], "予想(着順モデル)": []} for k in KINDS}
    for i, R in enumerate(RACES):
        od = R["od"]
        if not od.get("pay"):
            continue
        p1 = np.array(R["p0"], float)
        p2, p3 = cond_probs(m2, m3, R["X"])
        qh = tri_from(p1); qp = tri_from(p1, p2, p3)
        pw = float(max(R["pc"]))
        fit = R["d"] < cut
        for k in KINDS:
            mk_rows = race_market(k, od, od["pay"], od["ref"])
            if not mk_rows:
                continue
            inv = sum(1 / o for _, o, _, _ in mk_rows)
            mh, mp = agg_from_tri(k, qh), agg_from_tri(k, qp)
            mrank = {c: r for r, c in enumerate(sorted(mh, key=lambda c: -mh[c]), 1)}
            orank = {c: r for r, (c, _, _, _) in enumerate(sorted(mk_rows, key=lambda x: x[1]), 1)}
            for c, o, pay, h in mk_rows:
                mk = (1 / o) / inv
                qm, qq = max(mh.get(c, 1e-9), 1e-9), max(mp.get(c, 1e-9), 1e-9)
                rows[k].append([fit, i, math.log(mk), math.log(qm), math.log(qm / mk), math.log(qq / mk), math.log(o),
                                orank[c], mrank.get(c, 99), orank[c] - mrank.get(c, 99), pw, R["sn"], c[0], o, pay, h])
                if h and not fit:
                    ll[k]["市場"].append(math.log(mk)); ll[k]["予想(ハービル)"].append(math.log(qm))
                    ll[k]["予想(着順モデル)"].append(math.log(qq))
    out["的中目の平均対数確率(高いほど正確)"] = {KINDS[k]: {n: float(np.mean(v)) for n, v in d.items()} for k, d in ll.items()}
    out["B_差分モデル"] = {}
    for k in KINDS:
        A = np.array(rows[k], dtype=float)
        fit = A[:, 0].astype(bool); rid = A[:, 1].astype(int); X = A[:, 2:13]; o = A[:, 13]; pay = A[:, 14]; y = A[:, 15]
        res = {}
        hg = HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0,
                 categorical_features=[9, 10]).fit(X[fit], y[fit])
        sc = StandardScaler().fit(X[fit][:, :9])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[fit][:, :9]), y[fit])
        te = ~fit
        preds = {"差分GBM": hg.predict_proba(X[te])[:, 1], "差分ロジスティック": lr.predict_proba(sc.transform(X[te][:, :9]))[:, 1],
                 "市場だけ": np.exp(X[te][:, 0]), "予想(ハービル)": np.exp(X[te][:, 1])}
        rt, ot, pt, yt = rid[te], o[te], pay[te], y[te]
        for name, p in preds.items():
            # レース内で合計1にそろえる
            s = np.zeros(rt.max() + 1); np.add.at(s, rt, p); q = p / s[rt]
            hit_ll = float(np.mean(np.log(np.maximum(q[yt == 1], 1e-12))))
            ev = q * ot
            st = {"的中目の平均対数確率": hit_ll}
            for th in (1.0, 1.1, 1.2, 1.3, 1.5):
                m = ev >= th
                st[f"期待値{th}以上"] = summary(list(zip(rt[m], pt[m], yt[m])))
            # 1レース1点: 期待値が最大の組だけ(1.0以上の時)
            best = {}
            for j in np.where(ev >= 1.0)[0]:
                if rt[j] not in best or ev[j] > ev[best[rt[j]]]:
                    best[rt[j]] = j
            jj = np.array(list(best.values()), int)
            st["1レース1点(期待値最大・1.0以上)"] = summary(list(zip(rt[jj], pt[jj], yt[jj]))) if len(jj) else {"n": 0}
            res[name] = st
        out["B_差分モデル"][KINDS[k]] = res
        print(KINDS[k], json.dumps({n: {kk: (v["rr"] if isinstance(v, dict) and v.get("n") else v) for kk, v in s.items()} for n, s in res.items()}, ensure_ascii=False))
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
