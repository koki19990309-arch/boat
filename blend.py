"""
予想の確率をオッズ(みんなの読み)と混ぜると儲かるかの検証

  q_混合 ∝ q_予想^a × q_オッズ^b   (q_オッズ = 1/オッズ を合計1にしたもの)
  a, b は前半30日で決め(当たり目の確率が一番高くなる組)、後半30日で成績を確かめる
  買い方: 期待値 = q_混合 × オッズ が基準以上の買い目を100円ずつ

使い方: python blend.py   → data/blend.json
"""
import json
import math
import os
from itertools import permutations

import numpy as np

import backtest as B
import bot

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "blend.json")
GRID = [round(x * 0.1, 1) for x in range(0, 16)]


def combos(R, kind):
    """そのレースの全組: (組, 予想確率, オッズ, 払戻)"""
    p0 = R["p0"]
    o3, ex, tr, ret = R["res"]
    od = R["od"]["ex" if kind == "ex" else "tri"]
    model = {}
    if kind == "ex":
        for q, a, b in bot.ex2(p0):
            model[(a, b)] = q
    else:
        for q, a, b, c in bot.tri(p0):
            model[(a, b, c)] = q
    out = []
    for c, q in model.items():
        o = B.odd(od, c)
        if not o:
            continue
        if ret & set(c):
            pay = 100
        else:
            pay = (ex if c == o3[:2] else 0) if kind == "ex" else (tr if c == o3 else 0)
        win = (c == o3[:2]) if kind == "ex" else (c == o3)
        out.append((c, float(q), o, pay, win and not (ret & set(c))))
    if not out:
        return None
    qm = np.array([x[1] for x in out]); qm = qm / qm.sum()
    mk = np.array([1 / x[2] for x in out]); mk = mk / mk.sum()
    return {"c": [x[0] for x in out], "qm": qm, "mk": mk, "o": np.array([x[2] for x in out]),
            "pay": np.array([x[3] for x in out]), "win": np.array([x[4] for x in out]), "ret": bool(ret)}


def mix(C, a, b):
    z = a * np.log(np.maximum(C["qm"], 1e-9)) + b * np.log(np.maximum(C["mk"], 1e-9))
    z = np.exp(z - z.max())
    return z / z.sum()


def loglik(CS, a, b):
    s, n = 0.0, 0
    for C in CS:
        if C["ret"] or not C["win"].any():
            continue
        s += math.log(max(mix(C, a, b)[C["win"].argmax()], 1e-12)); n += 1
    return s / max(n, 1)


def agg(n, h, back):
    return {"n": int(n), "hit": int(h), "hr": h / n if n else 0, "buy": 100 * int(n), "back": int(back),
            "rr": back / (100 * n) if n else 0}


def evaluate(CS, a, b, th, top=None, need_app=None):
    n = h = back = 0
    for C in CS:
        q = mix(C, a, b)
        e = q * C["o"]
        idx = range(len(e)) if top is None else np.argsort(-C["qm"])[:top]
        for i in idx:
            if e[i] >= (th if need_app is None else need_app[id(C)]):
                n += 1; back += C["pay"][i]; h += int(C["win"][i])
    return agg(n, h, back)


def main():
    RACES, ntrain = B.prepare(60, 90)
    FIT = [R for R in RACES if R["k"] > 30]
    EV = [R for R in RACES if R["k"] <= 30]
    out = {"fit": [min(R["d"] for R in FIT).isoformat(), max(R["d"] for R in FIT).isoformat(), len(FIT)],
           "eval": [min(R["d"] for R in EV).isoformat(), max(R["d"] for R in EV).isoformat(), len(EV)], "kinds": {}}
    for kind, name, top in (("ex", "2連単", 2), ("tri", "3連単", 3)):
        CF = [c for c in (combos(R, kind) for R in FIT) if c]
        CE = [c for c in (combos(R, kind) for R in EV) if c]
        best = max(((a, b) for a in GRID for b in GRID if a + b > 0), key=lambda ab: loglik(CF, *ab))
        a, b = best
        # アプリと同じ基準(本命確率で1.1/1.2)
        need = {}
        for R, C in zip([R for R in EV if combos(R, kind)], CE):
            need[id(C)] = 1.2 if float(max(R["pc"])) < 0.5 else 1.1
        k = {"a": a, "b": b,
             "loglik_eval": {"予想だけ": loglik(CE, 1, 0), "オッズだけ": loglik(CE, 0, 1), "混合": loglik(CE, a, b)},
             "now_app": evaluate(CE, 1, 0, None, top=top, need_app=need),
             "all_buy": evaluate(CE, 1, 0, -1, top=top),
             "mix_top": {}, "mix_all": {}, "mix_all_fit": {}}
        for th in (1.0, 1.05, 1.1, 1.2, 1.3, 1.5):
            k["mix_top"][str(th)] = evaluate(CE, a, b, th, top=top)
            k["mix_all"][str(th)] = evaluate(CE, a, b, th)
            k["mix_all_fit"][str(th)] = evaluate(CF, a, b, th)
        out["kinds"][name] = k
        print(name, json.dumps(k, ensure_ascii=False))
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
