"""
◎の買い方の検証(バックテスト)

  * 古い期間で学習 → 新しい期間(テスト)で予想。テスト期間の結果は学習に使わない
  * オッズは Turnmark API(https://turnmark.github.io/)の過去オッズ(確定オッズ)
  * 買い目: 2連単2点・3連単3点(アプリと同じ)。期待値 = 確率 × オッズ
  * ◎ = 期待値が基準(本命の確率50%未満なら1.2、以上なら1.1)以上
  * 返還(フライング・出遅れ・欠場)は100円戻る

使い方: python backtest.py [テスト日数=60] [学習日数=90]
結果は data/backtest.json に保存
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import requests
from sklearn.isotonic import IsotonicRegression as ISO

import bot

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "backtest.json")
TM = "https://turnmark.github.io/api/v1/{y}/{d8}.json"


def tm_odds(d):
    try:
        r = requests.get(TM.format(y=d.year, d8=d.strftime("%Y%m%d")), timeout=60)
        if r.status_code != 200:
            return None
        st = r.json()["programs"]["stadiums"]
    except Exception:
        return None
    out = {}
    for sn, s in st.items():
        for rn, race in (s.get("races") or {}).items():
            od = race.get("odds") or {}
            out[(int(sn), int(rn))] = {"ex": od.get("exacta") or {}, "tri": od.get("trifecta") or {}}
    return out


def odd(o, c):
    try:
        x = o
        for k in c:
            x = x[str(k)]
        return float(x) if x else None
    except Exception:
        return None


def result(p):
    pl = bot.places(p)
    if not all(i in pl for i in (1, 2, 3)):
        return None
    o3 = (pl[1], pl[2], pl[3])
    ex = sum(x["payout"] for x in p["payouts"].get("exacta", []) if x["combination"] == "%d-%d" % o3[:2])
    tr = sum(x["payout"] for x in p["payouts"].get("trifecta", []) if x["combination"] == "%d-%d-%d" % o3)
    ret = set()
    for b in p["boats"]:
        try:
            if int(b["racer_place_number"]) in (14, 15, 16):
                ret.add(int(b["racer_boat_number"]))
        except Exception:
            pass
    return o3, ex, tr, ret


def prepare(test_n, train_n):
    """学習してテスト期間の各レースの確率・オッズ・結果を返す"""
    T = dt.datetime.now(bot.JST).date()
    first = dt.date(2026, 1, 1)
    last_k = min(test_n + 15 + train_n, (T - first).days)
    bot.HIST.clear(); bot.SEEN.clear()
    bot.fan_init()
    RS = {}
    for k in range(1, last_k + 67):
        if T - dt.timedelta(days=k) < first:
            break
        r = bot.load_results(T - dt.timedelta(days=k))
        if k <= last_k:
            RS[k] = r
    DATA = []   # (k, 日付, 場, R, X, 順位, 結果)
    for k in range(1, last_k + 1):
        d = T - dt.timedelta(days=k)
        rs = RS.get(k)
        pg = bot.cget("api/v1", d)
        pv = bot.cget("previews/v2", d)
        if not pg or not rs:
            continue
        st = pg["programs"]["stadiums"]
        PV = bot.pvmap(pv)
        for p in rs["results"]:
            key = (p["race_stadium_number"], p["race_number"])
            sn, rn = str(key[0]), str(key[1])
            if sn not in st or rn not in st[sn]["races"]:
                continue
            res = result(p)
            if not res:
                continue
            pl = bot.places(p)
            X = bot.feats(key[0], st[sn]["races"][rn], PV.get(key))
            if X is not None:
                DATA.append((k, d, key[0], key[1], X, pl, res))
    bot.cache_flush()
    # 学習: テスト期間より前だけを使う(本番と同じく、古い部分で学習→直近15日で確率補正→全部で学習し直し)
    TR = [r for r in DATA if r[0] > test_n + 15]
    CA = [r for r in DATA if test_n < r[0] <= test_n + 15]
    TE = [r for r in DATA if r[0] <= test_n]
    M = bot.train([(r[0], r[4], r[5]) for r in TR])
    P, Y = [], []
    for r in CA:
        P += list(bot.ml(M, r[4])[1])
        Y += [1 if r[5][1] == n else 0 for n in range(1, 7)]
    CAL = ISO(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(P, Y)
    M2 = bot.train([(r[0], r[4], r[5]) for r in TR + CA])
    print(f"学習 {len(TR) + len(CA)}レース / テスト {len(TE)}レース")
    ODDS, RACES = {}, []
    for k, d, sn, rn, X, pl, res in TE:
        if d not in ODDS:
            ODDS[d] = tm_odds(d) or {}
        od = ODDS[d].get((sn, rn))
        if not od:
            continue
        _, p0 = bot.ml(M2, X)
        RACES.append({"k": k, "d": d, "sn": sn, "rn": rn, "p0": p0, "pc": bot.cal(CAL, p0), "od": od, "res": res})
    return RACES, len(TR) + len(CA)


def main():
    test_n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    train_n = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    RACES, ntrain = prepare(test_n, train_n)
    bets = []   # 1点ごとの記録
    for R in RACES:
        d, sn, rn, p0, pc, od = R["d"], R["sn"], R["rn"], R["p0"], R["pc"], R["od"]
        o3, ex, tr, ret = R["res"]
        pw = float(max(pc))
        lv = bot.level(pw)[0]
        need = 1.2 if pw < 0.5 else 1.1
        cands = [("2連単", (a, b), q) for q, a, b in bot.ex2(p0)[:2]] + \
                [("3連単", (a, b, c), q) for q, a, b, c in bot.tri(p0)[:3]]
        for kind, c, q in cands:
            o = odd(od["ex" if kind == "2連単" else "tri"], c)
            if not o:
                continue
            if ret & set(c):
                pay = 100
            elif kind == "2連単":
                pay = ex if c == o3[:2] else 0
            else:
                pay = tr if c == o3 else 0
            bets.append({"d": d.isoformat(), "sn": sn, "rn": rn, "kind": kind, "c": "-".join(map(str, c)),
                         "q": float(q), "o": o, "e": float(q) * o, "need": need, "lv": lv,
                         "pay": pay, "hit": bool(pay and not (ret & set(c))), "ret": bool(ret & set(c))})

    def agg(bs):
        n = len(bs)
        if not n:
            return {"n": 0}
        h = sum(b["hit"] for b in bs)
        back = sum(b["pay"] for b in bs)
        return {"n": n, "hit": h, "hr": h / n, "buy": 100 * n, "back": back, "rr": back / (100 * n),
                "races": len({(b["d"], b["sn"], b["rn"]) for b in bs})}

    maru = [b for b in bets if b["e"] >= b["need"]]
    out = {"made": dt.datetime.now(bot.JST).strftime("%Y-%m-%d %H:%M"),
           "test_from": min(b["d"] for b in bets), "test_to": max(b["d"] for b in bets),
           "test_races": len({(b["d"], b["sn"], b["rn"]) for b in bets}), "train_races": ntrain,
           "all": {k: agg([b for b in bets if b["kind"] == k]) for k in ("2連単", "3連単")},
           "maru": {k: agg([b for b in maru if b["kind"] == k]) for k in ("2連単", "3連単")},
           "maru_total": agg(maru),
           "maru_level": {lv: {k: agg([b for b in maru if b["kind"] == k and b["lv"] == lv]) for k in ("2連単", "3連単")}
                          for lv in ("◎", "○", "△")},
           "ev": {}, "odds_band": {}, "weeks": {}}
    # 自信度ごと: 全部買い・期待値の基準別・前半/後半
    mid = sorted({b["d"] for b in bets})[len({b["d"] for b in bets}) // 2]
    out["level"] = {}
    for lv in ("◎", "○", "△"):
        L = [b for b in bets if b["lv"] == lv]
        out["level"][lv] = {"races": len({(b["d"], b["sn"], b["rn"]) for b in L}),
                            "all": {k: agg([b for b in L if b["kind"] == k]) for k in ("2連単", "3連単")},
                            "ev": {str(th): {k: agg([b for b in L if b["kind"] == k and b["e"] >= th]) for k in ("2連単", "3連単")}
                                   for th in (1.0, 1.1, 1.2, 1.5, 2.0)},
                            "half": {h: {k: agg([b for b in L if b["kind"] == k and (b["d"] < mid) == (h == "前半")]) for k in ("2連単", "3連単")}
                                     for h in ("前半", "後半")},
                            "half_maru": {h: {k: agg([b for b in L if b["kind"] == k and b["e"] >= b["need"] and (b["d"] < mid) == (h == "前半")]) for k in ("2連単", "3連単")}
                                          for h in ("前半", "後半")}}
    for th in (0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0):
        out["ev"][str(th)] = {k: agg([b for b in bets if b["kind"] == k and b["e"] >= th]) for k in ("2連単", "3連単")}
    bands = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]
    for lo, hi in bands:
        out["odds_band"][f"{lo}-{hi if hi < 1e9 else ''}"] = {
            k: agg([b for b in maru if b["kind"] == k and lo <= b["o"] < hi]) for k in ("2連単", "3連単")}
    for b in maru:
        w = dt.date.fromisoformat(b["d"])
        w = (w - dt.timedelta(days=w.weekday())).isoformat()
        out["weeks"].setdefault(w, []).append(b)
    out["weeks"] = {w: agg(bs) for w, bs in sorted(out["weeks"].items())}
    # 確定オッズと払戻の一致チェック(的中した買い目で 払戻 ≒ オッズ×100 か)
    hits = [b for b in bets if b["hit"]]
    out["check"] = {"n": len(hits), "median_ratio": float(np.median([b["pay"] / (b["o"] * 100) for b in hits])) if hits else None}
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("test_from", "test_to", "test_races", "all", "maru", "maru_total", "check")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
