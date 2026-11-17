"""
いちばん勝てる買い方を探す(総当たり + 後の期間で確認)

  * テスト150日のうち、古い90日(探す期間)で条件を総当たりして回収率の高いものを選ぶ
  * 新しい60日(確かめる期間)で、選んだ条件が本当に通用するかを見る
  * 条件: 券種 / 自信度 / 予想順位 / 期待値 / オッズ帯 / 予想とオッズのズレ / 会場

使い方: python search.py → data/search.json
"""
import datetime as dt
import itertools
import json
import os

import numpy as np

import backtest as B
import bot

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "search.json")


def rows_of(RACES):
    rows = []
    for R in RACES:
        p0, pc, od = R["p0"], R["pc"], R["od"]
        o3, ex, tr, ret = R["res"]
        pw = float(max(pc)); lv = bot.level(pw)[0]
        for kind, lst, top in (("2連単", [(q, (a, b)) for q, a, b in bot.ex2(p0)], 4),
                               ("3連単", [(q, (a, b, c)) for q, a, b, c in bot.tri(p0)], 6)):
            odt = od["ex" if kind == "2連単" else "tri"]
            allo = [B.odd(odt, c) for _, c in lst]
            inv = sum(1 / o for o in allo if o)
            if not inv:
                continue
            for rank, ((q, c), o) in enumerate(zip(lst, allo), 1):
                if rank > top or not o:
                    continue
                if ret & set(c):
                    pay = 100
                elif kind == "2連単":
                    pay = ex if c == o3[:2] else 0
                else:
                    pay = tr if c == o3 else 0
                mk = (1 / o) / inv
                rows.append((R["d"], R["sn"], kind == "2連単", "◎○△".index(lv), rank, float(q) * o, o, float(q) / mk,
                             pay, int(pay > 0 and not (ret & set(c)))))
    return rows


def main():
    RACES, ntrain = B.prepare(150, 90)
    rows = rows_of(RACES)
    days = sorted({r[0] for r in rows})
    cut = days[-60]
    A = np.array([[r[0] < cut, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]] for r in rows], dtype=float)
    fit, sn, isex, lv, rank, ev, o, ratio, pay, hit = A.T
    fit = fit.astype(bool)

    def stat(m):
        n = int(m.sum())
        return {"n": n, "hit": int(hit[m].sum()), "rr": float(pay[m].sum() / (100 * n)) if n else 0,
                "pl": int(pay[m].sum() - 100 * n)}

    conds = {
        "券種": [("2連単", isex == 1), ("3連単", isex == 0)],
        "自信度": [("全部", None), ("◎", lv == 0), ("○", lv == 1), ("△", lv == 2), ("○△", lv >= 1)],
        "予想順位": [("1位", rank == 1), ("2位まで", rank <= 2), ("3位まで", rank <= 3), ("全部", None)],
        "期待値": [("全部", None), ("1.0以上", ev >= 1.0), ("1.2以上", ev >= 1.2), ("1.5以上", ev >= 1.5), ("2.0以上", ev >= 2.0)],
        "オッズ": [("全部", None), ("5倍未満", o < 5), ("5〜10倍", (o >= 5) & (o < 10)), ("10〜20倍", (o >= 10) & (o < 20)),
                 ("20〜50倍", (o >= 20) & (o < 50)), ("50倍以上", o >= 50), ("10倍未満", o < 10), ("20倍以上", o >= 20)],
        "ズレ": [("全部", None), ("予想がオッズより1.2倍高い", ratio >= 1.2), ("1.5倍", ratio >= 1.5), ("2倍", ratio >= 2.0)],
    }
    keys = list(conds)
    res, seen = [], set()
    for combo in itertools.product(*[conds[k] for k in keys]):
        m = np.ones(len(pay), bool)
        for _, c in combo:
            if c is not None:
                m &= c
        mf, mt = m & fit, m & ~fit
        if mf.sum() < 800:
            continue
        sf = stat(mf)
        sig = (sf["n"], sf["pl"], int(mt.sum()))
        if sig in seen:
            continue
        seen.add(sig)
        res.append({"rule": {k: v[0] for k, v in zip(keys, combo)}, "探す期間": sf, "確かめる期間": stat(mt), "_m": m})
    # 会場ごと(券種×会場)
    for k_ex, kname in ((1, "2連単"), (0, "3連単")):
        for s in range(1, 25):
            m = (isex == k_ex) & (sn == s) & (rank <= (2 if k_ex else 3))
            mf, mt = m & fit, m & ~fit
            if mf.sum() < 800:
                continue
            res.append({"rule": {"券種": kname, "会場": bot.NAMES[s - 1], "予想順位": "アプリの買い目"},
                        "探す期間": stat(mf), "確かめる期間": stat(mt), "_m": m})
    res.sort(key=lambda r: -r["探す期間"]["rr"])
    top = res[:30]
    # 探す期間の上位30個を、確かめる期間で並べ直す
    for r in res:
        r.pop("_m", None)
    out = {"made": dt.datetime.now(bot.JST).strftime("%Y-%m-%d %H:%M"), "探す期間": [days[0].isoformat(), (dt.date.fromisoformat(cut.isoformat()) - dt.timedelta(days=1)).isoformat()],
           "確かめる期間": [cut.isoformat(), days[-1].isoformat()], "候補数": len(res),
           "基準": {"アプリの買い目 2連単": stat((isex == 1) & (rank <= 2) & ~fit), "アプリの買い目 3連単": stat((isex == 0) & (rank <= 3) & ~fit)},
           "上位30": top,
           "確かめる期間でも100%超": sum(1 for r in top if r["確かめる期間"]["n"] >= 100 and r["確かめる期間"]["rr"] >= 1.0),
           "全候補の確かめる期間の平均回収率": float(np.mean([r["確かめる期間"]["rr"] for r in res if r["確かめる期間"]["n"] >= 100]))}
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    for r in top[:15]:
        print(r["rule"], "探す", round(r["探す期間"]["rr"] * 100, 1), r["探す期間"]["n"], "→ 確かめる", round(r["確かめる期間"]["rr"] * 100, 1), r["確かめる期間"]["n"])


if __name__ == "__main__":
    main()
