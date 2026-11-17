#!/usr/bin/env python3
"""ボートレース予想アプリ用のデータを書き出す(GitHub Actions で実行)

    python app.py update    # モデル書き出し・今日のデータ・成績の記録・(設定があれば)通知
    python app.py summary   # 今日の結果まとめを通知(NTFY_TOPIC があれば)
"""
import os, sys, json, csv, datetime as dt
import numpy as np
import bot

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
REC = os.path.join(DATA, "records.csv")
RCOLS = ["日付", "会場", "場番号", "R", "自信", "本命", "確率", "2単予想", "3単予想",
         "結果", "2単払戻", "3単払戻", "2単的中払戻", "3単的中払戻", "返還"]
CLS = {1: "A1", 2: "A2", 3: "B1", 4: "B2"}
bot.STATE_FILE = os.path.join(DATA, "notify_state.json")  # 通知済みの記録はリポジトリに残す


def num(x, d=None):
    """JSONに入れる数値(NaN は null)。予想と同じ結果になるよう、既定では丸めない"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return round(x, d) if d is not None else x


def export_model(m):
    M2, CAL = m["M2"], m["CAL"]
    trees = []
    for preds in M2._predictors:
        nd = preds[0].nodes
        trees.append([nd["feature_idx"].tolist(),
                      [float(v) for v in nd["num_threshold"]],
                      nd["missing_go_to_left"].astype(int).tolist(),
                      nd["left"].tolist(), nd["right"].tolist(),
                      nd["is_leaf"].astype(int).tolist(),
                      [float(v) for v in nd["value"]]])
    out = {"date": m["date"], "base": float(np.ravel(M2._baseline_prediction)[0]), "trees": trees,
           "cal_x": [float(v) for v in CAL.X_thresholds_], "cal_y": [float(v) for v in CAL.y_thresholds_],
           "n_races": m.get("n", 0)}
    json.dump(out, open(os.path.join(DATA, "model.json"), "w"), separators=(",", ":"))


def pv_auto(pvr):
    if not pvr:
        return None
    boats = {}
    for b in pvr["boats"]:
        et = num(b.get("racer_exhibition_time"))
        boats[str(b["racer_boat_number"])] = {
            "et": et if et and et > 0 else None,
            "st": num(b.get("racer_start_timing")),
            "cn": b.get("racer_course_number")}
    has = sum(1 for v in boats.values() if v["et"]) == 6
    return {"wind": num(pvr.get("race_wind")), "wave": num(pvr.get("race_wave")),
            "wdir": num(pvr.get("race_wind_direction_number")), "boats": boats, "has": has}


def result_of(p):
    pl = bot.places(p)
    if not all(i in pl for i in (1, 2, 3)):
        return None
    o = "%d-%d-%d" % (pl[1], pl[2], pl[3])
    ex = sum(x["payout"] for x in p["payouts"].get("exacta", []) if x["combination"] == o[:3])
    tr = sum(x["payout"] for x in p["payouts"].get("trifecta", []) if x["combination"] == o)
    ret = ""   # 返還艇(フライング・出遅れ・欠場)
    for b in p["boats"]:
        try:
            if int(b["racer_place_number"]) in (14, 15, 16):
                ret += str(b["racer_boat_number"])
        except Exception:
            pass
    return {"o": o, "ex": ex, "tri": tr, "ret": "".join(sorted(ret))}


def back(combos, hit, pay, ret):
    """買い目ごとの払戻(100円あたり)の合計。返還艇を含む買い目は100円戻る"""
    return sum(pay if c == hit else (100 if any(ch in ret for ch in c.split("-")) else 0) for c in combos)


def race_json(sn, rn, race, pvr, d):
    o = d.toordinal()
    boats = []
    for n in range(1, 7):
        rc = race["racers"].get(str(n))
        if rc is None:
            return None
        nm = rc.get("number")
        nm = int(nm) if nm is not None else None
        boats.append({
            "n": n, "name": rc.get("name", ""), "cls": CLS.get(rc.get("rank_number"), ""),
            "base": [num(rc.get(k)) for k in bot.KEYS],
            "cs": [[num(v) for v in bot.stats(nm, o, int(sn), int(rn), c)] for c in range(1, 7)],
            "fan": [[num(v) for v in bot.fan_feats(nm, o, c)] for c in range(1, 7)]})
    return {"sn": int(sn), "name": bot.NAMES[int(sn) - 1], "rn": int(rn),
            "close": race["closed_at"][11:16], "boats": boats, "pv": pv_auto(pvr)}


def load_records():
    if not os.path.exists(REC):
        return []
    with open(REC, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def save_records(rows):
    rows.sort(key=lambda r: (r["日付"], int(r["場番号"]), int(r["R"])))
    with open(REC, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RCOLS)
        w.writeheader()
        w.writerows(rows)


def update_records(M2, CAL, days):
    rows = {(r["日付"], r["場番号"], r["R"]): r for r in load_records()}
    for d in days:
        pg = bot.get(bot.B + f"api/v1/{d:%Y}/{d:%Y%m%d}.json")
        pv = bot.get(bot.B + f"previews/v2/{d:%Y}/{d:%Y%m%d}.json")
        rs = bot.get(bot.B + f"results/v2/{d:%Y}/{d:%Y%m%d}.json")
        if not pg or not rs:
            continue
        st = pg["programs"]["stadiums"]
        PV = bot.pvmap(pv)
        for p in rs["results"]:
            key = (p["race_stadium_number"], p["race_number"])
            sn, rn = str(key[0]), str(key[1])
            if sn not in st or rn not in st[sn]["races"] or (d.isoformat(), sn, rn) in rows:
                continue
            res = result_of(p)
            if not res:
                continue
            X = bot.feats(key[0], st[sn]["races"][rn], PV.get(key))
            if X is None:
                continue
            _, p0 = bot.ml(M2, X)
            pc = bot.cal(CAL, p0)
            b = int(np.argmax(pc)) + 1
            c2 = ["%d-%d" % (a, bb) for q, a, bb in bot.ex2(p0)[:2]]
            c3 = ["%d-%d-%d" % (a, bb, c) for q, a, bb, c in bot.tri(p0)[:3]]
            rows[(d.isoformat(), sn, rn)] = {
                "日付": d.isoformat(), "会場": bot.NAMES[key[0] - 1], "場番号": sn, "R": rn,
                "自信": bot.level(float(pc[b - 1]))[0], "本命": b, "確率": round(float(pc[b - 1]), 3),
                "2単予想": " ".join(c2), "3単予想": " ".join(c3), "結果": res["o"],
                "2単払戻": back(c2, res["o"][:3], res["ex"], res["ret"]),
                "3単払戻": back(c3, res["o"], res["tri"], res["ret"]),
                "2単的中払戻": res["ex"], "3単的中払戻": res["tri"], "返還": res["ret"]}
    save_records(list(rows.values()))
    return list(rows.values())


def make_stats(rows):
    def agg(rs):
        n = len(rs)
        if not n:
            return None
        e2 = [float(r["2単払戻"]) for r in rs]
        e3 = [float(r["3単払戻"]) for r in rs]
        h2 = sum(1 for r in rs if r["結果"][:3] in r["2単予想"].split())
        h3 = sum(1 for r in rs if r["結果"] in r["3単予想"].split())
        return {"n": n, "h2": h2, "r2": sum(e2) / (200 * n),
                "h3": h3, "r3": sum(e3) / (300 * n),
                "b2": 200 * n, "p2": sum(e2), "b3": 300 * n, "p3": sum(e3)}
    by = lambda k: {v: agg([r for r in rows if r[k] == v]) for v in sorted({r[k] for r in rows})}
    days = sorted({r["日付"] for r in rows})
    res = {}
    for r in rows:
        if r["日付"] >= (dt.date.fromisoformat(days[-1]) - dt.timedelta(days=40)).isoformat():
            res.setdefault(r["日付"], {})[f'{r["場番号"]}-{r["R"]}'] = [r["結果"], float(r["2単的中払戻"]), float(r["3単的中払戻"]), r.get("返還") or ""]
    return {"all": agg(rows), "level": by("自信"), "venue": by("会場"),
            "days": {d: agg([r for r in rows if r["日付"] == d]) for d in days[-14:]},
            "results": res, "n_days": len(days),
            "hits": [[r["日付"], r["会場"], int(r["R"]), r["自信"], r["結果"],
                      float(r["2単的中払戻"]) if r["結果"][:3] in r["2単予想"].split() else 0,
                      float(r["3単的中払戻"]) if r["結果"] in r["3単予想"].split() else 0]
                     for r in rows if r["結果"][:3] in r["2単予想"].split() or r["結果"] in r["3単予想"].split()][-60:]}


def cmd_update():
    m = bot.get_model()
    mpath = os.path.join(DATA, "model.json")
    try:
        old = json.load(open(mpath)).get("date")
    except Exception:
        old = None
    if old != m["date"]:
        export_model(m)
    bot.today_history()
    now = dt.datetime.now(bot.JST)
    d = now.date()
    pg = bot.get(bot.B + f"api/v1/{d:%Y}/{d:%Y%m%d}.json")
    pv = bot.get(bot.B + f"previews/v2/{d:%Y}/{d:%Y%m%d}.json")
    rs = bot.get(bot.B + f"results/v2/{d:%Y}/{d:%Y%m%d}.json")
    RS = {(p["race_stadium_number"], p["race_number"]): p for p in (rs or {}).get("results", [])}
    races = []
    if pg:
        st = pg["programs"]["stadiums"]
        PV = bot.pvmap(pv)
        for sn in st:
            for rn, race in st[sn]["races"].items():
                rj = race_json(sn, rn, race, PV.get((int(sn), int(rn))), d)
                if rj:
                    r = RS.get((int(sn), int(rn)))
                    rj["res"] = result_of(r) if r else None
                    races.append(rj)
    races.sort(key=lambda r: (r["close"], r["sn"]))
    json.dump({"date": d.isoformat(), "updated": now.strftime("%H:%M"), "races": races},
              open(os.path.join(DATA, "today.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    rows = update_records(m["M2"], m["CAL"], [d - dt.timedelta(days=1), d])
    json.dump(make_stats(rows), open(os.path.join(DATA, "stats.json"), "w"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"今日 {len(races)}レース / 記録 {len(rows)}レース")
    if bot.TOPIC:
        bot.cmd_notify()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "update"
    {"update": cmd_update, "summary": bot.cmd_summary}[cmd]()
