"""
過去データの書き出し(2026/1/1〜昨日の全レース)
  races.csv   : 1艇1行。出走表・展示(締切前の情報) と 結果(result_ で始まる列。予想に使ってはいけない)
  payouts.csv : 1レース1行。2連単・3連単・2連複・3連複の的中組と払戻、返還艇
  列の説明.md
使い方: python export.py → export/boatrace_2026.zip
"""
import csv
import datetime as dt
import os
import zipfile

import backtest as B
import bot

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "export")
GRADE = {1: "A1", 2: "A2", 3: "B1", 4: "B2"}
RCOLS = [("rank_number", "racer_grade_num"), ("national_win_rate", "national_win_rate"),
         ("national_top_2_percent", "national_top2_rate"), ("national_top_3_percent", "national_top3_rate"),
         ("local_win_rate", "local_win_rate"), ("local_top_2_percent", "local_top2_rate"),
         ("motor_top_2_percent", "motor_top2_rate"), ("motor_top_3_percent", "motor_top3_rate"),
         ("boat_top_2_percent", "boat_top2_rate"), ("average_start_timing", "avg_st"),
         ("flying_count", "flying_count"), ("late_count", "late_count"), ("weight", "weight"), ("age", "age")]
HEAD = (["race_id", "race_date", "venue", "venue_no", "race_no", "close_time", "lane", "racer_number", "racer_name", "racer_grade"]
        + [n for _, n in RCOLS]
        + ["exhibition_time", "exhibition_st", "exhibition_course", "wind_speed", "wind_direction_no", "wave_height",
           "meet_races", "meet_avg_place", "meet_win_rate", "course_races", "course_win_rate", "course_top3_rate",
           "result_course", "result_st", "result_place_code", "finish"])
PHEAD = ["race_id", "race_date", "venue", "race_no", "exacta", "exacta_pay", "trifecta", "trifecta_pay",
         "quinella", "quinella_pay", "trio", "trio_pay", "refund_lanes"]


def v(x):
    return "" if x is None else x


def main():
    T = dt.datetime.now(bot.JST).date()
    first = dt.date(2026, 1, 1)
    days = [first + dt.timedelta(days=i) for i in range((T - first).days)]
    bot.HIST.clear(); bot.SEEN.clear()
    RS = {d: bot.load_results(d) for d in days}   # 今節・コース別成績の履歴(そのレースより前だけ使う)
    os.makedirs(OUTD, exist_ok=True)
    fr = open(os.path.join(OUTD, "races.csv"), "w", encoding="utf-8-sig", newline="")
    fp = open(os.path.join(OUTD, "payouts.csv"), "w", encoding="utf-8-sig", newline="")
    wr, wp = csv.writer(fr), csv.writer(fp)
    wr.writerow(HEAD); wp.writerow(PHEAD)
    nr = 0
    for d in days:
        pg = bot.cget("api/v1", d)
        pv = bot.cget("previews/v2", d)
        rs = RS.get(d)
        tm = B.tm_odds(d) or {}
        if not pg:
            continue
        PV = bot.pvmap(pv)
        RES = {(p["race_stadium_number"], p["race_number"]): p for p in (rs or {}).get("results", [])}
        o = d.toordinal()
        for sn_s, s in pg["programs"]["stadiums"].items():
            sn = int(sn_s)
            for rn_s, race in s["races"].items():
                rn = int(rn_s)
                rid = f"{d:%Y%m%d}_{sn:02d}_{rn:02d}"
                pvr = PV.get((sn, rn)) or {}
                pb = {b["racer_boat_number"]: b for b in pvr.get("boats", [])}
                res = RES.get((sn, rn)) or {}
                rb = {b["racer_boat_number"]: b for b in res.get("boats", [])}
                for n in range(1, 7):
                    rc = race["racers"].get(str(n))
                    if rc is None:
                        continue
                    b, r = pb.get(n, {}), rb.get(n, {})
                    num = rc.get("number")
                    cn = b.get("racer_course_number") or n
                    st = bot.stats(int(num) if num else None, o, sn, rn, int(cn))
                    code = r.get("racer_place_number")
                    try:
                        fin = int(code) if code is not None and int(code) <= 6 else ""
                    except Exception:
                        fin = ""
                    et = b.get("racer_exhibition_time")
                    wr.writerow([rid, d.isoformat(), bot.NAMES[sn - 1], sn, rn, (race.get("closed_at") or "")[11:16], n, v(num),
                                 v(rc.get("name")), GRADE.get(rc.get("rank_number"), "")]
                                + [v(rc.get(k)) for k, _ in RCOLS]
                                + [et if et and et > 0 else "", v(b.get("racer_start_timing")), v(b.get("racer_course_number")),
                                   v(pvr.get("race_wind")), v(pvr.get("race_wind_direction_number")), v(pvr.get("race_wave"))]
                                + ["" if x != x else round(x, 4) for x in st]
                                + [v(r.get("racer_course_number")), v(r.get("racer_start_timing")), v(code), fin])
                t = tm.get((sn, rn))
                if t and t.get("pay"):
                    P = t["pay"]
                    one = lambda k: next(iter(P.get(k, {}).items()), ("", ""))
                    row = [rid, d.isoformat(), bot.NAMES[sn - 1], rn]
                    for k in ("exacta", "trifecta", "quinella", "trio"):
                        row += list(one(k))
                    row.append("".join(map(str, t.get("ref") or [])))
                    wp.writerow(row)
                nr += 1
    fr.close(); fp.close()
    bot.cache_flush()
    doc = """# 列の説明(2026/1/1〜{last}、{n}レース)

## races.csv(1艇1行)
- race_id: 日付_場番号_R(例 20260101_02_01)/ venue: 会場名 / venue_no: 場番号(1桐生〜24大村)/ close_time: 締切時刻
- lane: 艇番(枠番)/ racer_number・racer_name・racer_grade(A1/A2/B1/B2)/ racer_grade_num(1=A1〜4=B2)
- 出走表(締切前に分かる): national_win_rate 全国勝率, national_top2_rate/top3_rate 全国2連率/3連率, local_* 当地,
  motor_top2_rate/top3_rate モーター, boat_top2_rate ボート, avg_st 平均ST, flying_count F数, late_count L数, weight 体重, age 年齢
- 展示(締切前に分かる): exhibition_time 展示タイム, exhibition_st 展示ST(マイナスはフライング), exhibition_course 展示の進入コース,
  wind_speed 風速m, wind_direction_no 風向番号(1〜16, スタートラインに対する向き), wave_height 波高cm
- 過去成績(そのレースより前のレースだけで集計・リークなし): meet_* 同じ会場の直近6日(今節), course_* 同じコースでの成績(1月以降)
- 結果(★予想の特徴量に使わないこと): result_course 本番コース, result_st 本番ST, result_place_code 着順コード(1〜6着, 7〜13失格, 14F, 15L, 16欠場), finish 着順(1〜6, それ以外は空)

## payouts.csv(1レース1行)
- exacta 2連単, trifecta 3連単, quinella 2連複, trio 3連複 の 的中組 と 払戻(100円あたり)
- refund_lanes: 返還艇(例 "12" = 1号艇と2号艇)。返還艇を含む買い目は100円戻る

## 出典
- 出走表・展示・結果: Boatrace Open API(https://boatraceopenapi.github.io/)
- 払戻: Turnmark API(https://turnmark.github.io/)
- どちらも非公式。オッズは含めていない(確定オッズは締切後の情報なので、買い方の判断には使えない)
""".format(last=days[-1].isoformat(), n=nr)
    open(os.path.join(OUTD, "列の説明.md"), "w", encoding="utf-8").write(doc)
    with zipfile.ZipFile(os.path.join(OUTD, "boatrace_2026.zip"), "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in ("races.csv", "payouts.csv", "列の説明.md"):
            z.write(os.path.join(OUTD, f), f)
    print("races", nr, os.path.getsize(os.path.join(OUTD, "boatrace_2026.zip")))


if __name__ == "__main__":
    main()
