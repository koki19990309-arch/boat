#!/usr/bin/env python3
"""ボートレース予想 自動通知ボット(GitHub Actions + ntfy)

使い方:
    python bot.py train     # 過去データで学習(1日1回)
    python bot.py notify    # 締切が近いレースの予想を通知
    python bot.py summary   # 今日の結果まとめを通知
環境変数:
    NTFY_TOPIC   通知先のトピック名(必須。他人に推測されない長い名前に)
    NISSU        学習日数(既定 90)
    SHIBORI      通知するレース: all / maru(○以上) / nijumaru(◎だけ)(既定 all)
    KAIJO        通知する会場をカンマ区切りで(空なら全会場)
"""
import os, sys, json, gzip, pickle, re, lzma, base64, glob
import datetime as dt
import numpy as np
import requests
import warnings
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.isotonic import IsotonicRegression as ISO
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.join(HERE, "cache")
os.makedirs(CDIR, exist_ok=True)


def cache_dir():
    return CDIR


JST = dt.timezone(dt.timedelta(hours=9))


B = "https://boatraceopenapi.github.io/"


NAMES = ("桐生 戸田 江戸川 平和島 多摩川 浜名湖 蒲郡 常滑 津 三国 びわこ 住之江 "
         "尼崎 鳴門 丸亀 児島 宮島 徳山 下関 若松 芦屋 福岡 唐津 大村").split()


KEYS = ["rank_number", "national_win_rate", "national_top_2_percent",
        "national_top_3_percent", "local_win_rate", "local_top_2_percent",
        "motor_top_2_percent", "motor_top_3_percent", "boat_top_2_percent",
        "average_start_timing", "flying_count", "late_count", "weight", "age"]


def get(u):
    try:
        r = requests.get(u, timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def f(x):
    try:
        return float(x)
    except Exception:
        return np.nan


HIST = {}


SEEN = set()


def hist_add(rs, d):
    """結果データを選手ごとの履歴に追加"""
    if not rs:
        return
    o = d.toordinal()
    for p in rs.get("results", []):
        sn, rn = p["race_stadium_number"], p["race_number"]
        for b in p["boats"]:
            num = b.get("racer_number")
            c = b.get("racer_course_number")
            if num is None or c is None:
                continue
            key = (o, sn, rn, num)
            if key in SEEN:
                continue
            try:
                pl = int(b.get("racer_place_number"))
            except Exception:
                pl = 6  # 失格・転覆など
            SEEN.add(key)
            HIST.setdefault(int(num), []).append((o, sn, rn, int(c), pl))


CACHE = {}


_DIRTY, _LOADED = set(), set()


STAT = {"drive": 0, "net": 0}


def _mfile(ym):
    return os.path.join(cache_dir(), f"m{ym}.pkl.gz")


def _month_load(ym):
    """ドライブの月まとめファイル(1か月分を1ファイル)を読み込む"""
    if ym in _LOADED:
        return
    _LOADED.add(ym)
    fn = _mfile(ym)
    if os.path.exists(fn):
        try:
            with gzip.open(fn, "rb") as fh:
                CACHE.update(pickle.load(fh))
        except Exception:
            pass


def cget(kind, d):
    """データを取得(確定した過去の日はドライブに月ごとにまとめて保存し、次回から再利用)"""
    d8 = d.strftime("%Y%m%d")
    _month_load(d8[:6])
    key = (kind, d8)
    if key in CACHE:
        STAT["drive"] += 1
        return CACHE[key]
    old = os.path.join(cache_dir(), f"{kind.split('/')[0]}_{d8}.json.gz")  # 以前の保存形式
    data = None
    if os.path.exists(old):
        try:
            with gzip.open(old, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
            STAT["drive"] += 1
        except Exception:
            data = None
    if data is None:
        data = get(B + f"{kind}/{d.strftime('%Y')}/{d8}.json")
        STAT["net"] += 1
    now = dt.datetime.now(JST)
    yd = now.date() - dt.timedelta(days=1)
    if data and (d < yd or (d == yd and now.hour >= 6)):
        CACHE[key] = data
        _DIRTY.add(d8[:6])
    return data


def cache_flush():
    """新しく取得した月の分だけ、ドライブに書き出す"""
    n = 0
    for ym in sorted(_DIRTY):
        part = {k: v for k, v in CACHE.items() if k[1][:6] == ym}
        tmp = _mfile(ym) + ".tmp"
        with gzip.open(tmp, "wb", compresslevel=5) as fh:
            pickle.dump(part, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, _mfile(ym))
        n += 1
    _DIRTY.clear()
    return n


def load_results(d):
    rs = cget("results/v2", d)
    hist_add(rs, d)
    return rs


def load(d):
    rs = load_results(d)
    return cget("api/v1", d), cget("previews/v2", d), rs


def stats(num, o, sn, rn, course):
    """今節(同じ会場の直近6日+今日の前のレース)とコース別の成績"""
    kn = ks = kw = cn = cw = c3 = 0
    for (do, s, r, c, pl) in HIST.get(num, []) if num is not None else []:
        if s == sn and (o - 6 <= do < o or (do == o and r < rn)):
            kn += 1; ks += pl; kw += pl == 1
        if do < o and c == course:
            cn += 1; cw += pl == 1; c3 += pl <= 3
    nan = np.nan
    return [kn, ks / kn if kn else nan, kw / kn if kn else nan,
            cn, cw / cn if cn else nan, c3 / cn if cn else nan]


def feats(sn, race, pvr):
    """1レース分の特徴量(6艇×項目)を作る"""
    rcs = race["racers"]
    pb, wind, wave, wdir = {}, np.nan, np.nan, np.nan
    if pvr:
        pb = {b["racer_boat_number"]: b for b in pvr["boats"]}
        wind, wave = f(pvr.get("race_wind")), f(pvr.get("race_wave"))
        wdir = f(pvr.get("race_wind_direction_number"))
    o = dt.date.fromisoformat(race["closed_at"][:10]).toordinal()
    rno = int(race.get("race_number") or 0)
    rows, ext = [], []
    for n in range(1, 7):
        rc = rcs.get(str(n))
        if rc is None:
            return None
        b = pb.get(n, {})
        et = f(b.get("racer_exhibition_time"))
        if not (et > 0):
            et = np.nan
        st = f(b.get("racer_start_timing"))
        cn = f(b.get("racer_course_number"))
        if not (cn >= 1):
            cn = n
        rows.append([sn, n, cn] + [f(rc.get(k)) for k in KEYS] + [et, st, wind, wave])
        num = rc.get("number")
        num = int(num) if num is not None else None
        ext.append([wdir] + stats(num, o, sn, rno, int(cn)) + fan_feats(num, o, cn))
    X = np.array(rows, dtype=float)
    E = np.array(ext, dtype=float)
    rel = []
    for M_, cols in ((X, (4, 7, 9, 12, 17, 18)), (E, (2, 5, 7, 9, 12))):
        for c in cols:  # レース内での相対値
            col = M_[:, c]
            rel.append(col - np.nanmean(col) if np.any(~np.isnan(col)) else col)
    return np.column_stack([X] + rel[:6] + [E[:, :7]] + rel[6:8] + [E[:, 7:]] + rel[8:])


def places(p):
    pl = {}
    for b in p["boats"]:
        try:
            pl[int(b["racer_place_number"])] = b["racer_boat_number"]
        except Exception:
            pass
    return pl


def pvmap(pv):
    return {(p["race_stadium_number"], p["race_number"]): p
            for p in (pv["previews"] if pv else [])}


def train(rows):
    Xs = np.vstack([r[1] for r in rows])
    ys = np.array([1 if r[2][1] == n else 0 for r in rows for n in range(1, 7)])
    m = HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=0)
    m.fit(Xs, ys)
    return m


def ml(m, X):
    p = m.predict_proba(X)[:, 1]
    p = p / p.sum()
    return [int(x) + 1 for x in np.argsort(p)[::-1]], p


def tri(p):
    """単勝確率から3連単の確率を計算(ハービル式)"""
    out = []
    for a in range(6):
        for b in range(6):
            if b == a:
                continue
            for c in range(6):
                if c in (a, b):
                    continue
                q = p[a] * p[b] / (1 - p[a]) * p[c] / (1 - p[a] - p[b])
                out.append((q, a + 1, b + 1, c + 1))
    out.sort(reverse=True)
    return out


def cal(c, p):
    q = c.predict(p)
    return q / q.sum()


FAN = []


def parse_fan(raw):
    """ファン手帳(1選手416バイト)から 能力指数 と コース別成績 を取り出す"""
    out, end = {}, None
    for r in raw.splitlines():
        if len(r) < 402 or not r[:4].isdigit():
            continue
        try:
            end = end or dt.date(int(r[187:191]), int(r[191:193]), int(r[193:195])).toordinal()
            v = [int(r[170:174]) / 100]
            for i in range(6):
                b, e = 82 + 13 * i, 198 + 34 * i
                cnt = [int(r[e + 3 * j:e + 3 * j + 3]) for j in range(6)]
                v += [int(r[b:b + 3]), int(r[b + 3:b + 7]) / 10, int(r[b + 7:b + 10]) / 100,
                      int(r[b + 10:b + 13]) / 100, cnt[0], sum(cnt[:3])]
            out[int(r[:4])] = np.array(v, float)
        except ValueError:
            continue
    return end, out


def fan_feats(num, o, course):
    """そのレースより前に締まった最新のファン手帳から、そのコースの成績を返す"""
    rec = None
    for end, d in FAN:
        if end < o and num is not None:
            rec = d.get(num, rec)
    nan = np.nan
    if rec is None:
        return [nan] * 7
    c = min(max(int(course), 1), 6)
    n, r2, st, sr, w, t3 = rec[1 + 6 * (c - 1):7 + 6 * (c - 1)]
    if n <= 0:
        return [rec[0], 0, nan, nan, nan, nan, nan]
    return [rec[0], n, w / n, r2, t3 / n, st, sr]


def ex2(p):
    """単勝確率から2連単の確率を計算"""
    out = [(p[a] * p[b] / (1 - p[a]), a + 1, b + 1)
           for a in range(6) for b in range(6) if a != b]
    out.sort(reverse=True)
    return out


def level(pw):
    return ("◎", "堅い") if pw >= 0.65 else ("○", "ふつう") if pw >= 0.5 else ("△", "荒れ注意")


# ======== ボット本体 ========
MODEL_FILE = os.path.join(CDIR, "model.pkl")
STATE_FILE = os.path.join(CDIR, "state.json")
TOPIC = os.environ.get("NTFY_TOPIC", "").strip()
SHIBORI = os.environ.get("SHIBORI", "all").strip() or "all"
KAIJO = [x.strip() for x in os.environ.get("KAIJO", "").replace("、", ",").split(",") if x.strip()]


def fan_init():
    FAN.clear()
    raw = open(os.path.join(HERE, "fan_data.txt")).read().replace("\n", "")
    t = lzma.decompress(base64.b64decode(raw)).decode().split("\n")
    FAN.append((int(t[0]), {int(x.split(",")[0]): np.array(x.split(",")[1:], float) for x in t[1:]}))


def send(title, msg, click=None, tags=None, prio=3):
    if not TOPIC:
        print("[通知先未設定]", title, "\n", msg)
        return
    body = {"topic": TOPIC, "title": title, "message": msg, "priority": prio}
    if click:
        body["click"] = click
    if tags:
        body["tags"] = tags
    try:
        r = requests.post("https://ntfy.sh/", json=body, timeout=20)
        print("送信", r.status_code, title)
    except Exception as e:
        print("送信失敗", e)


def load_state():
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {}


def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), ensure_ascii=False)


def build(nissu):
    """過去 nissu 日分で学習。予想用モデルと確率補正を返す"""
    T = dt.datetime.now(JST).date()
    N = min(int(nissu), (T - dt.date(2026, 1, 1)).days)
    HIST.clear(); SEEN.clear()
    fan_init()
    RS = {}
    for k in range(1, N + 67):
        r = load_results(T - dt.timedelta(days=k))
        if k <= N:
            RS[k] = r
    DATA = []
    for k in range(1, N + 1):
        d = T - dt.timedelta(days=k)
        rs = RS.pop(k, None)
        pg = cget("api/v1", d)
        pv = cget("previews/v2", d)
        if not pg or not rs:
            continue
        st = pg["programs"]["stadiums"]
        PV = pvmap(pv)
        for p in rs["results"]:
            key = (p["race_stadium_number"], p["race_number"])
            sn, rn = str(key[0]), str(key[1])
            if sn not in st or rn not in st[sn]["races"]:
                continue
            pl = places(p)
            if not all(i in pl for i in (1, 2, 3)):
                continue
            X = feats(key[0], st[sn]["races"][rn], PV.get(key))
            if X is not None:
                DATA.append((k, X, pl))
    cache_flush()
    TR = [r for r in DATA if r[0] > 15]
    TE = [r for r in DATA if r[0] <= 15]
    M = train(TR)
    P, Y = [], []
    for _, X, pl in TE:
        P += list(ml(M, X)[1])
        Y += [1 if pl[1] == n else 0 for n in range(1, 7)]
    CAL_ = ISO(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(P, Y)
    M2 = train(DATA)
    return M2, CAL_, len(DATA)


def cmd_train():
    nissu = int(os.environ.get("NISSU", "90") or 90)
    M2, CAL_, n = build(nissu)
    today = dt.datetime.now(JST).date().isoformat()
    pickle.dump({"date": today, "M2": M2, "CAL": CAL_, "n": n}, open(MODEL_FILE, "wb"))
    print(f"学習OK {today} {n}レース")


def get_model():
    today = dt.datetime.now(JST).date().isoformat()
    try:
        m = pickle.load(open(MODEL_FILE, "rb"))
        if m["date"] == today:
            return m
    except Exception:
        pass
    cmd_train()
    return pickle.load(open(MODEL_FILE, "rb"))


def today_history():
    """今日の前半のレース結果を今節成績に反映するため、履歴を読み直す"""
    HIST.clear(); SEEN.clear()
    fan_init()
    T = dt.datetime.now(JST).date()
    for k in range(0, 67):
        load_results(T - dt.timedelta(days=k))


def race_url(sn, rn, d):
    return (f"https://kyoteibiyori.com/race_shusso.php?place_no={sn}&race_no={rn}"
            f"&hiduke={d.strftime('%Y%m%d')}&slider=4")


def cmd_notify():
    m = get_model()
    M2, CAL_ = m["M2"], m["CAL"]
    today_history()
    now = dt.datetime.now(JST)
    d = now.date()
    pg = get(B + f"api/v1/{d:%Y}/{d:%Y%m%d}.json")
    pv = get(B + f"previews/v2/{d:%Y}/{d:%Y%m%d}.json")
    if not pg:
        print("今日のデータなし")
        return
    st = pg["programs"]["stadiums"]
    PV = pvmap(pv)
    state = load_state()
    sent = set(state.get(d.isoformat(), []))
    lo = now + dt.timedelta(minutes=int(os.environ.get("MAE_MIN", "5")))
    hi = now + dt.timedelta(minutes=int(os.environ.get("MAE_MAX", "45")))
    items = []
    for sn in st:
        nm = NAMES[int(sn) - 1]
        if KAIJO and nm not in KAIJO:
            continue
        for rn, race in st[sn]["races"].items():
            close = dt.datetime.fromisoformat(race["closed_at"]).replace(tzinfo=JST)
            key = f"{sn}-{rn}"
            if key in sent or not (lo <= close <= hi):
                continue
            items.append((close, sn, rn, race))
    items.sort()
    for close, sn, rn, race in items:
        X = feats(int(sn), race, PV.get((int(sn), int(rn))))
        if X is None:
            continue
        _, p0 = ml(M2, X)
        pc = cal(CAL_, p0)
        b = int(np.argmax(pc)) + 1
        pw = float(pc[b - 1])
        mk, lv = level(pw)
        sent.add(f"{sn}-{rn}")
        if SHIBORI == "nijumaru" and mk != "◎":
            continue
        if SHIBORI == "maru" and mk == "△":
            continue
        need = 1.2 if pw < 0.5 else 1.1
        tenji = "展示あり" if not np.any(np.isnan(X[:, 17])) else "展示前"
        lines = [f"本命 {b}号艇(1着{pw:.0%}) {mk}{lv} {tenji}", "【2連単】この倍率以上なら買い"]
        lines += [f"  {a}-{bb}  {need / q:.1f}倍〜 ({q:.0%})" for q, a, bb in ex2(p0)[:2]]
        lines += ["【3連単】"]
        lines += [f"  {a}-{bb}-{c}  {need / q:.0f}倍〜 ({q:.0%})" for q, a, bb, c in tri(p0)[:3]]
        lines += ["タップで直前情報"]
        nm = NAMES[int(sn) - 1]
        send(f"{close:%H:%M}締切 {nm}{rn}R {mk}", "\n".join(lines),
             click=race_url(sn, rn, d), prio=4 if mk == "◎" else 3)
    state = {d.isoformat(): sorted(sent)}
    save_state(state)
    print(f"通知 {len(items)}件をチェック")


def cmd_summary():
    m = get_model()
    M2 = m["M2"]
    today_history()
    d = dt.datetime.now(JST).date()
    pg = get(B + f"api/v1/{d:%Y}/{d:%Y%m%d}.json")
    pv = get(B + f"previews/v2/{d:%Y}/{d:%Y%m%d}.json")
    rs = get(B + f"results/v2/{d:%Y}/{d:%Y%m%d}.json")
    if not pg or not rs:
        send("今日の結果", "結果データがまだありません")
        return
    st = pg["programs"]["stadiums"]
    PV = pvmap(pv)
    n = h2 = b2 = h3 = b3 = 0
    per = {}
    for p in rs["results"]:
        key = (p["race_stadium_number"], p["race_number"])
        sn, rn = str(key[0]), str(key[1])
        if sn not in st or rn not in st[sn]["races"]:
            continue
        nm = NAMES[key[0] - 1]
        if KAIJO and nm not in KAIJO:
            continue
        pl = places(p)
        if not all(i in pl for i in (1, 2, 3)):
            continue
        X = feats(key[0], st[sn]["races"][rn], PV.get(key))
        if X is None:
            continue
        _, p0 = ml(M2, X)
        res = (pl[1], pl[2], pl[3])
        c2 = [(a, b) for q, a, b in ex2(p0)[:2]]
        c3 = [(a, b, c) for q, a, b, c in tri(p0)[:3]]
        n += 1
        pr = per.setdefault(nm, [0, 0])
        pr[0] += 1
        if res[:2] in c2:
            h2 += 1; pr[1] += 1
            b2 += sum(x["payout"] for x in p["payouts"].get("exacta", []) if x["combination"] == "%d-%d" % res[:2])
        if res in c3:
            h3 += 1
            b3 += sum(x["payout"] for x in p["payouts"].get("trifecta", []) if x["combination"] == "%d-%d-%d" % res)
    if not n:
        send("今日の結果", "結果が出たレースがありません")
        return
    msg = [f"{n}レース(各100円で全部買った場合)",
           f"2連単2点: 的中{h2} 回収率{b2 / (n * 200):.0%}",
           f"3連単3点: 的中{h3} 回収率{b3 / (n * 300):.0%}", ""]
    msg += [f"{k}: {v[1]}/{v[0]}R 的中(2単)" for k, v in per.items()]
    send(f"今日の結果 {d:%m/%d}", "\n".join(msg), tags=["bar_chart"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "notify"
    {"train": cmd_train, "notify": cmd_notify, "summary": cmd_summary}[cmd]()
