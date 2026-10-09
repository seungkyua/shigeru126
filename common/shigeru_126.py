#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시게루(藤本茂) 1:2:6 매매법 신호 알리미
=======================================

[매수]  (총 9단위 = 1 + 2 + 6, 순서대로 진행)
  1단계: RSI(14)가 30 아래로 떨어지면          -> 1 매수
  2단계: MACD가 시그널선을 골든크로스하면       -> 2 매수
  3단계: 종가가 일목균형표 구름 상단을 돌파하면 -> 6 매수

[매도]  (6까지 모두 매수한 '풀포지션' 이후에만 작동)
  - MACD 데드크로스                     -> 1 매도
  - RSI(14)가 50 아래로 떨어지면         -> 2 매도
  - 종가가 구름을 이탈하면               -> 남은 물량 전량 매도

사용법
  pip install yfinance pandas numpy
  python common/shigeru_126.py 005930 000660 AAPL 7203.T
  python common/shigeru_126.py 005930 --since 2026-08-01 --held 9   # 실제 보유 상태에서 감시
  python common/shigeru_126.py 005930 --exit top                    # 구름 '상단' 하회를 이탈로 간주
  python common/shigeru_126.py 005930 --amount 9000000              # 종목당 투자금 → 단계별 매수 금액/주수
  python common/shigeru_126.py --csv mydata.csv                     # Date,Open,High,Low,Close 컬럼 CSV
  (프로젝트 루트에서 실행 기준. python -m common.shigeru_126 ... 도 가능)

※ 투자 판단의 참고용입니다. 실제 매매 책임은 본인에게 있습니다.
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd

UNIT_BUY = {1: 1, 2: 2, 3: 6}
TOTAL_UNITS = 9


# ---------------------------------------------------------------- 데이터
def load_yahoo(code: str, period: str = "3y") -> tuple[pd.DataFrame, str]:
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance가 필요합니다:  pip install yfinance")
    # 코스피(.KS)에 없는 코스닥 종목을 찾을 때 yfinance가 찍는 404 오류 로그를 숨긴다
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    # 한국 6자리 종목코드면 코스피(.KS) -> 코스닥(.KQ) 순으로 시도
    candidates = [code]
    if code.isdigit() and len(code) == 6:
        candidates = [code + ".KS", code + ".KQ"]

    for sym in candidates:
        df = yf.download(sym, period=period, interval="1d",
                         auto_adjust=False, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close"]].dropna()
            if len(df) > 0:
                return df, sym
    raise ValueError(f"'{code}' 데이터를 찾을 수 없습니다.")


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close"]].dropna()


# ---------------------------------------------------------------- 지표
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close, high, low = d["Close"], d["High"], d["Low"]

    # RSI(14) - Wilder 방식
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = (100 - 100 / (1 + rs)).fillna(100)

    # MACD(12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["SIGNAL"] = d["MACD"].ewm(span=9, adjust=False).mean()

    # 일목균형표(9, 26, 52) - 오늘 날짜에 그려지는 구름 = 26일 전에 계산된 선행스팬
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    d["CLOUD_TOP"] = pd.concat([span_a, span_b], axis=1).max(axis=1)
    d["CLOUD_BOT"] = pd.concat([span_a, span_b], axis=1).min(axis=1)

    p = d.shift(1)  # 전일 값
    d["RSI_DOWN30"] = (p["RSI"] >= 30) & (d["RSI"] < 30)
    d["RSI_DOWN50"] = (p["RSI"] >= 50) & (d["RSI"] < 50)
    d["MACD_GC"] = (p["MACD"] <= p["SIGNAL"]) & (d["MACD"] > d["SIGNAL"])
    d["MACD_DC"] = (p["MACD"] >= p["SIGNAL"]) & (d["MACD"] < d["SIGNAL"])
    d["CLOUD_UP"] = (p["Close"] <= p["CLOUD_TOP"]) & (d["Close"] > d["CLOUD_TOP"])
    return d.dropna(subset=["CLOUD_TOP", "CLOUD_BOT"])


# ---------------------------------------------------------------- 1:2:6 상태머신
class Position:
    def __init__(self, held: int = 0):
        # stage: 0=없음, 1=1매수, 2=1+2매수, 3=풀포지션(9)
        self.units = held
        self.stage = {0: 0, 1: 1, 3: 2, 9: 3}.get(held, 3 if held else 0)
        self.sold_macd = False
        self.sold_rsi = False

    def reset(self):
        self.__init__(0)


def cloud_exit(row, mode: str) -> bool:
    level = row["CLOUD_BOT"] if mode == "bottom" else row["CLOUD_TOP"]
    return row["Close"] < level


def step(pos: Position, row, exit_mode: str) -> list[tuple[str, str, int]]:
    """하루치 신호 처리. (구분, 사유, 수량) 리스트 반환"""
    acts = []

    # ----- 매도 (풀포지션 이후에만) -----
    if pos.stage == 3 and pos.units > 0:
        if cloud_exit(row, exit_mode):
            acts.append(("매도", "일목 구름 이탈 → 전량", pos.units))
            pos.reset()
            return acts
        if row["MACD_DC"] and not pos.sold_macd:
            q = min(1, pos.units)
            pos.units -= q
            pos.sold_macd = True
            acts.append(("매도", "MACD 데드크로스", q))
        if row["RSI"] < 50 and not pos.sold_rsi:
            q = min(2, pos.units)
            pos.units -= q
            pos.sold_rsi = True
            acts.append(("매도", f"RSI 50 하회 ({row['RSI']:.1f})", q))
        return acts

    # ----- 매수 (순서대로) -----
    if pos.stage == 0 and row["RSI_DOWN30"]:
        pos.stage, pos.units = 1, pos.units + UNIT_BUY[1]
        acts.append(("매수", f"RSI 30 하회 ({row['RSI']:.1f})", UNIT_BUY[1]))
    elif pos.stage == 1 and row["MACD_GC"]:
        pos.stage, pos.units = 2, pos.units + UNIT_BUY[2]
        acts.append(("매수", "MACD 골든크로스", UNIT_BUY[2]))
    elif pos.stage == 2 and row["CLOUD_UP"]:
        pos.stage, pos.units = 3, pos.units + UNIT_BUY[3]
        acts.append(("매수", "구름 상단 돌파", UNIT_BUY[3]))
    return acts


def run(d: pd.DataFrame, since=None, held=0, exit_mode="bottom"):
    if since is not None:
        d = d[d.index >= pd.Timestamp(since)]
    pos = Position(held)
    log = []
    for dt, row in d.iterrows():
        for kind, why, q in step(pos, row, exit_mode):
            log.append((dt, kind, why, q, row["Close"]))
    return pos, log, d


# ---------------------------------------------------------------- 출력
def buy_money(amount: float, q: int, price: float) -> tuple[float, int]:
    """총 투자금 중 q/9에 해당하는 금액과, 그 금액으로 price에 살 수 있는 주수."""
    money = amount * q / TOTAL_UNITS
    return money, int(money // price)


def buy_hint(pos: Position, last) -> str:
    """매수 단계(0~2)에서 다음에 기다려야 할 조건."""
    if pos.stage == 0:
        return f"대기 중 — 다음 매수 조건: RSI 30 하회 (현재 {last['RSI']:.1f})"
    if pos.stage == 1:
        gap = last["MACD"] - last["SIGNAL"]
        return f"1/9 보유 — 다음 매수 조건: MACD 골든크로스 (MACD-시그널 {gap:+.3f})"
    return (f"3/9 보유 — 다음 매수 조건: 종가가 구름 상단 {last['CLOUD_TOP']:,.2f} 돌파 "
            f"(현재 {last['Close']:,.2f})")


def next_hint(pos: Position, last, exit_mode: str, amount: float | None = None) -> str:
    if pos.stage < 3:
        hint = buy_hint(pos, last)
        if amount:
            # 다음 단계에서 살 금액과, 현재가 기준으로 살 수 있는 주수
            money, shares = buy_money(amount, UNIT_BUY[pos.stage + 1], last["Close"])
            hint += f" / 다음 매수 금액 {money:,.0f} (현재가 기준 약 {shares:,}주)"
        return hint
    lvl = last["CLOUD_BOT"] if exit_mode == "bottom" else last["CLOUD_TOP"]
    parts = []
    if not pos.sold_macd:
        parts.append(f"MACD 데드크로스(차이 {last['MACD'] - last['SIGNAL']:+.3f})")
    if not pos.sold_rsi:
        parts.append(f"RSI 50 하회(현재 {last['RSI']:.1f})")
    parts.append(f"구름 이탈선 {lvl:,.2f} (현재가 {last['Close']:,.2f}, "
                 f"여유 {(last['Close'] / lvl - 1) * 100:+.1f}%)")
    return f"{pos.units}/9 보유 — 매도 감시: " + " / ".join(parts)


def format_today(buys: list, sells: list, amount: float | None = None,
                 held_before: int = TOTAL_UNITS) -> list[str]:
    """오늘(마지막 거래일) 발생한 매수/매도 신호를 눈에 띄는 블록으로 만든다.
    buys, sells 원소는 (구분, 사유, 수량, 종가) 튜플.
    amount(종목당 총 투자금)가 있으면 매수는 금액·주수를, 매도는 보유 수량 대비 비율을 덧붙인다.
    held_before는 오늘 매도 전 보유 단위 수."""
    lines = []
    if sells:
        lines.append("\n  🚨🚨 오늘 매도 신호 🚨🚨")
        held = held_before
        for _, why, q, px in sells:
            line = f"     ▶ {why}: {q}/9 매도  (종가 {px:,.2f})"
            if amount:
                line += "  → 보유 수량 전량" if q >= held else f"  → 보유 수량의 {q}/{held}"
            lines.append(line)
            held -= q
    if buys:
        lines.append("\n  🟢🟢 오늘 매수 신호 🟢🟢")
        for _, why, q, px in buys:
            line = f"     ▶ {why}: {q}/9 매수  (종가 {px:,.2f})"
            if amount:
                money, shares = buy_money(amount, q, px)
                line += f"  → {money:,.0f} 매수 (약 {shares:,}주)"
            lines.append(line)
    if not buys and not sells:
        lines.append("\n  오늘 신호 없음")
    return lines


def format_summary(results: list[dict]) -> list[str]:
    """여러 종목을 조회한 뒤, 오늘 신호가 난 종목을 매수/매도별로 모아 보여준다."""
    lines = ["#" * 70, "  오늘의 신호 요약"]
    for title, key, kind in (("🚨 매도 신호", "sells", "매도"),
                             ("🟢 매수 신호", "buys", "매수")):
        hits = [r for r in results if r[key]]
        lines.append(f"  {title} ({len(hits)}종목)")
        for r in hits:
            detail = ", ".join(f"{why} {q}/9 {kind}" for _, why, q, _ in r[key])
            lines.append(f"    - {r['name']}: {detail} → 현재 {r['units']}/9 보유")
    quiet = [r["name"] for r in results if not r["buys"] and not r["sells"]]
    if quiet:
        lines.append(f"  신호 없음: {', '.join(quiet)}")
    lines.append("#" * 70)
    return lines


def report(name: str, d: pd.DataFrame, args) -> dict | None:
    """한 종목의 상태를 출력하고, 요약용으로 오늘의 신호를 반환한다 (데이터 부족 시 None)."""
    pos, log, d = run(d, args.since, args.held, args.exit)
    if d.empty:
        print(f"[{name}] 데이터 부족"); return None
    last, last_dt = d.iloc[-1], d.index[-1]

    print("=" * 70)
    print(f"■ {name}   기준일 {last_dt:%Y-%m-%d}   종가 {last['Close']:,.2f}")
    print(f"  RSI {last['RSI']:.1f} | MACD {last['MACD']:.3f} / 시그널 {last['SIGNAL']:.3f} | "
          f"구름 {last['CLOUD_BOT']:,.2f} ~ {last['CLOUD_TOP']:,.2f}")

    # 마지막 거래일에 발생한 신호만 골라 (구분, 사유, 수량, 종가) 형태로
    today = [x[1:] for x in log if x[0] == last_dt]
    buys = [x for x in today if x[0] == "매수"]
    sells = [x for x in today if x[0] == "매도"]
    # 오늘 매도 전 보유 단위 = 현재 보유 + 오늘 판 수량 (전량 매도 시 현재 보유는 0)
    held_before = pos.units + sum(q for _, _, q, _ in sells)
    for line in format_today(buys, sells, args.amount, held_before):
        print(line)

    print("  " + next_hint(pos, last, args.exit, args.amount))

    if log:
        print(f"\n  최근 신호 기록 (최대 {args.n}건)")
        for dt, k, why, q, px in log[-args.n:]:
            print(f"    {dt:%Y-%m-%d}  {k}  {q}/9  {px:>12,.2f}  {why}")
    print()
    return {"name": name, "units": pos.units, "buys": buys, "sells": sells}


def positive_amount(text: str) -> float:
    """--amount 값 검증: 0보다 큰 숫자만 허용 (쉼표 허용, 예: 9,000,000)."""
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        raise argparse.ArgumentTypeError(f"숫자가 아닙니다: {text}")
    if value <= 0:
        raise argparse.ArgumentTypeError("투자금은 0보다 커야 합니다.")
    return value


def main():
    ap = argparse.ArgumentParser(description="시게루 1:2:6 매매 신호 알리미")
    ap.add_argument("tickers", nargs="*", help="예: 005930 AAPL 7203.T")
    ap.add_argument("--csv", help="Date,Open,High,Low,Close 컬럼의 CSV 파일")
    ap.add_argument("--since", help="이 날짜부터 시뮬레이션 (YYYY-MM-DD)")
    ap.add_argument("--held", type=int, default=0, choices=[0, 1, 3, 9],
                    help="--since 시점의 실제 보유 단위 (0/1/3/9)")
    ap.add_argument("--exit", default="bottom", choices=["bottom", "top"],
                    help="구름 이탈 기준: bottom=구름 하단 하회(기본), top=구름 상단 하회")
    ap.add_argument("--amount", type=positive_amount,
                    help="종목당 총 투자금(종목 통화 기준). 지정하면 단계별 매수 금액·주수를 표시")
    ap.add_argument("--n", type=int, default=10, help="표시할 최근 신호 개수")
    args = ap.parse_args()

    if args.held and not args.since:
        ap.error("--held 를 쓰려면 --since 날짜도 함께 지정하세요.")
    if not args.tickers and not args.csv:
        ap.error("종목코드나 --csv 를 입력하세요.")

    results = []
    if args.csv:
        results.append(report(args.csv, add_indicators(load_csv(args.csv)), args))
    for t in args.tickers:
        try:
            raw, sym = load_yahoo(t.upper())
            results.append(report(sym, add_indicators(raw), args))
        except Exception as e:
            print(f"[{t}] 오류: {e}\n")

    # 두 종목 이상 조회했으면 마지막에 매수/매도 신호 종목을 한눈에 모아 보여준다
    results = [r for r in results if r]
    if len(results) > 1:
        print("\n".join(format_summary(results)))


if __name__ == "__main__":
    main()
