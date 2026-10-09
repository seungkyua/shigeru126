#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시게루 1:2:6 매수 시작 종목 찾기
================================

stock/stock_list_krx.csv 의 KOSPI/KOSDAQ 종목(일반주·우선주, ETF/ETN 제외)을 훑어서,
1:2:6 매매법의 1단계 매수 조건(RSI(14)가 30 아래로 내려감)이 발생한 종목을 알려줍니다.
--amount 로 종목당 투자금을 넣으면 첫 1/9 매수 금액과 살 수 있는 주수를 함께 보여줍니다.
기본으로 주가 5,000원 미만 종목은 제외하며(--min-price 로 변경),
찾은 종목은 화면 출력과 함께 firstbuy/shigeru_start_YYYYMMDD.csv 로 저장합니다(--out 으로 변경).

사용법
  python firstbuy/shigeru_start.py                                  # 전 종목, 마지막 거래일 기준
  python firstbuy/shigeru_start.py --amount 9000000                 # 1단계 매수 금액(100만 원)과 주수
  python firstbuy/shigeru_start.py --days 3                         # 최근 3거래일 안에 신호가 난 종목
  python firstbuy/shigeru_start.py --market KOSDAQ --limit 300      # 코스닥 앞쪽 300종목만
  python firstbuy/shigeru_start.py --min-price 10000                # 1만 원 미만 제외 (0 = 제한 없음)
  python firstbuy/shigeru_start.py --include-etf                    # ETF/ETN도 포함
  python firstbuy/shigeru_start.py --out start.csv             # 저장 파일 이름 지정
  (프로젝트 루트에서 실행 기준. python -m firstbuy.shigeru_start ... 도 가능)

※ 투자 판단의 참고용입니다. 실제 매매 책임은 본인에게 있습니다.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# python firstbuy/shigeru_start.py 처럼 직접 실행해도 common 패키지를 찾도록 프로젝트 루트를 import 경로에 추가
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.shigeru_126 import add_indicators, buy_money, positive_amount, UNIT_BUY

SUFFIX = {"KOSPI": ".KS", "KOSDAQ": ".KQ"}
PRICE_COLS = ["Open", "High", "Low", "Close"]
DEFAULT_LIST = str(ROOT / "stock" / "stock_list_krx.csv")
OUT_DIR = Path(__file__).resolve().parent  # 결과 파일은 firstbuy/ 에 저장
STOCK_TYPE = "stock"  # 네이버 stockEndType: stock(일반주·우선주) / etf / etn


# ---------------------------------------------------------------- 종목 리스트
def yahoo_symbol(code: str, market: str) -> str:
    """종목코드 + 시장 → 야후 파이낸스 심볼 (예: 005930, KOSPI → 005930.KS)"""
    return code + SUFFIX[market]


def load_stock_list(path: str) -> pd.DataFrame:
    """종목코드,종목명,시장 컬럼의 CSV를 읽는다. 종목코드는 6자리 문자열로 맞춘다."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"종목코드": str})
    df["종목코드"] = df["종목코드"].str.zfill(6)
    return df[df["시장"].isin(SUFFIX)].reset_index(drop=True)


def filter_stocks(stocks: pd.DataFrame, market: str | None = None,
                  limit: int | None = None, stock_only: bool = True) -> pd.DataFrame:
    """종류·시장 필터와 앞에서부터 N개 제한 (CSV가 시가총액 순이라 앞쪽이 대형주).
    stock_only면 일반주·우선주만 남기고 ETF/ETN은 뺀다. 종류 컬럼이 없는 옛 CSV는 거르지 않는다."""
    if stock_only and "종류" in stocks.columns:
        stocks = stocks[stocks["종류"] == STOCK_TYPE]
    if market:
        stocks = stocks[stocks["시장"] == market]
    if limit:
        stocks = stocks.head(limit)
    return stocks.reset_index(drop=True)


# ---------------------------------------------------------------- 데이터
def split_download(raw: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """yf.download(group_by='ticker') 결과를 종목별 DataFrame으로 나눈다.
    가격이 전혀 없는 종목(상장폐지·거래정지 등)은 뺀다."""
    frames = {}
    for sym in symbols:
        if isinstance(raw.columns, pd.MultiIndex):
            if sym not in raw.columns.get_level_values(0):
                continue
            df = raw[sym]
        else:  # 한 종목만 받았을 때 단일 컬럼으로 오는 경우
            df = raw
        df = df[PRICE_COLS].dropna()
        if not df.empty:
            frames[sym] = df
    return frames


def download_all(symbols: list[str], chunk: int = 100, period: str = "1y") -> dict[str, pd.DataFrame]:
    """야후에서 일봉을 chunk개씩 묶어 받는다 (종목마다 따로 요청하는 것보다 훨씬 빠름)."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance가 필요합니다:  pip install yfinance")
    # 상장폐지·신주인수권 등 야후에 없는 종목마다 찍히는 오류 로그를 숨긴다 (개수는 main에서 요약)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    frames = {}
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        print(f"  데이터 받는 중... {min(i + chunk, len(symbols))}/{len(symbols)}",
              file=sys.stderr, flush=True)
        raw = yf.download(part, period=period, interval="1d", group_by="ticker",
                          auto_adjust=False, threads=True, progress=False)
        if raw is not None and not raw.empty:
            frames.update(split_download(raw, part))
    return frames


# ---------------------------------------------------------------- 신호
def start_signal_date(d: pd.DataFrame, days: int = 1):
    """최근 days 거래일 안에 RSI가 30 아래로 내려간 날(가장 최근)을 돌려준다. 없으면 None."""
    hits = d["RSI_DOWN30"].tail(days)
    hits = hits[hits.astype(bool)]
    return hits.index[-1] if len(hits) else None


def first_buy(amount: float, price: float) -> tuple[float, int]:
    """1단계(1/9) 매수 금액과 그 금액으로 price에 살 수 있는 주수."""
    return buy_money(amount, UNIT_BUY[1], price)


def find_candidates(stocks: pd.DataFrame, frames: dict[str, pd.DataFrame],
                    days: int = 1, min_price: float = 0) -> list[dict]:
    """지표가 계산된 frames에서 매수 시작 신호가 난 종목을 모아 RSI 낮은 순으로 정렬.
    마지막 종가가 min_price 미만인 저가주는 뺀다."""
    rows = []
    for s in stocks.itertuples(index=False):
        sym = yahoo_symbol(s.종목코드, s.시장)
        d = frames.get(sym)
        if d is None or d.empty:
            continue
        date = start_signal_date(d, days)
        if date is None:
            continue
        last = d.iloc[-1]
        if last["Close"] < min_price:
            continue
        rows.append({"종목코드": s.종목코드, "종목명": s.종목명, "시장": s.시장,
                     "신호일": date, "기준일": d.index[-1],
                     "종가": float(last["Close"]), "RSI": round(float(last["RSI"]), 1)})
    return sorted(rows, key=lambda r: r["RSI"])


def compute_indicators(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """종목별로 RSI 등 지표 계산. 데이터가 너무 짧아 계산이 안 되는 종목은 뺀다."""
    out = {}
    for sym, df in frames.items():
        try:
            d = add_indicators(df)
        except Exception:
            continue
        if not d.empty:
            out[sym] = d
    return out


# ---------------------------------------------------------------- 출력
def format_candidates(rows: list[dict], amount: float | None, min_price: float = 0) -> list[str]:
    if not rows:
        return ["매수 시작 종목이 없습니다. (--days 로 기간을 늘려 보세요)"]

    lines = ["=" * 78, f"  🟢 1:2:6 매수 시작 종목 {len(rows)}개  (RSI 30 하회 → 1/9 매수)"]
    if min_price:
        lines.append(f"  주가 {min_price:,.0f}원 미만 제외")
    if amount:
        money, _ = first_buy(amount, 1)
        lines.append(f"  종목당 투자금 {amount:,.0f} → 1단계 매수 금액 {money:,.0f}")
    lines.append("=" * 78)
    for i, r in enumerate(rows, 1):
        line = (f"{i:>3}. {r['종목명']}({r['종목코드']}, {r['시장']})  "
                f"신호일 {r['신호일']:%Y-%m-%d}  종가 {r['종가']:,.0f}  RSI {r['RSI']:.1f}")
        if amount:
            money, shares = first_buy(amount, r["종가"])
            line += (f"\n       → {money:,.0f} 매수 (약 {shares:,}주)" if shares
                     else f"\n       → 금액 부족: 1주 {r['종가']:,.0f} > 1단계 금액 {money:,.0f}")
        lines.append(line)
    lines.append("")
    lines.append("  다음 단계: MACD 골든크로스 시 2/9, 구름 상단 돌파 시 6/9 추가 매수")
    lines.append("  매수 후 관리:  buynsell/portfolio.csv 에 체결 내역을 적고 python buynsell/shigeru_portfolio.py")
    return lines


def default_out_path(rows: list[dict]) -> str:
    """기본 저장 경로: firstbuy/ 아래에 가장 최근 기준일 날짜를 붙인다 (예: firstbuy/shigeru_start_20261008.csv)."""
    latest = max(r["기준일"] for r in rows)
    return str(OUT_DIR / f"shigeru_start_{latest:%Y%m%d}.csv")


def save_candidates(rows: list[dict], path: str, amount: float | None) -> None:
    df = pd.DataFrame(rows)
    if amount and not df.empty:
        buys = [first_buy(amount, p) for p in df["종가"]]
        df["1단계금액"] = [round(m) for m, _ in buys]
        df["주수"] = [n for _, n in buys]
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- main
def non_negative_price(text: str) -> float:
    """--min-price 값 검증: 0 이상 숫자만 허용 (쉼표 허용, 예: 5,000)."""
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        raise argparse.ArgumentTypeError(f"숫자가 아닙니다: {text}")
    if value < 0:
        raise argparse.ArgumentTypeError("최소 주가는 0 이상이어야 합니다.")
    return value


def main():
    ap = argparse.ArgumentParser(description="시게루 1:2:6 매수 시작 종목 찾기")
    ap.add_argument("--list", default=DEFAULT_LIST, help="종목 리스트 CSV (기본 stock/stock_list_krx.csv)")
    ap.add_argument("--amount", type=positive_amount, help="종목당 총 투자금(원). 1단계는 이 금액의 1/9")
    ap.add_argument("--days", type=int, default=1, help="최근 며칠(거래일) 안의 신호까지 볼지 (기본 1 = 마지막 거래일)")
    ap.add_argument("--market", choices=list(SUFFIX), help="KOSPI 또는 KOSDAQ만")
    ap.add_argument("--limit", type=int, help="리스트 앞에서부터 N개 종목만 검사")
    ap.add_argument("--min-price", type=non_negative_price, default=5000,
                    help="이 가격(원) 미만 종목 제외 (기본 5000, 0 = 제한 없음)")
    ap.add_argument("--include-etf", action="store_true", help="ETF/ETN도 포함 (기본은 일반주·우선주만)")
    ap.add_argument("--out", help="결과 CSV 경로 (기본 firstbuy/shigeru_start_YYYYMMDD.csv)")
    args = ap.parse_args()
    if args.days < 1:
        ap.error("--days 는 1 이상이어야 합니다.")

    stocks = load_stock_list(args.list)
    if not args.include_etf and "종류" not in stocks.columns:
        print("  ⚠️ 종목 리스트에 '종류' 컬럼이 없어 ETF/ETN을 거를 수 없습니다. "
              "stock/stock_list_krx.csv를 지우고 python stock/stock_list.py 로 다시 받으세요.", file=sys.stderr)
    stocks = filter_stocks(stocks, args.market, args.limit, stock_only=not args.include_etf)
    print(f"  {len(stocks)}개 종목 검사 시작", file=sys.stderr)
    symbols = [yahoo_symbol(c, m) for c, m in zip(stocks["종목코드"], stocks["시장"])]
    frames = compute_indicators(download_all(symbols))
    missing = len(symbols) - len(frames)
    print(f"  데이터 확보 {len(frames)}/{len(symbols)}개 종목"
          + (f" (야후에 없거나 기간이 짧은 {missing}개 제외)" if missing else "") + "\n",
          file=sys.stderr)

    rows = find_candidates(stocks, frames, args.days, args.min_price)
    print("\n".join(format_candidates(rows, args.amount, args.min_price)))
    if rows:  # 찾은 종목이 있으면 항상 파일로도 남긴다
        out = args.out or default_out_path(rows)
        save_candidates(rows, out, args.amount)
        print(f"\n  💾 결과 저장: {out}")


if __name__ == "__main__":
    main()
