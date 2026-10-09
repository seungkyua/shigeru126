#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시게루 1:2:6 보유 종목 관리
==========================

내가 실제로 매매한 기록(buynsell/portfolio.csv)을 읽어서 종목별로
  - 지금 몇 단계(1/9, 3/9, 9/9)를 몇 주 보유 중인지
  - 마지막 매매 이후 1:2:6 신호가 났는지 (놓친 신호 포함)
  - 그래서 지금 얼마를 몇 주 사야 하는지 / 몇 주 팔아야 하는지
를 알려줍니다.

CSV 형식 (buynsell/portfolio_sample.csv 참고)
  종목코드,종목명,날짜,구분,단계,가격,수량,투자금
  009540,HD한국조선해양,2026-10-08,매수,1,291500,3,9000000

  구분   : 매수 / 매도
  단계   : 매수 1(RSI 30 하회) 2(MACD 골든크로스) 6(구름 상단 돌파)
           매도 1(MACD 데드크로스) 2(RSI 50 하회) 6(구름 이탈, 남은 물량 전량)
  가격   : 실제 체결 단가,  수량: 실제 체결 주수
  투자금 : 종목당 총 투자금(9단위 전체). 그 종목의 첫 줄에만 적어도 됩니다.

사용법
  python buynsell/shigeru_portfolio.py                          # buynsell/portfolio.csv 읽기
  python buynsell/shigeru_portfolio.py portfolio_sample.csv     # 다른 파일 (이름만 쓰면 buynsell/ 에서도 찾음)
  python buynsell/shigeru_portfolio.py --amount 9000000         # 투자금 칸이 빈 종목의 기본 투자금
  (프로젝트 루트에서 실행 기준. python -m buynsell.shigeru_portfolio ... 도 가능)
  할 일 목록은 buynsell/shigeru_todo_YYYYMMDD.csv 로 저장됩니다.

※ 투자 판단의 참고용입니다. 실제 매매 책임은 본인에게 있습니다.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# python buynsell/shigeru_portfolio.py 처럼 직접 실행해도 common 패키지를 찾도록 프로젝트 루트를 import 경로에 추가
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.shigeru_126 import (Position, TOTAL_UNITS, add_indicators, load_yahoo, next_hint,
                         positive_amount, step)

OUT_DIR = Path(__file__).resolve().parent  # 매매 기록·할 일 목록 모두 buynsell/ 에 둔다
DEFAULT_CSV = str(OUT_DIR / "portfolio.csv")
KINDS = ("매수", "매도")
STEPS = (1, 2, 6)


# ---------------------------------------------------------------- 매매 기록
def resolve_csv(path: str) -> str:
    """주어진 경로에 파일이 있으면 그대로, 없으면 buynsell/ 아래에서 같은 이름을 찾는다.
    (예: 루트에서 portfolio_sample.csv 라고만 적어도 buynsell/portfolio_sample.csv 를 읽음)"""
    if Path(path).exists():
        return path
    local = OUT_DIR / path
    return str(local) if local.exists() else path


def load_trades(path: str) -> pd.DataFrame:
    """매매 기록 CSV를 읽고 검증한 뒤 날짜순으로 정렬한다."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"종목코드": str}, comment="#",
                     skip_blank_lines=True)
    df.columns = df.columns.str.strip()
    df["종목코드"] = df["종목코드"].str.strip().str.upper()
    # 숫자로만 된 한국 종목코드는 6자리로 (엑셀에서 앞자리 0이 빠지는 경우 대비)
    digits = df["종목코드"].str.isdigit()
    df.loc[digits, "종목코드"] = df.loc[digits, "종목코드"].str.zfill(6)
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["구분"] = df["구분"].str.strip()
    if "투자금" not in df.columns:
        df["투자금"] = float("nan")
    if "종목명" not in df.columns:
        df["종목명"] = ""

    for i, r in df.iterrows():
        line = i + 2  # 헤더가 1행
        if r["구분"] not in KINDS:
            raise ValueError(f"{line}행: 구분은 매수/매도 중 하나여야 합니다 (입력: {r['구분']})")
        if r["단계"] not in STEPS:
            raise ValueError(f"{line}행: 단계는 1, 2, 6 중 하나여야 합니다 (입력: {r['단계']})")
    return df.sort_values("날짜", kind="stable").reset_index(drop=True)


def build_state(trades: pd.DataFrame, fallback_amount: float | None = None) -> dict:
    """한 종목의 매매 기록을 순서대로 적용해 현재 보유 상태를 만든다.
    전량 매도로 보유 주수가 0이 되면 사이클을 초기화한다 (다음 매수는 다시 1단계부터)."""
    s = _empty_cycle()
    s["amount"] = fallback_amount
    for r in trades.itertuples(index=False):
        q, qty = int(r.단계), int(r.수량)
        if pd.notna(r.투자금):
            s["amount"] = float(r.투자금)
        if r.구분 == "매수":
            s["bought"].add(q)
            s["units"] += q
            s["shares"] += qty
            s["cost"] += float(r.가격) * qty
            s["bought_shares"] += qty
        else:
            s["units"] = max(0, s["units"] - q)
            s["shares"] -= qty
            s["sold_macd"] |= q == 1
            s["sold_rsi"] |= q == 2
            if s["shares"] <= 0:
                s.update({k: v for k, v in _empty_cycle().items()})
        s["last_date"] = r.날짜
        s["name"] = r.종목명 if isinstance(r.종목명, str) else ""

    b = s["bought"]
    s["stage"] = 3 if 6 in b else 2 if 2 in b else 1 if 1 in b else 0
    s["avg_cost"] = s["cost"] / s["bought_shares"] if s["bought_shares"] else 0
    return s


def _empty_cycle() -> dict:
    return dict(bought=set(), units=0, shares=0, cost=0.0, bought_shares=0,
                sold_macd=False, sold_rsi=False)


def to_position(state: dict) -> Position:
    """CSV에서 만든 상태를 shigeru_126의 Position(1:2:6 상태머신)으로 옮긴다."""
    pos = Position(0)
    pos.stage, pos.units = state["stage"], state["units"]
    pos.sold_macd, pos.sold_rsi = state["sold_macd"], state["sold_rsi"]
    return pos


# ---------------------------------------------------------------- 신호
def pending_actions(d: pd.DataFrame, state: dict, exit_mode: str) -> list[tuple]:
    """마지막 매매일 다음 거래일부터 오늘까지 1:2:6 규칙을 적용해,
    아직 기록에 없는(=아직 실행하지 않은) 매수/매도를 (날짜, 구분, 사유, 단계, 신호일종가)로 돌려준다."""
    pos = to_position(state)
    acts = []
    for dt, row in d[d.index > state["last_date"]].iterrows():
        for kind, why, q in step(pos, row, exit_mode):
            acts.append((dt, kind, why, q, float(row["Close"])))
    return acts


def size_actions(actions: list[tuple], state: dict, amount: float | None,
                 price: float) -> list[dict]:
    """각 행동을 금액·주수로 바꾼다. 매수는 투자금×단계/9 를 현재가로 나눈 주수,
    매도는 그 시점 보유 주수 × (매도 단위 / 보유 단위), 구름 이탈은 전량."""
    units, shares = state["units"], state["shares"]
    out = []
    for dt, kind, why, q, _ in actions:
        if kind == "매수":
            money = amount * q / TOTAL_UNITS if amount else None
            n = int(money // price) if money else None
            units, shares = units + q, shares + (n or 0)
        else:
            n = shares if q >= units else min(shares, max(1, round(shares * q / units)))
            money = n * price
            units, shares = max(0, units - q), shares - n
        out.append({"날짜": dt, "구분": kind, "사유": why, "단계": q, "금액": money, "주수": n})
    return out


# ---------------------------------------------------------------- 출력
def format_actions(sized: list[dict], price: float) -> list[str]:
    lines = []
    for a in sized:
        head = f"    {'🟢 매수' if a['구분'] == '매수' else '🚨 매도'} {a['단계']}/9  " \
               f"[{a['날짜']:%Y-%m-%d} {a['사유']}]"
        if a["구분"] == "매수":
            if a["주수"] is None:
                tail = "  → 투자금을 몰라 금액 계산 불가 (CSV 투자금 칸 또는 --amount)"
            else:
                tail = f"  → {a['금액']:,.0f} 매수 = 약 {a['주수']:,}주 (현재가 {price:,.0f} 기준)"
        else:
            tail = f"  → {a['주수']:,}주 매도 (현재가 기준 약 {a['금액']:,.0f})"
        lines.append(head + tail)
    return lines


def describe_holding(state: dict, price: float) -> str:
    if state["shares"] <= 0:
        return "  보유 없음 (사이클 종료) — 다시 1단계 신호를 기다립니다"
    done = "·".join(str(q) for q in STEPS if q in state["bought"])
    pnl = (price / state["avg_cost"] - 1) * 100 if state["avg_cost"] else 0
    line = (f"  보유 {state['units']}/9 단위 ({done} 매수), {state['shares']:,}주, "
            f"평균단가 {state['avg_cost']:,.0f}, 평가손익 {pnl:+.1f}%")
    if state["amount"]:
        line += f" | 투자금 {state['amount']:,.0f}"
    return line


def report(code: str, trades: pd.DataFrame, args) -> list[dict]:
    """한 종목의 상태와 할 일을 출력하고, 요약용으로 할 일 목록을 돌려준다."""
    state = build_state(trades, args.amount)
    raw, sym = load_yahoo(code)
    d = add_indicators(raw)
    last, price = d.iloc[-1], float(d["Close"].iloc[-1])

    print("=" * 78)
    print(f"■ {state['name'] or code}({sym})   기준일 {d.index[-1]:%Y-%m-%d}   종가 {price:,.2f}")
    print(describe_holding(state, price))
    print(f"  마지막 기록 {state['last_date']:%Y-%m-%d}")

    sized = size_actions(pending_actions(d, state, args.exit), state, state["amount"], price)
    if sized:
        print("\n  ▶ 지금 해야 할 일 (마지막 기록 이후 발생한 신호)")
        print("\n".join(format_actions(sized, price)))
        if any(a["날짜"] < d.index[-1] for a in sized):
            print("    ※ 지난 날짜의 신호는 놓친 신호입니다. 지금 가격으로 따라갈지 판단하세요.")
    else:
        print("\n  ✅ 지금은 할 일 없음")
        print("  " + next_hint(to_position(state), last, args.exit, state["amount"]))
    print()
    return [{"종목코드": code, "종목명": state["name"], "기준일": d.index[-1], "현재가": price, **a}
            for a in sized]


def default_todo_path(todo: list[dict]) -> str:
    """기본 저장 경로: buynsell/ 아래에 가장 최근 기준일 날짜를 붙인다."""
    return str(OUT_DIR / f"shigeru_todo_{max(a['기준일'] for a in todo):%Y%m%d}.csv")


def main():
    ap = argparse.ArgumentParser(description="시게루 1:2:6 보유 종목 관리")
    ap.add_argument("csv", nargs="?", default=DEFAULT_CSV, help="매매 기록 CSV (기본 buynsell/portfolio.csv)")
    ap.add_argument("--amount", type=positive_amount, help="CSV에 투자금이 없는 종목의 종목당 총 투자금")
    ap.add_argument("--exit", default="bottom", choices=["bottom", "top"],
                    help="구름 이탈 기준: bottom=구름 하단 하회(기본), top=구름 상단 하회")
    ap.add_argument("--out", help="할 일 목록 CSV 경로 (기본 buynsell/shigeru_todo_YYYYMMDD.csv)")
    args = ap.parse_args()

    try:
        args.csv = resolve_csv(args.csv)
        trades = load_trades(args.csv)
    except FileNotFoundError:
        sys.exit(f"{args.csv} 파일이 없습니다. buynsell/portfolio_sample.csv 를 복사해서 작성하세요.")
    except ValueError as e:
        sys.exit(f"{args.csv} 오류: {e}")

    todo = []
    for code, group in trades.groupby("종목코드", sort=False):
        try:
            todo += report(code, group, args)
        except Exception as e:
            print(f"[{code}] 오류: {e}\n")

    print("#" * 78)
    if not todo:
        print("  오늘 할 일이 없습니다.")
    else:
        print(f"  오늘 할 일 {len(todo)}건")
        for a in todo:
            qty = f"{a['주수']:,}주" if a["주수"] is not None else "주수 미정"
            print(f"    - {a['종목명'] or a['종목코드']}: {a['구분']} {a['단계']}/9 {qty}  ({a['사유']})")
        out = args.out or default_todo_path(todo)
        pd.DataFrame(todo).to_csv(out, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        print(f"\n  💾 할 일 저장: {out}")
    print("#" * 78)


if __name__ == "__main__":
    main()
