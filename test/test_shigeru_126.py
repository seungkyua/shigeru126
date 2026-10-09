# -*- coding: utf-8 -*-
"""shigeru_126 매수/매도 알림 테스트 (네트워크 없이 가짜 지표 데이터 사용)"""
from types import SimpleNamespace

import pandas as pd

from common import shigeru_126 as s


def make_frame(rows):
    """지표 컬럼이 이미 계산된 DataFrame을 만든다. rows는 기본값을 덮어쓸 dict 목록."""
    base = dict(Close=100.0, RSI=45.0, MACD=0.0, SIGNAL=0.0,
                CLOUD_TOP=95.0, CLOUD_BOT=90.0,
                RSI_DOWN30=False, MACD_GC=False, MACD_DC=False, CLOUD_UP=False)
    idx = pd.date_range("2026-10-01", periods=len(rows), freq="D")
    return pd.DataFrame([{**base, **r} for r in rows], index=idx)


def args(**kw):
    return SimpleNamespace(**{"since": None, "held": 0, "exit": "bottom", "n": 10, "amount": None, **kw})


# ---------------------------------------------------------------- 오늘 신호 출력
def test_shouldShowBuyBannerWhenBuySignalToday():
    buys = [("매수", "RSI 30 하회 (28.0)", 1, 100.0)]
    lines = s.format_today(buys, [])
    assert any("🟢🟢 오늘 매수 신호 🟢🟢" in l for l in lines)


def test_shouldShowBuyDetailLineWhenBuySignalToday():
    buys = [("매수", "MACD 골든크로스", 2, 1234.5)]
    lines = s.format_today(buys, [])
    assert any("MACD 골든크로스: 2/9 매수" in l and "1,234.50" in l for l in lines)


def test_shouldShowSellBannerWhenSellSignalToday():
    sells = [("매도", "MACD 데드크로스", 1, 100.0)]
    lines = s.format_today([], sells)
    assert any("🚨🚨 오늘 매도 신호 🚨🚨" in l for l in lines)


def test_shouldShowNoSignalWhenNothingToday():
    assert any("오늘 신호 없음" in l for l in s.format_today([], []))


# ---------------------------------------------------------------- report 반환값
def test_shouldReturnBuySignalWhenRsiDropsBelow30OnLastDay(capsys):
    d = make_frame([{}, {"RSI": 28.0, "RSI_DOWN30": True}])
    result = s.report("TEST", d, args())
    assert [b[1] for b in result["buys"]] == ["RSI 30 하회 (28.0)"]


def test_shouldReturnSellSignalWhenCloudExitOnLastDay(capsys):
    d = make_frame([{}, {"Close": 80.0}])
    result = s.report("TEST", d, args(since="2026-10-01", held=9))
    assert [x[1] for x in result["sells"]] == ["일목 구름 이탈 → 전량"]


def test_shouldReturnNoneWhenDataEmpty(capsys):
    d = make_frame([{}])
    assert s.report("TEST", d, args(since="2030-01-01")) is None


# ---------------------------------------------------------------- 종목 요약
def test_shouldListBuyTickerInSummary():
    results = [
        {"name": "AAA", "units": 1, "buys": [("매수", "RSI 30 하회 (28.0)", 1, 10.0)], "sells": []},
        {"name": "BBB", "units": 0, "buys": [], "sells": []},
    ]
    text = "\n".join(s.format_summary(results))
    assert "🟢 매수 신호" in text and "AAA" in text.split("🟢 매수 신호")[1].split("\n")[1]


def test_shouldListSellTickerInSummary():
    results = [{"name": "CCC", "units": 6, "buys": [],
                "sells": [("매도", "MACD 데드크로스", 1, 10.0)]}]
    text = "\n".join(s.format_summary(results))
    assert "🚨 매도 신호" in text and "CCC" in text


def test_shouldShowNoSignalTickersInSummary():
    results = [{"name": "DDD", "units": 0, "buys": [], "sells": []}]
    text = "\n".join(s.format_summary(results))
    assert "신호 없음: DDD" in text


# ---------------------------------------------------------------- 투자금(--amount)
def test_shouldShowBuyAmountAndSharesWhenAmountGiven():
    buys = [("매수", "MACD 골든크로스", 2, 300.0)]
    text = "\n".join(s.format_today(buys, [], amount=9_000_000))
    assert "2,000,000 매수 (약 6,666주)" in text


def test_shouldNotShowAmountWhenAmountNotGiven():
    buys = [("매수", "MACD 골든크로스", 2, 300.0)]
    text = "\n".join(s.format_today(buys, []))
    assert "2,000,000" not in text


def test_shouldShowSellFractionOfHoldingsSequentiallyWhenAmountGiven():
    sells = [("매도", "MACD 데드크로스", 1, 100.0), ("매도", "RSI 50 하회 (48.0)", 2, 100.0)]
    text = "\n".join(s.format_today([], sells, amount=900, held_before=9))
    assert "보유 수량의 1/9" in text and "보유 수량의 2/8" in text


def test_shouldShowAllHoldingsWhenCloudExitSellsEverything():
    sells = [("매도", "일목 구름 이탈 → 전량", 6, 100.0)]
    text = "\n".join(s.format_today([], sells, amount=900, held_before=6))
    assert "보유 수량 전량" in text


def test_shouldShowNextBuyAmountInHintWhenAmountGiven():
    pos = s.Position(1)
    last = make_frame([{"Close": 300.0, "MACD": -1.0, "SIGNAL": 0.0}]).iloc[-1]
    assert "다음 매수 금액 2,000,000 (현재가 기준 약 6,666주)" in s.next_hint(pos, last, "bottom", 9_000_000)


def test_shouldNotShowNextBuyAmountWhenFullPosition():
    pos = s.Position(9)
    last = make_frame([{}]).iloc[-1]
    assert "다음 매수 금액" not in s.next_hint(pos, last, "bottom", 9_000_000)


def test_shouldPrintBuyAmountInReportWhenAmountGiven(capsys):
    d = make_frame([{}, {"RSI": 28.0, "RSI_DOWN30": True, "Close": 100.0}])
    s.report("TEST", d, args(amount=900_000))
    assert "100,000 매수 (약 1,000주)" in capsys.readouterr().out


def test_shouldPrintSellFractionInReportWhenAmountGiven(capsys):
    d = make_frame([{"RSI": 55.0}, {"RSI": 55.0, "MACD": -1.0, "SIGNAL": 0.0, "MACD_DC": True}])
    s.report("TEST", d, args(since="2026-10-01", held=9, amount=900_000))
    assert "보유 수량의 1/9" in capsys.readouterr().out


def test_shouldRejectNonPositiveAmount():
    import argparse
    import pytest
    with pytest.raises(argparse.ArgumentTypeError):
        s.positive_amount("0")


def test_shouldNotShowEmptySeparatorWhenMacdAndRsiAlreadySold():
    pos = s.Position(9)
    pos.units, pos.sold_macd, pos.sold_rsi = 6, True, True
    last = make_frame([{}]).iloc[-1]
    assert "매도 감시: 구름 이탈선" in s.next_hint(pos, last, "bottom")


# ---------------------------------------------------------------- 야후 조회
def test_shouldFallBackToKosdaqQuietlyWhenKospiHasNoData(monkeypatch):
    import logging
    import sys
    import types

    def fake_download(sym, **kw):
        if sym.endswith(".KS"):
            return pd.DataFrame()
        idx = pd.date_range("2026-10-01", periods=1)
        return pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0]}, index=idx)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    _, sym = s.load_yahoo("043260")
    assert sym == "043260.KQ" and logging.getLogger("yfinance").level == logging.CRITICAL
