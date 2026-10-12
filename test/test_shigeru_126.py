# -*- coding: utf-8 -*-
"""shigeru_126 매수/매도 알림 테스트 (네트워크 없이 가짜 지표 데이터 사용)"""
from types import SimpleNamespace

import pandas as pd
import pytest

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


def test_shouldReturnSellSignalWhenCloudExitOnLastDayAfterMacdAndRsiSells(capsys):
    d = make_frame([{"RSI": 55.0},
                    {"RSI": 55.0, "MACD": -1.0, "MACD_DC": True},
                    {"RSI": 45.0},
                    {"RSI": 45.0, "Close": 80.0}])
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
    assert "다음 매도 조건: 구름 이탈선" in s.next_hint(pos, last, "bottom")


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


# ---------------------------------------------------------------- 장중 현재가
def naver_item(code, status="OPEN", close="91,500", open_="88,600", high="92,400", low="86,500",
               at="2026-10-12T11:12:25.177+09:00"):
    return {"itemCode": code, "marketStatus": status, "closePrice": close, "openPrice": open_,
            "highPrice": high, "lowPrice": low, "localTradedAt": at}


def daily(dates, close=100.0):
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close}, index=idx)


def test_shouldParseCommaPricesWhenMarketOpen():
    q = s.parse_live_quotes({"datas": [naver_item("087010")]})["087010"]
    assert (q["Open"], q["High"], q["Low"], q["Close"]) == (88_600, 92_400, 86_500, 91_500)
    assert q["time"] == pd.Timestamp("2026-10-12 11:12:25.177")


def test_shouldSkipQuoteWhenMarketClosed():
    assert s.parse_live_quotes({"datas": [naver_item("005930", status="CLOSE")]}) == {}


def test_shouldUseCurrentPriceForMissingOpenWhenTradingHalted():
    q = s.parse_live_quotes({"datas": [naver_item("005930", open_="0", high="0", low="0")]})["005930"]
    assert (q["Open"], q["High"], q["Low"]) == (91_500, 91_500, 91_500)


def test_shouldSkipOnlyItemWithoutPriceAndKeepOthers():
    payload = {"datas": [naver_item("000001", close="-", open_="-", high="-", low="-"),
                         naver_item("005930", open_="-")]}
    quotes = s.parse_live_quotes(payload)
    assert list(quotes) == ["005930"] and quotes["005930"]["Open"] == 91_500


def test_shouldReplaceTodayBarWithLiveQuote():
    q = s.parse_live_quotes({"datas": [naver_item("087010")]})["087010"]
    d = s.apply_live_quote(daily(["2026-10-08", "2026-10-12"]), q)
    assert list(d.index) == [pd.Timestamp("2026-10-08"), pd.Timestamp("2026-10-12")]
    assert d["Close"].iloc[-1] == 91_500 and d["High"].iloc[-1] == 92_400


def test_shouldAppendTodayBarWhenDailyDataLacksToday():
    q = s.parse_live_quotes({"datas": [naver_item("087010")]})["087010"]
    d = s.apply_live_quote(daily(["2026-10-07", "2026-10-08"]), q)
    assert d.index[-1] == pd.Timestamp("2026-10-12") and len(d) == 3 and d["Close"].iloc[-1] == 91_500


def test_shouldReturnNoQuotesWhenNetworkFails(monkeypatch):
    import requests

    def boom(*a, **kw):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "get", boom)
    assert s.fetch_live_quotes(["005930"]) == {}


def test_shouldRequestOnlyKoreanCodes(monkeypatch):
    import requests
    seen = []

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"datas": [naver_item("005930")]}

    monkeypatch.setattr(requests, "get", lambda url, **kw: seen.append(url) or Resp())
    quotes = s.fetch_live_quotes(["005930", "AAPL"])
    assert list(quotes) == ["005930"] and seen[0].endswith("/005930")


def test_shouldNotCallNetworkWhenNoKoreanCodes(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **kw: pytest.fail("호출되면 안 됨"))
    assert s.fetch_live_quotes(["AAPL"]) == {}


def test_shouldLabelPriceAsCloseOrLive():
    q = s.parse_live_quotes({"datas": [naver_item("087010")]})["087010"]
    assert s.price_label(None) == "종가"
    assert s.price_label(q["time"]) == "현재가(장중 11:12)"


# ---------------------------------------------------------------- 매도 순서 (MACD → RSI → 구름)
def full_position():
    return s.Position(9)


def row(**kw):
    return make_frame([kw]).iloc[-1]


def test_shouldNotSellOnCloudExitBeforeMacdDeadCross():
    pos = full_position()
    assert s.step(pos, row(Close=80.0, RSI=45.0), "bottom") == []
    assert pos.units == 9


def test_shouldNotSellOnRsiBelow50BeforeMacdDeadCross():
    pos = full_position()
    assert s.step(pos, row(RSI=45.0), "bottom") == []


def test_shouldNotSellOnCloudExitBeforeRsiSell():
    pos = full_position()
    pos.units, pos.sold_macd = 8, True
    assert s.step(pos, row(Close=80.0, RSI=55.0), "bottom") == []
    assert pos.units == 8


def test_shouldSellAllOnCloudExitAfterMacdAndRsiSells():
    pos = full_position()
    pos.units, pos.sold_macd, pos.sold_rsi = 6, True, True
    assert s.step(pos, row(Close=80.0), "bottom") == [("매도", "일목 구름 이탈 → 전량", 6)]
    assert pos.stage == 0 and pos.units == 0


def test_shouldSellInOrderWhenAllConditionsHitSameDay():
    pos = full_position()
    acts = s.step(pos, row(Close=80.0, RSI=45.0, MACD=-1.0, MACD_DC=True), "bottom")
    assert [(a[1].split(" (")[0], a[2]) for a in acts] == [
        ("MACD 데드크로스", 1), ("RSI 50 하회", 2), ("일목 구름 이탈 → 전량", 6)]


def test_shouldHintMacdDeadCrossFirstInSellStage():
    hint = s.next_hint(full_position(), row(), "bottom")
    assert "다음 매도 조건: MACD 데드크로스" in hint and "RSI" not in hint and "구름" not in hint


def test_shouldHintRsiAfterMacdSold():
    pos = full_position()
    pos.units, pos.sold_macd = 8, True
    hint = s.next_hint(pos, row(RSI=55.0), "bottom")
    assert "다음 매도 조건: RSI 50 하회 (현재 55.0)" in hint and "구름" not in hint
