# -*- coding: utf-8 -*-
"""shigeru_portfolio (매매 기록 → 다음 매수/매도 안내) 테스트 — 네트워크 없이 실행"""
import pandas as pd
import pytest

from pathlib import Path

from buynsell import shigeru_portfolio as pf

HEADER = "종목코드,종목명,날짜,구분,단계,가격,수량,투자금\n"


def trades_from(text, tmp_path):
    p = tmp_path / "pf.csv"
    p.write_text(HEADER + text, encoding="utf-8-sig")
    return pf.load_trades(str(p))


def frame(rows, start="2026-10-01"):
    """지표가 계산된 것처럼 보이는 DataFrame (shigeru_126.step 이 쓰는 컬럼만)."""
    base = dict(Close=100.0, RSI=55.0, MACD=1.0, SIGNAL=0.0, CLOUD_TOP=95.0, CLOUD_BOT=90.0,
                RSI_DOWN30=False, MACD_GC=False, MACD_DC=False, CLOUD_UP=False)
    idx = pd.date_range(start, periods=len(rows), freq="B")
    return pd.DataFrame([{**base, **r} for r in rows], index=idx)


# ---------------------------------------------------------------- CSV 읽기
def test_shouldZeroPadKoreanCodeWhenLoading(tmp_path):
    t = trades_from("9540,HD한국조선해양,2026-10-08,매수,1,291500,3,9000000\n", tmp_path)
    assert t["종목코드"].tolist() == ["009540"]


def test_shouldKeepForeignTickerAsIs(tmp_path):
    t = trades_from("aapl,애플,2026-01-06,매수,1,262.36,4,9000\n", tmp_path)
    assert t["종목코드"].tolist() == ["AAPL"]


def test_shouldSortTradesByDate(tmp_path):
    t = trades_from("005930,삼성전자,2026-10-02,매수,2,100,2,900\n"
                    "005930,삼성전자,2026-10-01,매수,1,100,1,900\n", tmp_path)
    assert t["단계"].tolist() == [1, 2]


def test_shouldRejectUnknownTradeKind(tmp_path):
    with pytest.raises(ValueError, match="구분"):
        trades_from("005930,삼성전자,2026-10-01,보유,1,100,1,900\n", tmp_path)


def test_shouldRejectUnknownStep(tmp_path):
    with pytest.raises(ValueError, match="단계"):
        trades_from("005930,삼성전자,2026-10-01,매수,3,100,1,900\n", tmp_path)


# ---------------------------------------------------------------- 보유 상태
def test_shouldBeStage1WhenOnlyFirstBuyRecorded(tmp_path):
    t = trades_from("005930,삼성전자,2026-10-01,매수,1,100,10,9000\n", tmp_path)
    s = pf.build_state(t)
    assert (s["stage"], s["units"], s["shares"]) == (1, 1, 10)


def test_shouldBeStage2WhenFirstAndSecondBuysRecorded(tmp_path):
    t = trades_from("005930,삼성전자,2026-10-01,매수,1,100,10,9000\n"
                    "005930,삼성전자,2026-10-05,매수,2,110,18,\n", tmp_path)
    s = pf.build_state(t)
    assert (s["stage"], s["units"], s["shares"]) == (2, 3, 28)


def test_shouldComputeWeightedAverageCost(tmp_path):
    t = trades_from("005930,삼성전자,2026-10-01,매수,1,100,10,9000\n"
                    "005930,삼성전자,2026-10-05,매수,2,130,10,\n", tmp_path)
    assert pf.build_state(t)["avg_cost"] == 115


def test_shouldMarkMacdSoldWhenStep1SellRecorded(tmp_path):
    t = trades_from("005930,삼성전자,2026-10-01,매수,1,100,10,9000\n"
                    "005930,삼성전자,2026-10-05,매수,2,100,20,\n"
                    "005930,삼성전자,2026-10-09,매수,6,100,60,\n"
                    "005930,삼성전자,2026-10-20,매도,1,120,10,\n", tmp_path)
    s = pf.build_state(t)
    assert (s["stage"], s["units"], s["shares"], s["sold_macd"], s["sold_rsi"]) == (3, 8, 80, True, False)


def test_shouldResetCycleWhenAllSharesSold(tmp_path):
    t = trades_from("005930,삼성전자,2026-10-01,매수,1,100,10,9000\n"
                    "005930,삼성전자,2026-10-05,매수,2,100,20,\n"
                    "005930,삼성전자,2026-10-09,매수,6,100,60,\n"
                    "005930,삼성전자,2026-10-20,매도,6,90,90,\n", tmp_path)
    s = pf.build_state(t)
    assert (s["stage"], s["units"], s["shares"], s["last_date"]) == (0, 0, 0, pd.Timestamp("2026-10-20"))


def test_shouldUseAmountFromCsvColumn(tmp_path):
    t = trades_from("005930,삼성전자,2026-10-01,매수,1,100,10,9000\n", tmp_path)
    assert pf.build_state(t)["amount"] == 9000


def test_shouldUseFallbackAmountWhenCsvColumnEmpty(tmp_path):
    t = trades_from("005930,삼성전자,2026-10-01,매수,1,100,10,\n", tmp_path)
    assert pf.build_state(t, fallback_amount=18000)["amount"] == 18000


# ---------------------------------------------------------------- 기록 이후 신호
def test_shouldFindMissedGoldenCrossAfterLastTrade():
    state = dict(stage=1, units=1, shares=10, sold_macd=False, sold_rsi=False,
                 last_date=pd.Timestamp("2026-10-01"))
    d = frame([{}, {"MACD_GC": True}, {}])   # 10/1 기록일, 10/2 골든크로스, 10/5 오늘
    acts = pf.pending_actions(d, state, "bottom")
    assert [(a[0], a[1], a[3]) for a in acts] == [(pd.Timestamp("2026-10-02"), "매수", 2)]


def test_shouldIgnoreSignalOnOrBeforeLastTradeDate():
    state = dict(stage=1, units=1, shares=10, sold_macd=False, sold_rsi=False,
                 last_date=pd.Timestamp("2026-10-02"))
    d = frame([{}, {"MACD_GC": True}, {}])
    assert pf.pending_actions(d, state, "bottom") == []


def test_shouldFindCloudExitWhenMacdAndRsiAlreadySold():
    state = dict(stage=3, units=6, shares=60, sold_macd=True, sold_rsi=True,
                 last_date=pd.Timestamp("2026-10-01"))
    d = frame([{}, {"Close": 80.0}])
    assert [a[2] for a in pf.pending_actions(d, state, "bottom")] == ["일목 구름 이탈 → 전량"]


def test_shouldNotSellRightAfterThirdBuyWhenCloudExitComesFirst():
    # OCI홀딩스 사례: 6 매수 직후 구름 이탈이 나와도 MACD 데드크로스 전이면 매도하지 않는다
    state = dict(stage=3, units=9, shares=186, sold_macd=False, sold_rsi=False,
                 last_date=pd.Timestamp("2026-10-01"))
    d = frame([{}, {"Close": 80.0, "RSI": 45.0}])
    assert pf.pending_actions(d, state, "bottom") == []


# ---------------------------------------------------------------- 금액/주수
def test_shouldSizeBuyByAmountAndCurrentPrice():
    state = dict(units=1, shares=10)
    acts = [(pd.Timestamp("2026-10-02"), "매수", "MACD 골든크로스", 2, 95.0)]
    sized = pf.size_actions(acts, state, amount=9000, price=100.0)
    assert (sized[0]["금액"], sized[0]["주수"]) == (2000, 20)


def test_shouldSizeSellAsShareOfHoldingsSequentially():
    state = dict(units=9, shares=90)
    acts = [(pd.Timestamp("2026-10-02"), "매도", "MACD 데드크로스", 1, 100.0),
            (pd.Timestamp("2026-10-02"), "매도", "RSI 50 하회 (48.0)", 2, 100.0)]
    assert [x["주수"] for x in pf.size_actions(acts, state, amount=9000, price=100.0)] == [10, 20]


def test_shouldSellAllSharesOnCloudExit():
    state = dict(units=6, shares=61)
    acts = [(pd.Timestamp("2026-10-02"), "매도", "일목 구름 이탈 → 전량", 6, 100.0)]
    assert pf.size_actions(acts, state, amount=9000, price=100.0)[0]["주수"] == 61


def test_shouldLeaveBuySharesEmptyWhenAmountUnknown():
    state = dict(units=1, shares=10)
    acts = [(pd.Timestamp("2026-10-02"), "매수", "MACD 골든크로스", 2, 95.0)]
    assert pf.size_actions(acts, state, amount=None, price=100.0)[0]["주수"] is None


# ---------------------------------------------------------------- 출력
def test_shouldPrintBuyInstructionWithAmountAndShares():
    sized = [{"날짜": pd.Timestamp("2026-10-02"), "구분": "매수", "사유": "MACD 골든크로스",
              "단계": 2, "금액": 2000.0, "주수": 20}]
    text = "\n".join(pf.format_actions(sized, price=100.0))
    assert "🟢 매수" in text and "2,000" in text and "20주" in text and "2026-10-02" in text


def test_shouldPrintSellInstructionWithShares():
    sized = [{"날짜": pd.Timestamp("2026-10-02"), "구분": "매도", "사유": "MACD 데드크로스",
              "단계": 1, "금액": 1000.0, "주수": 10}]
    text = "\n".join(pf.format_actions(sized, price=100.0))
    assert "🚨 매도" in text and "10주" in text


def test_shouldPrintNothingToDoWhenNoActions():
    assert pf.format_actions([], price=100.0) == []


# ---------------------------------------------------------------- 기본 경로
def test_shouldDefaultPortfolioCsvToBuynsellDirectory():
    assert Path(pf.DEFAULT_CSV) == Path(pf.__file__).resolve().parent / "portfolio.csv"


def test_shouldUseGivenPathWhenItExists(tmp_path):
    p = tmp_path / "my.csv"
    p.write_text("x", encoding="utf-8")
    assert pf.resolve_csv(str(p)) == str(p)


def test_shouldFallBackToBuynsellDirectoryWhenFileNameNotFound(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = Path(pf.__file__).resolve().parent / "portfolio_sample.csv"
    assert pf.resolve_csv("portfolio_sample.csv") == str(expected)


def test_shouldKeepGivenPathWhenNotFoundAnywhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert pf.resolve_csv("nothing.csv") == "nothing.csv"


def test_shouldSaveTodoFileInBuynsellDirectory():
    todo = [{"기준일": pd.Timestamp("2026-10-08")}]
    assert pf.default_todo_path(todo) == str(Path(pf.__file__).parent / "shigeru_todo_20261008.csv")


# ---------------------------------------------------------------- 장중 현재가
def yahoo_frame(end="2026-10-08", n=120, close=100.0):
    idx = pd.bdate_range(end=end, periods=n, name="Date")
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close}, index=idx)


def report_args(**kw):
    from types import SimpleNamespace
    return SimpleNamespace(**{"amount": None, "exit": "bottom", **kw})


LIVE = {"Open": 120.0, "High": 125.0, "Low": 118.0, "Close": 123.0,
        "time": pd.Timestamp("2026-10-12 11:12")}


def test_shouldUseLivePriceAsTodayWhenMarketOpen(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pf, "load_yahoo", lambda code: (yahoo_frame(), code + ".KS"))
    trades = trades_from("005930,삼성전자,2026-10-01,매수,1,100,10,9000\n", tmp_path)
    pf.report("005930", trades, report_args(), quote=LIVE)
    out = capsys.readouterr().out
    assert "기준일 2026-10-12" in out and "현재가(장중 11:12) 123.00" in out


def test_shouldKeepYahooCloseWhenNoLiveQuote(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pf, "load_yahoo", lambda code: (yahoo_frame(), code + ".KS"))
    trades = trades_from("005930,삼성전자,2026-10-01,매수,1,100,10,9000\n", tmp_path)
    pf.report("005930", trades, report_args())
    out = capsys.readouterr().out
    assert "기준일 2026-10-08" in out and "종가 100.00" in out and "장중" not in out


def test_shouldWarnTodaySignalMayChangeWhenLive(tmp_path, monkeypatch, capsys):
    # 1/9 보유 + 오늘 장중 급등으로 MACD 골든크로스 → 할 일에 시세시각이 붙고 경고가 나온다
    raw = yahoo_frame()
    raw.iloc[-30:, :] = [[100.0 - i] * 4 for i in range(30)]
    monkeypatch.setattr(pf, "load_yahoo", lambda code: (raw, code + ".KS"))
    trades = trades_from("005930,삼성전자,2026-10-08,매수,1,71,10,9000\n", tmp_path)
    todo = pf.report("005930", trades, report_args(), quote={**LIVE, "Close": 90.0, "High": 90.0})
    out = capsys.readouterr().out
    assert [a["사유"] for a in todo] == ["MACD 골든크로스"]
    assert todo[0]["시세시각"] == "2026-10-12 11:12" and "장 마감 전까지" in out
