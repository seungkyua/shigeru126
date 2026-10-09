# -*- coding: utf-8 -*-
"""shigeru_start (1:2:6 매수 시작 종목 찾기) 테스트 — 네트워크 없이 실행"""
import pandas as pd
import pytest

from pathlib import Path

from firstbuy import shigeru_start as st


def indicator_frame(rsi_down30_flags, close=100.0, rsi=28.0):
    """add_indicators 결과처럼 RSI/RSI_DOWN30/Close 컬럼만 가진 DataFrame."""
    idx = pd.date_range("2026-10-01", periods=len(rsi_down30_flags), freq="B")
    return pd.DataFrame({"Close": close, "RSI": rsi, "RSI_DOWN30": rsi_down30_flags}, index=idx)


STOCKS = pd.DataFrame({"종목코드": ["005930", "035720"], "종목명": ["삼성전자", "카카오"],
                       "시장": ["KOSPI", "KOSDAQ"]})


# ---------------------------------------------------------------- 종목 리스트
def test_shouldMapKospiToKsSuffix():
    assert st.yahoo_symbol("005930", "KOSPI") == "005930.KS"


def test_shouldMapKosdaqToKqSuffix():
    assert st.yahoo_symbol("035720", "KOSDAQ") == "035720.KQ"


def test_shouldLoadCodesAsSixDigitStringsWhenCsvHasBom(tmp_path):
    p = tmp_path / "list.csv"
    p.write_text("종목코드,종목명,시장\n5930,삼성전자,KOSPI\n", encoding="utf-8-sig")
    assert st.load_stock_list(str(p))["종목코드"].tolist() == ["005930"]


def test_shouldKeepOnlyRequestedMarketWhenMarketFilterGiven():
    assert st.filter_stocks(STOCKS, market="KOSDAQ")["종목명"].tolist() == ["카카오"]


def test_shouldKeepFirstNStocksWhenLimitGiven():
    assert len(st.filter_stocks(STOCKS, limit=1)) == 1


# ---------------------------------------------------------------- 다운로드 결과 분리
def test_shouldSplitMultiIndexDownloadIntoPerTickerFrames():
    idx = pd.date_range("2026-10-01", periods=2)
    cols = pd.MultiIndex.from_product([["A.KS", "B.KQ"], ["Open", "High", "Low", "Close"]])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    assert sorted(st.split_download(raw, ["A.KS", "B.KQ"])) == ["A.KS", "B.KQ"]


def test_shouldDropTickerWhenAllPricesMissing():
    idx = pd.date_range("2026-10-01", periods=2)
    cols = pd.MultiIndex.from_product([["A.KS", "B.KQ"], ["Open", "High", "Low", "Close"]])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    raw.loc[:, "B.KQ"] = float("nan")
    assert list(st.split_download(raw, ["A.KS", "B.KQ"])) == ["A.KS"]


# ---------------------------------------------------------------- 매수 시작 신호
def test_shouldReturnSignalDateWhenRsiCrossedBelow30OnLastDay():
    d = indicator_frame([False, False, True])
    assert st.start_signal_date(d, days=1) == d.index[-1]


def test_shouldReturnNoneWhenCrossIsOlderThanWindow():
    d = indicator_frame([False, True, False])
    assert st.start_signal_date(d, days=1) is None


def test_shouldReturnRecentCrossWhenWindowCoversIt():
    d = indicator_frame([False, True, False])
    assert st.start_signal_date(d, days=2) == d.index[1]


# ---------------------------------------------------------------- 1단계 매수 금액
def test_shouldComputeFirstBuyAsOneNinthOfAmount():
    assert st.first_buy(9_000_000, 300.0) == (1_000_000, 3333)


def test_shouldReturnZeroSharesWhenPriceExceedsFirstBuyMoney():
    assert st.first_buy(900_000, 200_000.0) == (100_000, 0)


# ---------------------------------------------------------------- 후보 조립 및 출력
def test_shouldBuildCandidateWithNameWhenSignalFound():
    frames = {"005930.KS": indicator_frame([False, True], close=60_000.0, rsi=27.5),
              "035720.KQ": indicator_frame([False, False])}
    rows = st.find_candidates(STOCKS, frames, days=1)
    assert [(r["종목명"], r["RSI"]) for r in rows] == [("삼성전자", 27.5)]


def test_shouldSortCandidatesByRsiAscending():
    frames = {"005930.KS": indicator_frame([True], rsi=29.0),
              "035720.KQ": indicator_frame([True], rsi=25.0)}
    rows = st.find_candidates(STOCKS, frames, days=1)
    assert [r["종목명"] for r in rows] == ["카카오", "삼성전자"]


def test_shouldShowFirstBuyAmountInReportWhenAmountGiven():
    frames = {"005930.KS": indicator_frame([True], close=60_000.0)}
    rows = st.find_candidates(STOCKS, frames, days=1)
    text = "\n".join(st.format_candidates(rows, amount=9_000_000))
    assert "1,000,000" in text and "16주" in text


def test_shouldWarnWhenAmountTooSmallForOneShare():
    frames = {"005930.KS": indicator_frame([True], close=200_000.0)}
    rows = st.find_candidates(STOCKS, frames, days=1)
    text = "\n".join(st.format_candidates(rows, amount=900_000))
    assert "금액 부족" in text


def test_shouldSayNoCandidatesWhenListEmpty():
    assert any("매수 시작 종목이 없습니다" in l for l in st.format_candidates([], amount=None))


def test_shouldRejectNonPositiveAmount():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        st.positive_amount("-1")


# ---------------------------------------------------------------- 종류(ETF/ETN) 필터
TYPED = pd.DataFrame({"종목코드": ["005930", "005935", "069500", "500001"],
                      "종목명": ["삼성전자", "삼성전자우", "KODEX 200", "어떤 ETN"],
                      "시장": ["KOSPI"] * 4, "종류": ["stock", "stock", "etf", "etn"]})


def test_shouldKeepOnlyStocksAndPreferredWhenStockOnly():
    assert st.filter_stocks(TYPED)["종목명"].tolist() == ["삼성전자", "삼성전자우"]


def test_shouldKeepEtfAndEtnWhenStockOnlyDisabled():
    assert len(st.filter_stocks(TYPED, stock_only=False)) == 4


def test_shouldNotFailWhenTypeColumnMissing():
    assert len(st.filter_stocks(STOCKS)) == 2


def test_shouldKeepTypeColumnWhenLoadingList(tmp_path):
    p = tmp_path / "list.csv"
    p.write_text("종목코드,종목명,시장,종류\n069500,KODEX 200,KOSPI,etf\n", encoding="utf-8-sig")
    assert st.load_stock_list(str(p))["종류"].tolist() == ["etf"]


# ---------------------------------------------------------------- 최소 주가 필터
def test_shouldExcludeCandidateWhenPriceBelowMinPrice():
    frames = {"005930.KS": indicator_frame([True], close=4_990.0),
              "035720.KQ": indicator_frame([True], close=5_000.0)}
    rows = st.find_candidates(STOCKS, frames, days=1, min_price=5_000)
    assert [r["종목명"] for r in rows] == ["카카오"]


def test_shouldShowMinPriceInHeaderWhenGiven():
    frames = {"005930.KS": indicator_frame([True], close=60_000.0)}
    rows = st.find_candidates(STOCKS, frames, days=1)
    assert any("5,000원 미만 제외" in l for l in st.format_candidates(rows, None, min_price=5_000))


def test_shouldRejectNegativeMinPrice():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        st.non_negative_price("-1")


def test_shouldAcceptCommaInMinPrice():
    assert st.non_negative_price("10,000") == 10_000


# ---------------------------------------------------------------- 결과 파일
def test_shouldNameOutputFileByLatestBaseDate():
    frames = {"005930.KS": indicator_frame([False, True])}
    rows = st.find_candidates(STOCKS, frames, days=1)
    expected = Path(st.__file__).parent / f"shigeru_start_{rows[0]['기준일']:%Y%m%d}.csv"
    assert st.default_out_path(rows) == str(expected)


def test_shouldSaveFirstBuyAmountAndIntegerSharesToCsv(tmp_path):
    frames = {"005930.KS": indicator_frame([True], close=60_000.0)}
    rows = st.find_candidates(STOCKS, frames, days=1)
    out = tmp_path / "out.csv"
    st.save_candidates(rows, str(out), amount=9_000_000)
    saved = pd.read_csv(out, encoding="utf-8-sig", dtype={"종목코드": str})
    assert saved.loc[0, ["종목코드", "1단계금액", "주수"]].tolist() == ["005930", 1_000_000, 16]


def test_shouldDefaultStockListToStockDirectory():
    root = Path(st.__file__).resolve().parent.parent
    assert Path(st.DEFAULT_LIST) == root / "stock" / "stock_list_krx.csv"


def test_shouldGuideToPortfolioProgramWithNewPathAfterCandidates():
    frames = {"005930.KS": indicator_frame([True], close=60_000.0)}
    rows = st.find_candidates(STOCKS, frames, days=1)
    text = "\n".join(st.format_candidates(rows, amount=None))
    assert "buynsell/portfolio.csv" in text and "python buynsell/shigeru_portfolio.py" in text
