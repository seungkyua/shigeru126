# -*- coding: utf-8 -*-
"""stock_list 테스트 — 네이버 API 응답을 가짜 세션으로 대체"""
from pathlib import Path

from stock import stock_list


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class FakeSession:
    def get(self, url, params=None, timeout=None):
        return FakeResponse({"totalCount": 2, "stocks": [
            {"itemCode": "005930", "stockName": "삼성전자", "stockEndType": "stock"},
            {"itemCode": "069500", "stockName": "KODEX 200", "stockEndType": "etf"},
        ]})


def test_shouldSaveStockEndTypeAsTypeColumn(monkeypatch):
    monkeypatch.setattr(stock_list.time, "sleep", lambda s: None)
    m = stock_list.StockListManager()
    m.session = FakeSession()
    assert [s["종류"] for s in m.scrape_market_stocks("KOSPI")] == ["stock", "etf"]


def test_shouldKeepStockListCsvInStockDirectory():
    m = stock_list.StockListManager()
    assert Path(m.stock_file) == Path(stock_list.__file__).parent / "stock_list_krx.csv"
