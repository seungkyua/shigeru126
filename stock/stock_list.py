#!/usr/bin/env python3
"""
네이버 증권 API로 종목 리스트 관리

사용법 (프로젝트 루트에서)
  python stock/stock_list.py      # stock/stock_list_krx.csv 가 없으면 새로 받아서 저장
  최신으로 갱신하려면 stock/stock_list_krx.csv 를 지우고 다시 실행하세요.
"""

import pandas as pd
import requests
import logging
import os
from pathlib import Path
import time
from datetime import datetime
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockListManager:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        # 실행 위치와 상관없이 이 파일과 같은 stock/ 디렉토리의 CSV를 사용
        self.stock_file = str(Path(__file__).resolve().parent / 'stock_list_krx.csv')

    MARKET_VALUE_API = 'https://m.stock.naver.com/api/stocks/marketValue/{market}'
    
    def get_stock_list_from_internet(self):
        """인터넷에서 종목 리스트 가져오기"""
        try:
            logger.info("인터넷에서 종목 리스트 가져오는 중...")
            
            # 네이버 금융에서 종목 리스트 스크래핑
            df = self.scrape_stocks_from_naver_finance()
            
            if not df.empty:
                # 중복 제거
                df = df.drop_duplicates(subset=['종목코드'], keep='first')
                
                # 종목코드를 6자리로 맞추기
                df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
                
                # 시장 컬럼이 없으면 기본값 설정
                if '시장' not in df.columns:
                    df['시장'] = 'KOSPI'
                
                logger.info(f"총 {len(df)}개 종목 가져옴")
                return df
            else:
                logger.error("인터넷에서 종목 데이터를 가져올 수 없습니다.")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"인터넷에서 종목 리스트 가져오기 실패: {e}")
            return pd.DataFrame()
    
    def scrape_stocks_from_naver_finance(self):
        """네이버 증권 API에서 종목 리스트 가져오기"""
        try:
            logger.info("네이버 증권에서 종목 리스트 가져오는 중...")
            
            stocks = []
            
            # 기존 finance.naver.com/sise/sise_market_sum 페이지는 stock.naver.com(SPA)으로
            # 리다이렉트되어 HTML 테이블이 없으므로, 새 사이트가 사용하는 JSON API를 사용
            logger.info("KOSPI 종목 가져오기 시작...")
            stocks.extend(self.scrape_market_stocks("KOSPI"))
            
            logger.info("KOSDAQ 종목 가져오기 시작...")
            stocks.extend(self.scrape_market_stocks("KOSDAQ"))
            
            if stocks:
                # 중복 제거
                df = pd.DataFrame(stocks)
                df = df.drop_duplicates(subset=['종목코드'], keep='first')
                logger.info(f"네이버 증권에서 총 {len(df)}개 종목 발견")
                return df
            else:
                logger.warning("네이버 증권에서 종목을 찾을 수 없습니다.")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"네이버 증권 종목 리스트 가져오기 실패: {e}")
            return pd.DataFrame()
    
    def scrape_market_stocks(self, market_name, page_size=100):
        """특정 시장(KOSPI/KOSDAQ)의 모든 종목을 시가총액 순으로 가져오기"""
        stocks = []
        seen_codes = set()
        page = 1
        total_pages = 1  # 첫 응답의 totalCount로 갱신
        
        while page <= total_pages:
            try:
                logger.info(f"{market_name} 페이지 {page} 가져오는 중...")
                response = self.session.get(
                    self.MARKET_VALUE_API.format(market=market_name),
                    params={'page': page, 'pageSize': page_size},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                
                if page == 1:
                    total_count = int(data.get('totalCount', 0))
                    total_pages = max(1, -(-total_count // page_size))
                    logger.info(f"{market_name} 총 {total_count}개 종목, {total_pages}페이지")
                
                items = data.get('stocks') or []
                if not items:
                    logger.warning(f"{market_name} 페이지 {page}에서 종목을 찾을 수 없습니다.")
                    break
                
                page_count = 0
                for item in items:
                    stock_code = str(item.get('itemCode', ''))
                    stock_name = (item.get('stockName') or '').strip()
                    if re.match(r'^\d{6}$', stock_code) and stock_name and stock_code not in seen_codes:
                        seen_codes.add(stock_code)
                        stocks.append({
                            '종목코드': stock_code,
                            '종목명': stock_name,
                            '시장': market_name,
                            # stock(일반주·우선주) / etf / etn
                            '종류': item.get('stockEndType') or '',
                        })
                        page_count += 1
                
                logger.info(f"{market_name} 페이지 {page}에서 {page_count}개 종목 발견 (누적: {len(stocks)}개)")
                
            except Exception as e:
                logger.warning(f"{market_name} 페이지 {page} 가져오기 실패: {e}")
            
            page += 1
            # 너무 빠른 요청 방지
            time.sleep(0.2)
        
        return stocks
    
    def save_stock_list(self, stock_list):
        """종목 리스트를 CSV 파일에 저장"""
        try:
            if not stock_list.empty:
                stock_list.to_csv(self.stock_file, index=False, encoding='utf-8-sig')
                logger.info(f"종목 리스트가 {self.stock_file}에 저장되었습니다.")
                return True
            else:
                logger.warning("저장할 종목 리스트가 없습니다.")
                return False
        except Exception as e:
            logger.error(f"종목 리스트 저장 실패: {e}")
            return False
    
    def load_stock_list_from_file(self):
        """CSV 파일에서 종목 리스트 읽기"""
        try:
            if os.path.exists(self.stock_file):
                stock_list = pd.read_csv(self.stock_file, encoding='utf-8-sig')
                # 종목코드를 6자리로 맞추기
                stock_list['종목코드'] = stock_list['종목코드'].astype(str).str.zfill(6)
                # 시장 컬럼이 없으면 기본값 설정
                if '시장' not in stock_list.columns:
                    stock_list['시장'] = 'KOSPI'
                logger.info(f"파일에서 {len(stock_list)}개 종목 읽어옴: {self.stock_file}")
                return stock_list
            else:
                logger.info(f"종목 리스트 파일이 없습니다: {self.stock_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"파일에서 종목 리스트 읽기 실패: {e}")
            return pd.DataFrame()
    
    def get_stock_list(self):
        """종목 리스트 가져오기 (CSV 파일 우선, 없으면 인터넷에서 가져오기)"""
        # 먼저 CSV 파일에서 읽기 시도
        stock_list = self.load_stock_list_from_file()
        
        if not stock_list.empty:
            logger.info("저장된 CSV 파일에서 종목 리스트를 사용합니다.")
            return stock_list
        
        # CSV 파일이 없으면 인터넷에서 가져오기
        logger.info("저장된 CSV 파일이 없어서 인터넷에서 종목 리스트를 가져옵니다.")
        stock_list = self.get_stock_list_from_internet()
        
        if not stock_list.empty:
            # 가져온 데이터를 CSV 파일에 저장
            self.save_stock_list(stock_list)
            return stock_list
        else:
            logger.error("인터넷에서도 종목 리스트를 가져올 수 없습니다.")
            return pd.DataFrame()

def get_stock_list():
    """종목 리스트 가져오기 함수 (기존 호환성 유지)"""
    stock_manager = StockListManager()
    return stock_manager.get_stock_list()

if __name__ == "__main__":
    # 테스트 실행
    stock_manager = StockListManager()
    stock_list = stock_manager.get_stock_list()
    
    if not stock_list.empty:
        print(f"총 {len(stock_list)}개 종목:")
        print(stock_list.head(10))
        
        # 시장별 분포
        if '시장' in stock_list.columns:
            market_dist = stock_list['시장'].value_counts()
            print(f"\n시장별 분포:")
            for market, count in market_dist.items():
                print(f"  {market}: {count}개")
        else:
            print(f"\n시장 정보 없음 (기본값: KOSPI)")
    else:
        print("종목 리스트를 가져올 수 없습니다.") 