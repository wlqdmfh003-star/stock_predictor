import pandas as pd
import numpy as np
import yfinance as yf
import os
import pickle
try:
    from pykrx import stock as krx
except Exception:
    krx = None
    print("WARNING: pykrx import failed — KRX 기반 데이터는 사용 불가 (pykrx 설치 또는 환경 확인 필요)")
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')


class DataFetcher:
    def __init__(self, market="KOSPI+KOSDAQ", top_n=200,
                 min_market_cap=100_000_000_000, min_volume_bil=50):
        self.market         = market
        self.top_n          = top_n
        self.min_market_cap = min_market_cap
        self.min_volume_bil = min_volume_bil
        self.today          = datetime.now().strftime("%Y%m%d")
        # 일봉 1년 / 주봉 2년 / 월봉 5년
        self.yf_start_daily   = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        self.yf_start_weekly  = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        self.yf_start_monthly = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")

    # ── 종목 리스트 ───────────────────────────────────────────────────────────
    def _get_ticker_list(self):
        markets = []
        if "KOSPI"  in self.market: markets.append("KOSPI")
        if "KOSDAQ" in self.market: markets.append("KOSDAQ")

        tickers = []
        self._ticker_market_map = {}
        self._ticker_name_map   = {}

        for mkt in markets:
            try:
                t = krx.get_market_ticker_list(self.today, market=mkt)
                if t and len(t) > 0:
                    for code in t:
                        if code not in self._ticker_market_map:
                            self._ticker_market_map[code] = mkt
                    tickers.extend(t)
                    print(f"[OK] pykrx {mkt} 종목리스트: {len(t)}개")
                    try:
                        for code in t:
                            if code not in self._ticker_name_map:
                                n = krx.get_market_ticker_name(code)
                                if n and str(n).strip():
                                    self._ticker_name_map[code] = str(n).strip()
                    except Exception:
                        pass
                    continue
            except Exception:
                pass

            # 네이버 폴백
            try:
                import requests
                from bs4 import BeautifulSoup
                mkt_code = "0" if mkt == "KOSPI" else "1"
                t_naver  = []
                for page in range(1, 40):
                    url  = (f"https://finance.naver.com/sise/sise_market_sum.naver"
                            f"?sosok={mkt_code}&page={page}")
                    resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
                    from bs4 import BeautifulSoup
                    soup  = BeautifulSoup(resp.text, "html.parser")
                    rows  = soup.select("table.type_2 tbody tr")
                    found = 0
                    for row in rows:
                        a = row.select_one("td a[href*='code=']")
                        if a:
                            code = a["href"].split("code=")[-1].strip()
                            name = a.get_text(strip=True)
                            if len(code)==6 and code.isdigit():
                                t_naver.append(code)
                                if code not in self._ticker_market_map:
                                    self._ticker_market_map[code] = mkt
                                if name and code not in self._ticker_name_map:
                                    self._ticker_name_map[code] = name
                                found += 1
                    if found == 0: break
                    time.sleep(0.05)
                tickers.extend(t_naver)
                print(f"[OK] 네이버 {mkt} 종목리스트: {len(t_naver)}개")
            except Exception as e:
                print(f"[WARN] {mkt} 종목리스트 수집 실패: {e}")

        return list(set(tickers))

    # ... (snapshot of current file)
