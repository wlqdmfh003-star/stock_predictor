import pandas as pd
import numpy as np
import yfinance as yf
import os
import pickle
try:
    from pykrx import stock as krx
except Exception:
    krx = None
    print("WARNING: pykrx import failed — KRX 기반 데이터는 사용 불가")
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
        self.yf_start_daily   = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        self.yf_start_weekly  = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        self.yf_start_monthly = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")

    # ── 종목 리스트 ───────────────────────────────────────────────────────────
    def _get_ticker_list(self):
        # ★ v7.2: 캐시 - 같은 날 같은 시장은 재수집 안 함 (2번 불러오기 방지)
        cache_key = f"{self.today}_{self.market}"
        if hasattr(self, '_ticker_cache_key') and self._ticker_cache_key == cache_key:
            return list(self._ticker_market_map.keys())

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

        self._ticker_cache_key = cache_key  # ★ 캐시 키 저장
        return list(set(tickers))

    # ── 사전 스코어링 ─────────────────────────────────────────────────────────
    def _get_filtered_tickers(self):
        tickers = self._get_ticker_list()
        if not tickers:
            return self._default_tickers()
        ticker_set = set(tickers)

        try:
            markets = []
            if "KOSPI"  in self.market: markets.append("KOSPI")
            if "KOSDAQ" in self.market: markets.append("KOSDAQ")

            cap_frames, vol_frames = [], []
            for mkt in markets:
                try:
                    c = krx.get_market_cap_by_ticker(self.today, market=mkt)
                    cap_frames.append(c)
                except Exception:
                    pass
                try:
                    v = krx.get_market_trading_value_by_ticker(self.today, market=mkt)
                    vol_frames.append(v)
                except Exception:
                    pass

            if not cap_frames:
                return tickers[:self.top_n]

            cap_df = pd.concat(cap_frames)
            cap_df = cap_df[~cap_df.index.duplicated(keep='first')]
            cap_df = cap_df[cap_df["시가총액"] >= self.min_market_cap]

            if vol_frames:
                vol_df = pd.concat(vol_frames)
                vol_df = vol_df[~vol_df.index.duplicated(keep='first')]
                if "거래대금" in vol_df.columns:
                    vol_df = vol_df[vol_df["거래대금"] >= self.min_volume_bil*1e8]
                    valid  = set(cap_df.index) & set(vol_df.index)
                else:
                    valid = set(cap_df.index)
            else:
                valid = set(cap_df.index)

            filtered = cap_df[cap_df.index.isin(valid) & cap_df.index.isin(ticker_set)].copy()

            # ★ v7.2 개선: ETF/리츠/스팩/우선주 완전 제외
            etf_keywords = [
                # 국내 ETF 운용사 브랜드명
                'KODEX','TIGER','KBSTAR','ARIRANG','HANARO',
                'KOSEF','FOCUS','SOL','ACE','PLUS','RISE','KIWOOM',
                'TIMEFOLIO','SMART','TREX','WON','BNK','MARVELEX',
                'HANA','NH','POSCO','KCGI','KB','MIRAE','MERITZ',
                # ETF 유형 키워드
                'ETF','리츠','REIT','스팩','SPAC',
                '채권','국고채','회사채','단기채','통안채',
                '레버리지','인버스','선물','합성',
                # 해외 ETF 키워드
                'S&P','나스닥','다우존스','MSCI','RUSSELL',
                '미국','중국','일본','인도','베트남','유럽',
                '글로벌','차이나','CHINA','GLOBAL',
                # 인덱스 펀드 키워드
                '200','150','100','TOP5','TDF',
                'ESG','배당','가치','성장','모멘텀',
                # 커버드콜/액티브 ETF
                '커버드콜','배당귀족','밸류업',
            ]

            # pykrx에서 직접 종목명 가져와서 ETF/우선주 필터링
            etf_codes = set()
            try:
                # 전체 종목명 한번에 가져오기
                today_str = datetime.now().strftime("%Y%m%d")
                kospi_names  = krx.get_market_ticker_list(today_str, market="KOSPI")
                kosdaq_names = krx.get_market_ticker_list(today_str, market="KOSDAQ")
                all_tickers  = list(kospi_names) + list(kosdaq_names)

                # 종목명 딕셔너리 구성
                name_map = {}
                for code in filtered.index.tolist():
                    try:
                        name = krx.get_market_ticker_name(code)
                        name_map[code] = str(name) if name else ""
                    except Exception:
                        name_map[code] = ""

                for code, name in name_map.items():
                    # ETF 키워드 필터
                    if any(kw in name for kw in etf_keywords):
                        etf_codes.add(code)
                    # 우선주 필터 (종목명 끝이 '우','우B','우C','1우' 등)
                    if name.endswith(('우', '우B', '우C', '1우', '2우', '3우')):
                        etf_codes.add(code)
                    # 스팩 필터 (종목명에 '스팩','SPAC' 포함)
                    if '스팩' in name or 'SPAC' in name.upper():
                        etf_codes.add(code)

            except Exception as _e:
                # pykrx 실패 시 내부 캐시로 필터링
                name_map = getattr(self, '_ticker_name_map', {})
                for code, name in name_map.items():
                    if any(kw in str(name) for kw in etf_keywords):
                        etf_codes.add(code)

            if etf_codes:
                before = len(filtered)
                filtered = filtered[~filtered.index.isin(etf_codes)]
                print(f"  [ETF/리츠/스팩 제외] {len(etf_codes)}개 제외 → 순수 주식 {len(filtered)}개 남음")

            if vol_frames:
                vm = pd.concat(vol_frames)
                vm = vm[~vm.index.duplicated(keep='first')]
                if "거래대금" in vm.columns:
                    filtered["거래대금"] = filtered.index.map(
                        lambda c: float(vm.loc[c,"거래대금"]) if c in vm.index else 0.0)
                else:
                    filtered["거래대금"] = 0.0
            else:
                filtered["거래대금"] = 0.0

            kospi_codes  = {c for c,m in self._ticker_market_map.items() if m=="KOSPI"}
            kosdaq_codes = {c for c,m in self._ticker_market_map.items() if m=="KOSDAQ"}

            pool_size   = 200
            kospi_pool  = filtered[filtered.index.isin(kospi_codes)]\
                          .sort_values("거래대금",ascending=False).index.tolist()[:pool_size]
            kosdaq_pool = filtered[filtered.index.isin(kosdaq_codes)]\
                          .sort_values("거래대금",ascending=False).index.tolist()[:pool_size]
            pool        = kospi_pool + kosdaq_pool

            print(f"[분석] 사전 스코어링 중... (KOSPI {len(kospi_pool)}개 / KOSDAQ {len(kosdaq_pool)}개)")

            if not kospi_pool and not kosdaq_pool:
                return filtered.sort_values("거래대금",ascending=False).index.tolist()[:self.top_n]

            pre_scores = self._score_candidates_pykrx(pool)

            half = self.top_n // 2
            kospi_top  = sorted([c for c in kospi_pool  if c in pre_scores],
                                key=lambda c: pre_scores[c], reverse=True)[:half]
            kosdaq_top = sorted([c for c in kosdaq_pool if c in pre_scores],
                                key=lambda c: pre_scores[c], reverse=True)[:half]

            result, used = [], set()
            for k, q in zip(kospi_top, kosdaq_top):
                result.append(k); used.add(k)
                result.append(q); used.add(q)
            extra = sorted([c for c in pool if c not in used and c in pre_scores],
                           key=lambda c: pre_scores[c], reverse=True)
            result += extra
            return result[:self.top_n]

        except Exception as e:
            print(f"필터 오류: {e}")
            # ★ v7.2: 오류 시에도 yfinance로 시가총액 확인
            try:
                valid_codes = []
                for code in tickers:
                    try:
                        for suffix in [".KS", ".KQ"]:
                            info = yf.Ticker(f"{code}{suffix}").info
                            mc  = float(info.get("marketCap") or 0)
                            if mc >= self.min_market_cap:
                                valid_codes.append(code)
                                break
                    except Exception:
                        continue
                    if len(valid_codes) >= self.top_n:
                        break
                return valid_codes[:self.top_n] if valid_codes else tickers[:self.top_n]
            except Exception:
                return tickers[:self.top_n]

    def _score_candidates_pykrx(self, codes):
        scores    = {}
        start_30d = (datetime.now()-timedelta(days=50)).strftime("%Y%m%d")
        for code in codes:
            try:
                ohlcv  = krx.get_market_ohlcv_by_date(start_30d, self.today, code)
                if ohlcv is None or len(ohlcv) < 10:
                    scores[code] = 0.0; continue
                close  = ohlcv["종가"].astype(float).values
                volume = ohlcv["거래량"].astype(float).values
                score  = 0.0
                if len(close)>=6:  score += np.clip((close[-1]/close[-6]-1)*100*2,-20,20)
                if len(close)>=21: score += np.clip((close[-1]/close[-21]-1)*100*1.5,-20,20)
                if len(volume)>=11:
                    r = volume[-1]/(volume[-11:-1].mean()+1e-9)
                    score += 25 if r>=3 else 15 if r>=2 else 8 if r>=1.5 else 0
                if len(close)>=15:
                    d = np.diff(close[-15:])
                    g = np.where(d>0,d,0).mean(); l = np.where(d<0,-d,0).mean()
                    rsi = 100-100/(1+g/(l+1e-9))
                    score += 20 if 30<=rsi<=45 else 10 if 45<rsi<=55 else 5 if rsi<30 else -15 if rsi>75 else 0
                if len(ohlcv)>=2:
                    ph=float(ohlcv["고가"].iloc[-2]); pl=float(ohlcv["저가"].iloc[-2])
                    po=float(ohlcv["시가"].iloc[-1])
                    if close[-1]>=po+(ph-pl)*0.5: score+=20
                if len(close)>=20 and close.max()>0 and close[-1]/close.max()>=0.90: score+=10
                if len(close)>=21:
                    ma5=close[-5:].mean(); ma20=close[-20:].mean()
                    score += 10 if ma5>ma20 else -5
                scores[code] = float(score)
            except Exception:
                scores[code] = 0.0
            time.sleep(0.03)
        return scores

    # ── yfinance 재시도 래퍼 (업로드 파일) ─────────────────────────────────
    def _history_with_retry(self, ticker, start, interval=None, attempts=3, backoff=1.0):
        for i in range(attempts):
            try:
                if interval:
                    df = ticker.history(start=start, interval=interval, auto_adjust=True)
                else:
                    df = ticker.history(start=start, auto_adjust=True)
                return df
            except Exception:
                time.sleep(backoff * (2 ** i))
        return None

    # ── 캐시 로드/저장 (업로드 파일) ────────────────────────────────────────
    def _cache_path(self, code):
        d = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
        d = os.path.abspath(d)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{code}.pkl")

    def _load_cache(self, code):
        p = self._cache_path(code)
        try:
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            return None
        return None

    def _save_cache(self, code, daily, weekly, monthly):
        p = self._cache_path(code)
        try:
            with open(p, 'wb') as f:
                pickle.dump({'daily':daily, 'weekly':weekly, 'monthly':monthly}, f)
        except Exception:
            pass

    def _log_failed(self, code):
        p = os.path.join(os.path.dirname(__file__), '..', 'data', 'fetch_failures.txt')
        p = os.path.abspath(p)
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'a', encoding='utf-8') as f:
                f.write(f"{code}\n")
        except Exception:
            pass

    # ── 일봉+주봉+월봉 수집 (업로드 파일 + v7.2 개선) ────────────────────
    def _fetch_single(self, code):
        code_clean = str(code).replace("$", "").replace(" ", "").strip()
        if not code_clean or not code_clean.isdigit() or len(code_clean) != 6:
            return None
        code = code_clean

        market_map = getattr(self, '_ticker_market_map', {})
        mkt        = market_map.get(code, "")
        suffixes   = [".KS",".KQ"] if mkt!="KOSDAQ" else [".KQ",".KS"]

        # 1) pykrx 우선 수집 (업로드 파일)
        if krx is not None:
            try:
                start_daily   = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
                start_weekly  = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
                start_monthly = (datetime.now() - timedelta(days=365*5)).strftime("%Y%m%d")
                ohlcv_d = krx.get_market_ohlcv_by_date(start_daily, self.today, code)
                if ohlcv_d is not None and len(ohlcv_d) >= 60:
                    daily = pd.DataFrame(index=pd.to_datetime(ohlcv_d.index))
                    daily["close"]  = ohlcv_d["종가"].astype(float).values
                    daily["open"]   = ohlcv_d["시가"].astype(float).values
                    daily["high"]   = ohlcv_d["고가"].astype(float).values
                    daily["low"]    = ohlcv_d["저가"].astype(float).values
                    daily["volume"] = ohlcv_d.get("거래량", pd.Series(0.0, index=ohlcv_d.index)).astype(float).values

                    weekly = None
                    try:
                        ohlcv_w = krx.get_market_ohlcv_by_date(start_weekly, self.today, code)
                        if ohlcv_w is not None and len(ohlcv_w) >= 10:
                            weekly = pd.DataFrame(index=pd.to_datetime(ohlcv_w.index))
                            weekly["close"]  = ohlcv_w["종가"].astype(float).values
                            weekly["open"]   = ohlcv_w["시가"].astype(float).values
                            weekly["high"]   = ohlcv_w["고가"].astype(float).values
                            weekly["low"]    = ohlcv_w["저가"].astype(float).values
                            weekly["volume"] = ohlcv_w.get("거래량", pd.Series(0.0, index=ohlcv_w.index)).astype(float).values
                    except Exception:
                        weekly = None

                    monthly = None
                    try:
                        ohlcv_m = krx.get_market_ohlcv_by_date(start_monthly, self.today, code)
                        if ohlcv_m is not None and len(ohlcv_m) >= 6:
                            monthly = pd.DataFrame(index=pd.to_datetime(ohlcv_m.index))
                            monthly["close"]  = ohlcv_m["종가"].astype(float).values
                            monthly["open"]   = ohlcv_m["시가"].astype(float).values
                            monthly["high"]   = ohlcv_m["고가"].astype(float).values
                            monthly["low"]    = ohlcv_m["저가"].astype(float).values
                            monthly["volume"] = ohlcv_m.get("거래량", pd.Series(0.0, index=ohlcv_m.index)).astype(float).values
                    except Exception:
                        monthly = None

                    # ★ 시가총액 수집
                    try:
                        mc  = krx.get_market_cap_by_date(self.today, self.today, code)
                        cap = float(mc["시가총액"].iloc[-1]) if len(mc) > 0 else 0.0
                    except Exception:
                        cap = 0.0

                    last = daily.iloc[-1]; prev = daily.iloc[-2] if len(daily) > 1 else daily.iloc[-1]
                    close = float(last.get('close', 0)); vol = float(last.get('volume', 0))
                    prev_close = float(prev.get('close', 0))
                    if close <= 0:
                        return None

                    name_map = getattr(self, '_ticker_name_map', {})
                    name = name_map.get(code, code)
                    stock_market = market_map.get(code, "KOSPI")

                    return {
                        "code": code, "name": name, "market": stock_market,
                        "current_price": close,
                        "open": float(last.get('open', 0)),
                        "high": float(last.get('high', 0)),
                        "low":  float(last.get('low', 0)),
                        "volume": vol, "volume_bil": vol * close / 1e8,
                        "market_cap": cap,
                        "ohlcv": daily, "ohlcv_weekly": weekly, "ohlcv_monthly": monthly,
                        "prev_close": prev_close,
                        "prev_high": float(prev.get('high', 0)),
                        "prev_low":  float(prev.get('low', 0)),
                        "change_pct": (close / prev_close - 1) * 100 if prev_close > 0 else 0.0,
                    }
            except Exception:
                pass

        # 2) 캐시 확인 (업로드 파일)
        try:
            cached = self._load_cache(code)
            if cached:
                daily   = cached.get('daily')
                weekly  = cached.get('weekly')
                monthly = cached.get('monthly')
                if daily is not None and len(daily) >= 60:
                    last = daily.iloc[-1]; prev = daily.iloc[-2] if len(daily) > 1 else daily.iloc[-1]
                    close = float(last.get('close', 0)); vol = float(last.get('volume', 0))
                    prev_close = float(prev.get('close', 0))
                    if close > 0:
                        name_map = getattr(self, '_ticker_name_map', {})
                        name = name_map.get(code, code)
                        stock_market = market_map.get(code, "KOSPI")
                        return {
                            "code": code, "name": name, "market": stock_market,
                            "current_price": close,
                            "open": float(last.get('open', 0)),
                            "high": float(last.get('high', 0)),
                            "low":  float(last.get('low', 0)),
                            "volume": vol, "volume_bil": vol * close / 1e8,
                            "market_cap": 0.0,
                            "ohlcv": daily, "ohlcv_weekly": weekly, "ohlcv_monthly": monthly,
                            "prev_close": prev_close,
                            "prev_high": float(prev.get('high', 0)),
                            "prev_low":  float(prev.get('low', 0)),
                            "change_pct": (close / prev_close - 1) * 100 if prev_close > 0 else 0.0,
                        }
        except Exception:
            pass

        # 3) yfinance 폴백
        for suffix in suffixes:
            try:
                ticker = yf.Ticker(f"{code}{suffix}")
                raw_d = self._history_with_retry(ticker, self.yf_start_daily)
                if raw_d is None or len(raw_d) < 60:
                    continue
                raw_d.columns = [str(c).lower() for c in raw_d.columns]
                if "close" not in raw_d.columns:
                    continue

                daily = pd.DataFrame(index=raw_d.index)
                daily["close"]  = raw_d["close"].astype(float)
                daily["open"]   = raw_d.get("open",   raw_d["close"]).astype(float)
                daily["high"]   = raw_d.get("high",   raw_d["close"]).astype(float)
                daily["low"]    = raw_d.get("low",    raw_d["close"]).astype(float)
                daily["volume"] = raw_d.get("volume", pd.Series(0.0, index=raw_d.index)).astype(float)
                daily = daily.dropna(subset=["close"])
                if len(daily) < 60:
                    continue

                weekly = None
                try:
                    raw_w = self._history_with_retry(ticker, self.yf_start_weekly, interval="1wk")
                    if raw_w is not None and len(raw_w) >= 10:
                        raw_w.columns = [str(c).lower() for c in raw_w.columns]
                        if "close" in raw_w.columns:
                            weekly = pd.DataFrame(index=raw_w.index)
                            weekly["close"]  = raw_w["close"].astype(float)
                            weekly["open"]   = raw_w.get("open",  raw_w["close"]).astype(float)
                            weekly["high"]   = raw_w.get("high",  raw_w["close"]).astype(float)
                            weekly["low"]    = raw_w.get("low",   raw_w["close"]).astype(float)
                            weekly["volume"] = raw_w.get("volume", pd.Series(0.0, index=raw_w.index)).astype(float)
                            weekly = weekly.dropna(subset=["close"])
                except Exception:
                    weekly = None

                monthly = None
                try:
                    raw_m = self._history_with_retry(ticker, self.yf_start_monthly, interval="1mo")
                    if raw_m is not None and len(raw_m) >= 6:
                        raw_m.columns = [str(c).lower() for c in raw_m.columns]
                        if "close" in raw_m.columns:
                            monthly = pd.DataFrame(index=raw_m.index)
                            monthly["close"]  = raw_m["close"].astype(float)
                            monthly["open"]   = raw_m.get("open",  raw_m["close"]).astype(float)
                            monthly["high"]   = raw_m.get("high",  raw_m["close"]).astype(float)
                            monthly["low"]    = raw_m.get("low",   raw_m["close"]).astype(float)
                            monthly["volume"] = raw_m.get("volume", pd.Series(0.0, index=raw_m.index)).astype(float)
                            monthly = monthly.dropna(subset=["close"])
                except Exception:
                    monthly = None

                # ★ 시가총액 yfinance
                cap = 0.0
                try:
                    info = ticker.info
                    mc = info.get("marketCap")
                    if mc and float(mc) > 0:
                        cap = float(mc)
                except Exception:
                    pass

                try:
                    self._save_cache(code, daily, weekly, monthly)
                except Exception:
                    pass

                last = daily.iloc[-1]; prev = daily.iloc[-2] if len(daily) > 1 else daily.iloc[-1]
                close = float(last.get('close', 0)); vol = float(last.get('volume', 0))
                prev_close = float(prev.get('close', 0))
                if close <= 0:
                    continue

                name_map = getattr(self, '_ticker_name_map', {})
                name = name_map.get(code, code)
                stock_market = market_map.get(code, "KOSPI")

                return {
                    "code": code, "name": name, "market": stock_market,
                    "current_price": close,
                    "open": float(last.get('open', 0)),
                    "high": float(last.get('high', 0)),
                    "low":  float(last.get('low', 0)),
                    "volume": vol, "volume_bil": vol * close / 1e8,
                    "market_cap": cap,
                    "ohlcv": daily, "ohlcv_weekly": weekly, "ohlcv_monthly": monthly,
                    "prev_close": prev_close,
                    "prev_high": float(prev.get('high', 0)),
                    "prev_low":  float(prev.get('low', 0)),
                    "change_pct": (close / prev_close - 1) * 100 if prev_close > 0 else 0.0,
                }

            except Exception:
                continue

        try:
            self._log_failed(code)
        except Exception:
            pass
        return None

    # ── 병렬 수집 (업로드 파일 개선 + v7.2 필터) ─────────────────────────
    def fetch_all_parallel(self, max_workers=4, per_task_timeout=30):
        tickers = self._get_filtered_tickers()
        if not tickers:
            print("[ERROR] 종목 리스트 조회 실패")
            return self._generate_demo_data()

        print(f"[수집] {len(tickers)}개 종목 병렬 수집 중... (workers={max_workers})")
        records = []

        from concurrent.futures import wait, FIRST_COMPLETED

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_code = {executor.submit(self._fetch_single, code): code for code in tickers}
            start_times = {f: time.time() for f in future_to_code}
            pending = set(future_to_code.keys())
            completed = 0

            while pending:
                done, not_done = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for f in list(done):
                    code = future_to_code.get(f)
                    try:
                        res = f.result(timeout=1)
                        if res and res.get('code') == code:
                            records.append(res)
                    except Exception:
                        pass
                    pending.discard(f)
                    completed += 1
                    if completed % 30 == 0:
                        print(f"   가격 다운로드: {completed}/{len(tickers)}")

                now = time.time()
                for f in list(pending):
                    if now - start_times.get(f, now) > per_task_timeout:
                        code = future_to_code.get(f)
                        try:
                            f.cancel()
                        except Exception:
                            pass
                        pending.discard(f)
                        completed += 1

        if not records:
            return self._generate_demo_data()

        df = pd.DataFrame(records)
        df = df[df["current_price"] > 0].reset_index(drop=True)

        if "name" in df.columns:
            def _fix_name(row):
                n = row["name"]
                if isinstance(n, (pd.DataFrame, list)): return str(row.get("code", "-"))
                if not isinstance(n, str) or n.strip() in ("", "nan"): return str(row.get("code", "-"))
                return n
            df["name"] = df.apply(_fix_name, axis=1)

        # ★ v7.2: 시가총액/거래대금 최종 필터
        if "market_cap" in df.columns and self.min_market_cap > 0:
            before = len(df)
            if "volume_bil" in df.columns:
                mask = (
                    (df["market_cap"] >= self.min_market_cap) |
                    ((df["market_cap"] == 0) & (df["volume_bil"] >= self.min_volume_bil))
                )
                df_filtered = df[mask].reset_index(drop=True)
            else:
                df_filtered = df[
                    (df["market_cap"] >= self.min_market_cap) |
                    (df["market_cap"] == 0)
                ].reset_index(drop=True)
            after = len(df_filtered)
            if before != after:
                print(f"  [최종필터] {before-after}개 시가총액 미달 종목 제거")
            # ★ v7.2: 안전장치 - 필터 후 너무 적으면 원본 유지
            if after >= 10:
                df = df_filtered
            else:
                print(f"  [WARN] 필터 후 {after}개 너무 적음 → 원본 {before}개 유지")

        return df

    def _default_tickers(self):
        return ["005930","000660","035720","005380","051910",
                "006400","035420","207940","068270","028260"]

    def _generate_demo_data(self):
        np.random.seed(42)
        names = ["삼성전자","SK하이닉스","카카오","현대차","LG에너지솔루션",
                 "삼성SDI","NAVER","삼성바이오로직스","셀트리온","POSCO홀딩스",
                 "에코프로","HLB","알테오젠","포스코퓨처엠","엘앤에프",
                 "레인보우로보틱스","리가켐바이오","클래시스","파마리서치","휴젤"]
        codes = ["005930","000660","035720","005380","051910",
                 "006400","035420","207940","068270","005490",
                 "247540","028300","196170","003670","066970",
                 "277810","343510","214150","214450","145020"]
        rows = []
        for i, code in enumerate(codes[:len(names)]):
            price  = np.random.randint(10000, 500000)
            dates  = pd.date_range(end=datetime.now(), periods=250, freq="B")
            prices = price * np.cumprod(1 + np.random.randn(250) * 0.015)
            ohlcv  = pd.DataFrame({"close": prices,
                "open":   prices * (1 + np.random.randn(250) * 0.005),
                "high":   prices * (1 + np.abs(np.random.randn(250)) * 0.01),
                "low":    prices * (1 - np.abs(np.random.randn(250)) * 0.01),
                "volume": np.random.randint(100000, 2000000, 250).astype(float)}, index=dates)
            wdates  = pd.date_range(end=datetime.now(), periods=104, freq="W")
            wprices = price * np.cumprod(1 + np.random.randn(104) * 0.03)
            weekly  = pd.DataFrame({"close": wprices,
                "open":   wprices * (1 + np.random.randn(104) * 0.01),
                "high":   wprices * (1 + np.abs(np.random.randn(104)) * 0.02),
                "low":    wprices * (1 - np.abs(np.random.randn(104)) * 0.02),
                "volume": np.random.randint(500000, 10000000, 104).astype(float)}, index=wdates)
            mdates  = pd.date_range(end=datetime.now(), periods=60, freq="ME")
            mprices = price * np.cumprod(1 + np.random.randn(60) * 0.05)
            monthly = pd.DataFrame({"close": mprices,
                "open":   mprices * (1 + np.random.randn(60) * 0.02),
                "high":   mprices * (1 + np.abs(np.random.randn(60)) * 0.04),
                "low":    mprices * (1 - np.abs(np.random.randn(60)) * 0.04),
                "volume": np.random.randint(2000000, 50000000, 60).astype(float)}, index=mdates)
            vol = float(np.random.randint(100000, 2000000))
            mkt = "KOSPI" if i < 10 else "KOSDAQ"
            rows.append({
                "code": code, "name": names[i], "market": mkt,
                "current_price": float(prices[-1]),
                "open": float(prices[-1] * 0.998), "high": float(prices[-1] * 1.015),
                "low": float(prices[-1] * 0.985), "volume": vol,
                "volume_bil": float(prices[-1]) * vol / 1e8,
                "market_cap": float(price * 1e7 * np.random.uniform(10, 500)),
                "ohlcv": ohlcv, "ohlcv_weekly": weekly, "ohlcv_monthly": monthly,
                "prev_close": float(prices[-2]), "prev_high": float(prices[-2] * 1.01),
                "prev_low": float(prices[-2] * 0.99),
                "change_pct": float((prices[-1] / prices[-2] - 1) * 100),
                "inst_net": int(np.random.randint(-500000, 500000)),
                "foreign_net": int(np.random.randint(-1000000, 1000000)),
            })
        return pd.DataFrame(rows)