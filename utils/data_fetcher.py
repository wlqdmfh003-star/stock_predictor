import pandas as pd
import numpy as np
import yfinance as yf
from pykrx import stock as krx
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
                    print(f"✅ pykrx {mkt} 종목리스트: {len(t)}개")
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
                print(f"✅ 네이버 {mkt} 종목리스트: {len(t_naver)}개")
            except Exception as e:
                print(f"⚠️ {mkt} 종목리스트 수집 실패: {e}")

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

            print(f"📊 사전 스코어링 중... (KOSPI {len(kospi_pool)}개 / KOSDAQ {len(kosdaq_pool)}개)")

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

    # ── 일봉+주봉+월봉 수집 (트리플 타임프레임) ─────────────────────────────
    def _fetch_single(self, code):
        market_map = getattr(self, '_ticker_market_map', {})
        mkt        = market_map.get(code, "")
        suffixes   = [".KS",".KQ"] if mkt!="KOSDAQ" else [".KQ",".KS"]

        for suffix in suffixes:
            try:
                ticker = yf.Ticker(f"{code}{suffix}")

                # ① 일봉 (1년)
                raw_d = ticker.history(start=self.yf_start_daily, auto_adjust=True)
                if raw_d is None or len(raw_d) < 60: continue
                raw_d.columns = [str(c).lower() for c in raw_d.columns]
                if "close" not in raw_d.columns: continue

                daily = pd.DataFrame(index=raw_d.index)
                daily["close"]  = raw_d["close"].astype(float)
                daily["open"]   = raw_d.get("open",   raw_d["close"]).astype(float)
                daily["high"]   = raw_d.get("high",   raw_d["close"]).astype(float)
                daily["low"]    = raw_d.get("low",    raw_d["close"]).astype(float)
                daily["volume"] = raw_d.get("volume", pd.Series(0.0,index=raw_d.index)).astype(float)
                daily = daily.dropna(subset=["close"])
                if len(daily) < 60: continue

                # ② 주봉 (2년)
                weekly = None
                try:
                    raw_w = ticker.history(start=self.yf_start_weekly, interval="1wk", auto_adjust=True)
                    if raw_w is not None and len(raw_w) >= 10:
                        raw_w.columns = [str(c).lower() for c in raw_w.columns]
                        if "close" in raw_w.columns:
                            weekly = pd.DataFrame(index=raw_w.index)
                            weekly["close"]  = raw_w["close"].astype(float)
                            weekly["open"]   = raw_w.get("open",  raw_w["close"]).astype(float)
                            weekly["high"]   = raw_w.get("high",  raw_w["close"]).astype(float)
                            weekly["low"]    = raw_w.get("low",   raw_w["close"]).astype(float)
                            weekly["volume"] = raw_w.get("volume",pd.Series(0.0,index=raw_w.index)).astype(float)
                            weekly = weekly.dropna(subset=["close"])
                except Exception:
                    weekly = None

                # ③ 월봉 (5년) ← 신규
                monthly = None
                try:
                    raw_m = ticker.history(start=self.yf_start_monthly, interval="1mo", auto_adjust=True)
                    if raw_m is not None and len(raw_m) >= 6:
                        raw_m.columns = [str(c).lower() for c in raw_m.columns]
                        if "close" in raw_m.columns:
                            monthly = pd.DataFrame(index=raw_m.index)
                            monthly["close"]  = raw_m["close"].astype(float)
                            monthly["open"]   = raw_m.get("open",  raw_m["close"]).astype(float)
                            monthly["high"]   = raw_m.get("high",  raw_m["close"]).astype(float)
                            monthly["low"]    = raw_m.get("low",   raw_m["close"]).astype(float)
                            monthly["volume"] = raw_m.get("volume",pd.Series(0.0,index=raw_m.index)).astype(float)
                            monthly = monthly.dropna(subset=["close"])
                except Exception:
                    monthly = None

                return {"code":code, "ohlcv":daily, "ohlcv_weekly":weekly, "ohlcv_monthly":monthly}

            except Exception:
                continue
        return None

    # ── 병렬 수집 ────────────────────────────────────────────────────────────
    def fetch_all_parallel(self):
        tickers = self._get_filtered_tickers()
        records = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_code = {executor.submit(self._fetch_single,code):code for code in tickers}
            done = 0
            for future in as_completed(future_to_code):
                code   = future_to_code[future]
                result = future.result()
                if result and result["code"]==code:
                    records.append(result)
                done += 1
                if done%30==0: print(f"   가격 다운로드: {done}/{len(tickers)}")

        if not records:
            return self._generate_demo_data()

        rows = []
        for rec in records:
            code    = rec["code"]
            ohlcv   = rec["ohlcv"]
            weekly  = rec.get("ohlcv_weekly")
            monthly = rec.get("ohlcv_monthly")
            try:
                last = ohlcv.iloc[-1]; prev = ohlcv.iloc[-2] if len(ohlcv)>1 else ohlcv.iloc[-1]
                name_map = getattr(self,'_ticker_name_map',{})
                if code in name_map and name_map[code]:
                    name = name_map[code]
                else:
                    try:
                        n = krx.get_market_ticker_name(code)
                        name = str(n).strip() if n and str(n).strip() else code
                    except: name = code

                try:
                    mc  = krx.get_market_cap_by_date(self.today,self.today,code)
                    cap = float(mc["시가총액"].iloc[-1]) if len(mc)>0 else 0.0
                except: cap = 0.0

                market_map   = getattr(self,'_ticker_market_map',{})
                stock_market = market_map.get(code,"KOSPI")
                close = float(last["close"]); vol = float(last["volume"])
                prev_close = float(prev["close"])
                if close<=0: continue

                rows.append({
                    "code":code,"name":name,"market":stock_market,
                    "current_price":close,
                    "open":float(last["open"]),"high":float(last["high"]),"low":float(last["low"]),
                    "volume":vol,"volume_bil":vol*close/1e8,"market_cap":cap,
                    "ohlcv":ohlcv,"ohlcv_weekly":weekly,"ohlcv_monthly":monthly,
                    "prev_close":prev_close,"prev_high":float(prev["high"]),"prev_low":float(prev["low"]),
                    "change_pct":(close/prev_close-1)*100 if prev_close>0 else 0.0,
                })
            except Exception: continue

        if not rows: return self._generate_demo_data()

        df = pd.DataFrame(rows)
        df = df[df["current_price"]>0].reset_index(drop=True)

        if "name" in df.columns:
            def _fix_name(row):
                n = row["name"]
                if isinstance(n,(pd.DataFrame,list)): return str(row.get("code","-"))
                if not isinstance(n,str) or n.strip() in ("","nan"): return str(row.get("code","-"))
                return n
            df["name"] = df.apply(_fix_name, axis=1)

        return df

    # ── 기관수급 ─────────────────────────────────────────────────────────────
    def fetch_institution_data(self, df):
        inst_data = {}
        start_5d  = (datetime.now()-timedelta(days=10)).strftime("%Y%m%d")
        print("🏛️ 기관/외인 수급 수집 중...")
        total = len(df)
        for i, code in enumerate(df["code"].tolist()):
            try:
                inv = krx.get_market_trading_value_by_date(start_5d,self.today,code)
                if inv is not None and len(inv)>0:
                    inst_net    = inv["기관합계"].sum()   if "기관합계"   in inv.columns else 0
                    foreign_net = inv["외국인합계"].sum() if "외국인합계" in inv.columns else 0
                else: inst_net, foreign_net = 0, 0
                inst_data[code] = {"inst_net":inst_net,"foreign_net":foreign_net}
            except Exception:
                inst_data[code] = {"inst_net":0,"foreign_net":0}
            if (i+1)%20==0: print(f"   기관수급 진행: {i+1}/{total}")
            time.sleep(0.05)
        df = df.copy()
        df["inst_net"]    = df["code"].map(lambda c:inst_data.get(c,{}).get("inst_net",0))
        df["foreign_net"] = df["code"].map(lambda c:inst_data.get(c,{}).get("foreign_net",0))
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
        for i,code in enumerate(codes[:len(names)]):
            price  = np.random.randint(10000,500000)
            dates  = pd.date_range(end=datetime.now(),periods=250,freq="B")
            prices = price * np.cumprod(1+np.random.randn(250)*0.015)
            ohlcv  = pd.DataFrame({"close":prices,
                "open":prices*(1+np.random.randn(250)*0.005),
                "high":prices*(1+np.abs(np.random.randn(250))*0.01),
                "low": prices*(1-np.abs(np.random.randn(250))*0.01),
                "volume":np.random.randint(100000,2000000,250).astype(float)},index=dates)
            wdates  = pd.date_range(end=datetime.now(),periods=104,freq="W")
            wprices = price * np.cumprod(1+np.random.randn(104)*0.03)
            weekly  = pd.DataFrame({"close":wprices,
                "open":wprices*(1+np.random.randn(104)*0.01),
                "high":wprices*(1+np.abs(np.random.randn(104))*0.02),
                "low": wprices*(1-np.abs(np.random.randn(104))*0.02),
                "volume":np.random.randint(500000,10000000,104).astype(float)},index=wdates)
            mdates  = pd.date_range(end=datetime.now(),periods=60,freq="ME")
            mprices = price * np.cumprod(1+np.random.randn(60)*0.05)
            monthly = pd.DataFrame({"close":mprices,
                "open":mprices*(1+np.random.randn(60)*0.02),
                "high":mprices*(1+np.abs(np.random.randn(60))*0.04),
                "low": mprices*(1-np.abs(np.random.randn(60))*0.04),
                "volume":np.random.randint(2000000,50000000,60).astype(float)},index=mdates)
            vol = float(np.random.randint(100000,2000000))
            mkt = "KOSPI" if i<10 else "KOSDAQ"
            rows.append({
                "code":code,"name":names[i],"market":mkt,
                "current_price":float(prices[-1]),
                "open":float(prices[-1]*0.998),"high":float(prices[-1]*1.015),
                "low":float(prices[-1]*0.985),"volume":vol,
                "volume_bil":float(prices[-1])*vol/1e8,
                "market_cap":float(price*1e7*np.random.uniform(10,500)),
                "ohlcv":ohlcv,"ohlcv_weekly":weekly,"ohlcv_monthly":monthly,
                "prev_close":float(prices[-2]),"prev_high":float(prices[-2]*1.01),
                "prev_low":float(prices[-2]*0.99),
                "change_pct":float((prices[-1]/prices[-2]-1)*100),
                "inst_net":int(np.random.randint(-500000,500000)),
                "foreign_net":int(np.random.randint(-1000000,1000000)),
            })
        return pd.DataFrame(rows)