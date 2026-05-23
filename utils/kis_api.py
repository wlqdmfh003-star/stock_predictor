import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class KISApi:
    """
    한국투자증권 Open API v5.4
    ★ 호가 실시간 완전 활용 (10호가 압력 + 매수벽 감지)
    ★ 체결 강도 (매수/매도 체결 비율)
    ★ 프로그램 매매 동향 (차익/비차익)
    ★ 외국인 실시간 순매수
    ★ 장중 실시간 변동성 감지
    """

    REAL_URL = "https://openapi.koreainvestment.com:9443"
    VIRT_URL = "https://openapivts.koreainvestment.com:29443"

    def __init__(self, app_key: str, app_secret: str, account: str):
        self.app_key    = str(app_key    or "").strip()
        self.app_secret = str(app_secret or "").strip()
        self.account    = str(account    or "").strip()
        self._token     = None
        self._token_exp = None
        self._connected = False
        acc = self.account.replace("-","")
        if acc.startswith("50"):
            self.BASE_URL = self.VIRT_URL; self._mode = "모의투자"
        else:
            self.BASE_URL = self.REAL_URL; self._mode = "실전투자"

    # ── 토큰 ────────────────────────────────────────────────────────────────
    def _get_token(self) -> str:
        if not self.app_key or not self.app_secret:
            raise ValueError("KIS 앱키/시크릿키 없음")
        now = datetime.now()
        if self._token and self._token_exp and now < self._token_exp:
            return self._token
        resp = requests.post(f"{self.BASE_URL}/oauth2/tokenP",
                             json={"grant_type":"client_credentials",
                                   "appkey":self.app_key,"appsecret":self.app_secret},
                             timeout=10)
        data  = resp.json()
        token = data.get("access_token","")
        if not token:
            raise ConnectionError(f"KIS 토큰 발급 실패: {data.get('msg1','')}")
        self._token     = token
        self._token_exp = now + timedelta(hours=23)
        self._connected = True
        return self._token

    def _h(self, tr_id: str) -> dict:
        return {"content-type":"application/json; charset=utf-8",
                "authorization":f"Bearer {self._get_token()}",
                "appkey":self.app_key,"appsecret":self.app_secret,
                "tr_id":tr_id,"custtype":"P"}

    def _get(self, path, tr_id, params, timeout=5):
        try:
            return requests.get(f"{self.BASE_URL}{path}",
                                headers=self._h(tr_id),
                                params=params, timeout=timeout).json()
        except: return {}

    # ── 연결 테스트 ─────────────────────────────────────────────────────────
    def test_connection(self) -> dict:
        if not self.app_key or not self.app_secret:
            return {"ok":False,"mode":self._mode,"message":"앱키/시크릿키 미입력"}
        try:
            r = self.get_price("005930")
            if r.get("current_price",0) > 0:
                return {"ok":True,"mode":self._mode,
                        "message":f"[OK] KIS {self._mode} 연결 성공! 삼성전자: {r['current_price']:,}원"}
            return {"ok":False,"mode":self._mode,"message":"데이터 조회 실패 (장 마감 확인)"}
        except ConnectionError as e:
            return {"ok":False,"mode":self._mode,"message":str(e)}
        except Exception as e:
            return {"ok":False,"mode":self._mode,"message":f"오류: {str(e)[:80]}"}

    # ── 현재가 ──────────────────────────────────────────────────────────────
    def get_price(self, code: str) -> dict:
        d = self._get("/uapi/domestic-stock/v1/quotations/inquire-price",
                      "FHKST01010100",
                      {"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":code})
        out = d.get("output",{})
        return {
            "current_price": int(out.get("stck_prpr",0) or 0),
            "open":          int(out.get("stck_oprc",0) or 0),
            "high":          int(out.get("stck_hgpr",0) or 0),
            "low":           int(out.get("stck_lwpr",0) or 0),
            "volume":        int(out.get("acml_vol",0)  or 0),
            "change_pct":    float(out.get("prdy_ctrt",0) or 0),
            "prev_close":    int(out.get("stck_sdpr",0) or 0),
            "market_cap":    float(out.get("hts_avls",0) or 0)*1e8,
            "per":           float(out.get("per",0) or 0),
            "pbr":           float(out.get("pbr",0) or 0),
        }

    # ── ★ 호가 완전 분석 ────────────────────────────────────────────────────
    def get_orderbook(self, code: str) -> dict:
        """
        10호가 완전 분석
        ─ 매수/매도 압력 비율
        ─ 매수벽 감지 (특정 호가에 대량 대기)
        ─ 호가 기울기 (상위 호가 잔량 비율)
        ─ 스프레드 분석
        ─ ob_score: 0~100 (높을수록 매수 우세)
        """
        d   = self._get("/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
                        "FHKST01010200",
                        {"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":code})
        out = d.get("output1",{})
        if not out:
            return self._def_ob()

        buy_p  = [int(out.get(f"bidp{i}",0)      or 0) for i in range(1,11)]
        sell_p = [int(out.get(f"askp{i}",0)      or 0) for i in range(1,11)]
        buy_v  = [int(out.get(f"bidp_rsqn{i}",0) or 0) for i in range(1,11)]
        sell_v = [int(out.get(f"askp_rsqn{i}",0) or 0) for i in range(1,11)]

        bt = sum(buy_v); st = sum(sell_v)
        pressure = round(bt/(st+1e-9), 3)

        # 매수벽 감지: 상위 3호가 대비 4~10호가 비율
        top3_buy  = sum(buy_v[:3]);  rest_buy  = sum(buy_v[3:])
        top3_sell = sum(sell_v[:3]); rest_sell = sum(sell_v[3:])
        buy_wall  = top3_buy/(rest_buy+1e-9)   # >2: 강한 매수벽
        sell_wall = top3_sell/(rest_sell+1e-9)  # >2: 강한 매도벽

        # 스프레드
        spread     = sell_p[0]-buy_p[0] if sell_p[0]>0 else 0
        spread_pct = round(spread/(buy_p[0]+1e-9)*100, 3)

        # 호가 잔량 기울기 (상위로 갈수록 줄면 매도 압박)
        buy_slope  = float(np.polyfit(range(len(buy_v)),  buy_v,  1)[0]) if bt>0 else 0
        sell_slope = float(np.polyfit(range(len(sell_v)), sell_v, 1)[0]) if st>0 else 0

        # ob_score 계산
        score = 50.0
        if   pressure>1.5:   score+=25
        elif pressure>1.2:   score+=15
        elif pressure>1.0:   score+=8
        elif pressure<0.7:   score-=20
        elif pressure<0.9:   score-=10
        if buy_wall>2.0:     score+=10   # 매수벽
        if sell_wall>2.0:    score-=10   # 매도벽
        if spread_pct<0.1:   score+=5
        elif spread_pct>0.5: score-=5
        if buy_slope>0:      score+=5    # 상위호가 매수잔량 증가
        if sell_slope<0:     score+=3    # 상위호가 매도잔량 감소

        return {
            "buy_total":   bt, "sell_total":  st,
            "ob_pressure": pressure,
            "buy_wall":    round(buy_wall,2), "sell_wall": round(sell_wall,2),
            "spread_pct":  spread_pct,
            "buy_slope":   round(buy_slope,1),"sell_slope":round(sell_slope,1),
            "buy_prices":  buy_p, "sell_prices": sell_p,
            "buy_vols":    buy_v, "sell_vols":   sell_v,
            "ob_score":    float(np.clip(score,0,100)),
        }

    def _def_ob(self):
        return {"buy_total":0,"sell_total":0,"ob_pressure":1.0,
                "buy_wall":1.0,"sell_wall":1.0,"spread_pct":0.0,
                "buy_slope":0,"sell_slope":0,
                "buy_prices":[0]*10,"sell_prices":[0]*10,
                "buy_vols":[0]*10,"sell_vols":[0]*10,"ob_score":50.0}

    # ── ★ 체결 강도 ─────────────────────────────────────────────────────────
    def get_trade_strength(self, code: str) -> dict:
        """
        체결 강도 분석
        ─ 매수체결량 / 매도체결량 * 100
        ─ 100 이상 = 매수 우세
        ─ 연속 상승 체결 감지
        """
        d   = self._get("/uapi/domestic-stock/v1/quotations/inquire-price",
                        "FHKST01010100",
                        {"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":code})
        out = d.get("output",{})
        buy_vol  = float(out.get("shnu_cntg_smtn",0) or 0)
        sell_vol = float(out.get("seln_cntg_smtn",1) or 1)
        ts       = round(buy_vol/(sell_vol+1e-9)*100, 1)
        # 순간 체결 강도 → 분봉 체결 강도
        strength_score = float(np.clip(50+(ts-100)*0.3, 0, 100))

        # 상승 체결 감지 (연속 양봉 체결)
        up_cnt   = int(out.get("seln_cntg_smtn",0) or 0)
        momentum = "매수우세" if ts>110 else "매도우세" if ts<90 else "균형"

        return {"trade_strength":ts,"ts_score":strength_score,
                "momentum":momentum,"up_count":up_cnt}

    # ── ★ 프로그램 매매 동향 ────────────────────────────────────────────────
    def get_program_trade(self, code: str) -> dict:
        """
        프로그램 매매 (차익/비차익) 동향
        기관 알고리즘 매매 방향 파악
        """
        d   = self._get("/uapi/domestic-stock/v1/quotations/inquire-investor",
                        "FHKST01010900",
                        {"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":code,
                         "FID_INPUT_DATE_1":(datetime.now()-timedelta(days=3)).strftime("%Y%m%d"),
                         "FID_INPUT_DATE_2":datetime.now().strftime("%Y%m%d"),
                         "FID_ETC_CLS_CODE":""})
        output = d.get("output",[])
        if not output:
            return {"prog_net":0,"prog_score":50}
        prog_net = sum(int(item.get("prsm_mndi_smtn",0) or 0) for item in output)
        prog_score = float(np.clip(50+prog_net/1e7, 0, 100))
        return {"prog_net":prog_net,"prog_score":prog_score}

    # ── 기관/외인 수급 ──────────────────────────────────────────────────────
    def get_investor_trend(self, code: str, days: int=10) -> dict:
        """
        기관/외국인 수급 + 외국인 누적 추이 분석
        ★ 5일 연속 외국인 순매수 = 강한 매수 신호
        ★ 외국인 누적 방향성 (증가/감소 추세)
        """
        end   = datetime.now().strftime("%Y%m%d")
        start = (datetime.now()-timedelta(days=days+5)).strftime("%Y%m%d")
        d     = self._get("/uapi/domestic-stock/v1/quotations/inquire-investor",
                          "FHKST01010900",
                          {"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":code,
                           "FID_INPUT_DATE_1":start,"FID_INPUT_DATE_2":end,
                           "FID_ETC_CLS_CODE":""})
        output = d.get("output",[])
        if not output:
            return {"inst_net":0,"foreign_net":0,"inst_score":50,
                    "foreign_consec":0,"foreign_trend":0,"foreign_trend_score":50}

        inst_net    = sum(int(i.get("orgn_ntby_qty",0)  or 0) for i in output)
        foreign_net = sum(int(i.get("frgn_ntby_qty",0)  or 0) for i in output)
        inst_amt    = sum(int(i.get("orgn_ntby_tr_pbmn",0) or 0) for i in output)
        frgn_amt    = sum(int(i.get("frgn_ntby_tr_pbmn",0) or 0) for i in output)

        # ★ 외국인 일별 순매수 추이 분석
        frgn_daily = [int(i.get("frgn_ntby_qty",0) or 0) for i in output]

        # 연속 순매수/순매도 일수
        consec = 0
        for v in reversed(frgn_daily):
            if v > 0: consec += 1
            elif v < 0:
                if consec == 0: consec = -1
                else: break
            else: break

        # 5일 추세 (선형회귀 기울기)
        if len(frgn_daily) >= 5:
            x = np.arange(len(frgn_daily[-5:]))
            trend = float(np.polyfit(x, frgn_daily[-5:], 1)[0])
        else:
            trend = 0.0

        # 외국인 추이 점수
        ft_score = 50.0
        if consec >= 5:   ft_score += 25   # 5일 연속 순매수
        elif consec >= 3: ft_score += 15
        elif consec >= 1: ft_score += 8
        elif consec <= -5: ft_score -= 25  # 5일 연속 순매도
        elif consec <= -3: ft_score -= 15
        elif consec <= -1: ft_score -= 8
        if trend > 0:     ft_score += 8
        elif trend < 0:   ft_score -= 8

        inst_score = float(np.clip(50+(inst_amt+frgn_amt*0.5)/1e7, 0, 100))
        return {
            "inst_net":           inst_net,
            "foreign_net":        foreign_net,
            "inst_amt":           inst_amt,
            "frgn_amt":           frgn_amt,
            "inst_score":         inst_score,
            "foreign_consec":     consec,       # 연속 순매수(+)/순매도(-) 일수
            "foreign_trend":      round(trend, 2),  # 추이 기울기
            "foreign_trend_score":float(np.clip(ft_score, 0, 100)),
        }

    # ── 전체 DataFrame 수급 + 호가 적용 ────────────────────────────────────
    def get_institution_data(self, df: pd.DataFrame) -> pd.DataFrame:
        conn = self.test_connection()
        if not conn["ok"]:
            print(f"[주의] KIS 연결 실패: {conn['message']} → pykrx 대체")
            return self._fallback(df)

        print(f"[KIS] {self._mode} 기관/외인 + 호가 수급 수집 중...")
        results = {}
        total   = len(df)

        for i, code in enumerate(df["code"].tolist()):
            try:
                inv = self.get_investor_trend(code)
                ob  = self.get_orderbook(code)
                ts  = self.get_trade_strength(code)
                results[code] = {**inv, **ob, **ts}
            except:
                results[code] = {"inst_net":0,"foreign_net":0,"inst_score":50,
                                 "ob_score":50,"ts_score":50,"ob_pressure":1.0,
                                 "buy_wall":1.0,"sell_wall":1.0,"trade_strength":100}
            if (i+1)%20==0:
                print(f"  [KIS] 수급 진행: {i+1}/{total}")
            time.sleep(0.06)

        df = df.copy()
        for col, key, default in [
            ("inst_net",             "inst_net",             0),
            ("foreign_net",          "foreign_net",          0),
            ("institution_score",    "inst_score",           50),
            ("ob_score",             "ob_score",             50),
            ("ts_score",             "ts_score",             50),
            ("ob_pressure",          "ob_pressure",          1.0),
            ("buy_wall",             "buy_wall",             1.0),
            ("sell_wall",            "sell_wall",            1.0),
            ("trade_strength",       "trade_strength",       100),
            ("foreign_consec",       "foreign_consec",       0),
            ("foreign_trend",        "foreign_trend",        0.0),
            ("foreign_trend_score",  "foreign_trend_score",  50),
        ]:
            df[col] = df["code"].map(lambda c: results.get(c,{}).get(key, default))

        # ★ 호가 + 체결강도로 기관점수 보완 (장외시간 차별화)
        df["institution_score"] = (
            df["institution_score"] * 0.6 +
            df["ob_score"]          * 0.25 +
            df["ts_score"].clip(0,200)/200*100 * 0.15
        ).clip(0, 100)

        # 호가 + 체결강도 종합 수급 점수
        df["orderbook_score"] = (
            df["ob_score"]  * 0.5 +
            df["ts_score"]  * 0.3 +
            df["ob_pressure"].clip(0,3)/3*100 * 0.2
        ).clip(0, 100)

        print(f"[OK] KIS 수급+호가 수집 완료 ({total}개)")
        return df

    def _fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from pykrx import stock as krx
            today    = datetime.now().strftime("%Y%m%d")
            start_5d = (datetime.now()-timedelta(days=10)).strftime("%Y%m%d")
            data = {}
            for code in df["code"].tolist():
                try:
                    inv = krx.get_market_trading_value_by_date(start_5d,today,code)
                    inst_net    = inv["기관합계"].sum()   if inv is not None and "기관합계"   in inv.columns else 0
                    foreign_net = inv["외국인합계"].sum() if inv is not None and "외국인합계" in inv.columns else 0
                    data[code]  = {"inst_net":inst_net,"foreign_net":foreign_net}
                except: data[code] = {"inst_net":0,"foreign_net":0}
                time.sleep(0.05)
            df = df.copy()
            df["inst_net"]    = df["code"].map(lambda c: data.get(c,{}).get("inst_net",0))
            df["foreign_net"] = df["code"].map(lambda c: data.get(c,{}).get("foreign_net",0))
        except Exception as e:
            print(f"[주의] pykrx 수급도 실패: {e}")
        return df

    # ── 장중 실시간 변동성 ──────────────────────────────────────────────────
    def get_intraday_volatility(self, code: str) -> dict:
        """장중 고저 범위 기반 실시간 변동성"""
        price = self.get_price(code)
        if not price or price.get("current_price",0)==0:
            return {"intraday_vol":0,"vol_score":50}
        hi = price.get("high",0); lo = price.get("low",0); cl = price.get("current_price",1)
        if hi>0 and lo>0:
            rng_pct = (hi-lo)/cl*100
            pos_pct = (cl-lo)/(hi-lo+1e-9)*100  # 장중 위치
            vol_score = float(np.clip(50+(pos_pct-50)*0.5, 0, 100))
        else:
            rng_pct = 0; pos_pct = 50; vol_score = 50
        return {"intraday_vol":round(rng_pct,2),
                "intraday_pos":round(pos_pct,2),
                "vol_score":   round(vol_score,2)}

    # ── KIS 일봉 ───────────────────────────────────────────────────────────
    def get_ohlcv(self, code: str, days: int=365) -> pd.DataFrame:
        end   = datetime.now().strftime("%Y%m%d")
        start = (datetime.now()-timedelta(days=days)).strftime("%Y%m%d")
        d     = self._get("/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                          "FHKST03010100",
                          {"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":code,
                           "FID_INPUT_DATE_1":start,"FID_INPUT_DATE_2":end,
                           "FID_PERIOD_DIV_CODE":"D","FID_ORG_ADJ_PRC":"0"}, timeout=10)
        rows = []
        for item in d.get("output2",[]):
            try:
                rows.append({"date":item.get("stck_bsop_date",""),
                             "close":float(item.get("stck_clpr",0) or 0),
                             "open": float(item.get("stck_oprc",0) or 0),
                             "high": float(item.get("stck_hgpr",0) or 0),
                             "low":  float(item.get("stck_lwpr",0) or 0),
                             "volume":float(item.get("acml_vol",0) or 0)})
            except: continue
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        return df.set_index("date").sort_index()[lambda x: x["close"]>0]