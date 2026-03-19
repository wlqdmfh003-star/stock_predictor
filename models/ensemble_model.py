import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class EnsembleModel:
    """
    v6.0 앙상블 모델 — 피처 109개 + CatBoost 4모델 앙상블
    ─ 일봉  50개  모멘텀/이평/오실레이터/거래량/패턴/통계
    ─ 주봉  20개  중기 추세/오실레이터/거래량/패턴
    ─ 월봉  15개  장기 추세/위치/모멘텀
    ─ 고급  24개  허스트/엔트로피/ER/프랙탈/자기상관/베타/왜도·첨도
    ─ XGBoost + LightGBM (없으면 강화 룰기반)
    ─ 3중 타임프레임 교차신호 스코어
    """

    N_DAILY   = 50
    N_WEEKLY  = 20
    N_MONTHLY = 15
    N_ADV     = 24
    N_TOTAL   = 50 + 20 + 15 + 24   # 109

    def __init__(self, lstm_weight=0.25, xgb_weight=0.28,
                 lgbm_weight=0.25, cat_weight=0.22):
        # 기본 가중치 (동적 조정 전 초기값)
        self.lstm_weight_base = lstm_weight
        self.xgb_weight_base  = xgb_weight
        self.lgbm_weight_base = lgbm_weight
        self.cat_weight_base  = cat_weight
        self.lstm_weight = lstm_weight
        self.xgb_weight  = xgb_weight
        self.lgbm_weight = lgbm_weight
        self.cat_weight  = cat_weight
        self.has_xgb = self.has_lgbm = self.has_cat = False
        try:
            import xgboost as xgb
            self._xgb = xgb; self.has_xgb = True
            print("✅ XGBoost 사용 가능")
        except ImportError:
            print("[주의] XGBoost 없음 → 룰기반 대체")
        try:
            import lightgbm as lgb
            self._lgb = lgb; self.has_lgbm = True
            print("✅ LightGBM 사용 가능")
        except ImportError:
            print("[주의] LightGBM 없음 → 룰기반 대체")
        try:
            import catboost as cb
            self._cb = cb; self.has_cat = True
            print("✅ CatBoost 사용 가능")
        except ImportError:
            print("[주의] CatBoost 없음 (pip install catboost)")
        self.xgb_model = self.lgbm_model = self.cat_model = None
        self._trained  = False
        self._last_adv = None
        # 분봉 피처 저장소
        self._minute_features = {}
        # SHAP 설명기
        self._shap_explainer  = None

    # ══════════════════════════════════════════════════════════════════════════
    # 피처 추출 — 일봉 50개
    # ══════════════════════════════════════════════════════════════════════════
    def _extract_daily_features(self, ohlcv) -> np.ndarray:
        try:
            c = ohlcv["close"].astype(float).values
            o = ohlcv["open"].astype(float).values   if "open"   in ohlcv.columns else c.copy()
            h = ohlcv["high"].astype(float).values   if "high"   in ohlcv.columns else c.copy()
            l = ohlcv["low"].astype(float).values    if "low"    in ohlcv.columns else c.copy()
            v = ohlcv["volume"].astype(float).values if "volume" in ohlcv.columns else np.ones(len(c))
        except:
            return np.zeros(self.N_DAILY, dtype=np.float32)

        def safe(arr, idx, default=0.0):
            return float(arr[idx]) if len(arr)>abs(idx) else float(default)

        def ma(n): return c[-n:].mean() if len(c)>=n else c.mean()
        def mom(n): return (c[-1]/c[-n]-1)*100 if len(c)>=n and c[-n]>0 else 0.0

        # ── 모멘텀 5개 [0-4] ─────────────────────────────────────────────────
        f0  = mom(6)    # 5일 모멘텀
        f1  = mom(11)   # 10일 모멘텀
        f2  = mom(21)   # 20일 모멘텀
        f3  = mom(61)   # 60일 모멘텀
        f4  = mom(121)  # 120일 모멘텀

        # ── 이동평균 7개 [5-11] ───────────────────────────────────────────────
        ma5=ma(5); ma20=ma(20); ma60=ma(60); ma120=ma(120)
        f5  = (c[-1]/ma5  -1)*100 if ma5>0  else 0.0
        f6  = (c[-1]/ma20 -1)*100 if ma20>0 else 0.0
        f7  = (c[-1]/ma60 -1)*100 if ma60>0 else 0.0
        f8  = (c[-1]/ma120-1)*100 if ma120>0 else 0.0
        f9  = 1.0 if ma5>ma20>ma60  else (-1.0 if ma5<ma20<ma60  else 0.0)   # 단기정배열
        f10 = 1.0 if ma20>ma60>ma120 else (-1.0 if ma20<ma60<ma120 else 0.0)  # 중기정배열
        f11 = 1.0 if ma5>ma20 and ma20>ma60 and ma60>ma120 else \
              (-1.0 if ma5<ma20 and ma20<ma60 and ma60<ma120 else 0.0)         # 완전정배열

        # ── RSI 3개 [12-14] ───────────────────────────────────────────────────
        def rsi(n=14):
            if len(c)<n+1: return 50.0
            d=np.diff(c[-(n+1):]); g=np.where(d>0,d,0).mean(); lo=np.where(d<0,-d,0).mean()
            return float(100-100/(1+g/(lo+1e-9)))
        f12 = rsi(14); f13 = rsi(9); f14 = rsi(21)

        # ── 볼린저밴드 3개 [15-17] ────────────────────────────────────────────
        if len(c)>=20:
            bm=c[-20:].mean(); bs=c[-20:].std()+1e-9
            f15 = float((c[-1]-bm)/(bs*2))           # %B
            f16 = float(bs*4/bm*100)                  # 밴드폭%
            f17 = float((c[-1]-(bm-2*bs))/(4*bs+1e-9)) # 위치0~1
        else:
            f15=f16=f17=0.0

        # ── 스토캐스틱 2개 [18-19] ────────────────────────────────────────────
        if len(c)>=14 and len(h)>=14 and len(l)>=14:
            lo14=l[-14:].min(); hi14=h[-14:].max()
            k=float((c[-1]-lo14)/(hi14-lo14+1e-9)*100)
            d_stoch=float(np.mean([(c[-i]-l[-(14+i-1):(-(i-1)) if i>1 else None].min() if len(l)>=14+i-1 else 0)/
                                   (h[-(14+i-1):(-(i-1)) if i>1 else None].max()-l[-(14+i-1):(-(i-1)) if i>1 else None].min()+1e-9)*100
                                   for i in range(1,4)]))
            f18=k; f19=d_stoch
        else:
            f18=f19=50.0

        # ── MACD 3개 [20-22] ─────────────────────────────────────────────────
        def ema_arr(arr, n):
            if len(arr)==0: return 0.0
            if len(arr)<n:  return float(np.mean(arr))
            k=2/(n+1); e=float(arr[0])
            for x in arr: e=float(x)*k+e*(1-k)
            return e
        if len(c)>=26:
            macd_line = ema_arr(c,12) - ema_arr(c,26)
            if len(c)>=35:
                # 최근 9개 시점의 MACD 값으로 시그널 계산
                macd_pts = np.array([
                    ema_arr(c[:max(26,len(c)-8+i)],12) - ema_arr(c[:max(26,len(c)-8+i)],26)
                    for i in range(9)
                ])
                signal = ema_arr(macd_pts, 9)
            else:
                signal = macd_line
            f20=float(macd_line); f21=float(signal); f22=float(macd_line-signal)
        else:
            f20=f21=f22=0.0

        # ── Williams %R 1개 [23] ─────────────────────────────────────────────
        if len(h)>=14 and len(l)>=14:
            f23=float((h[-14:].max()-c[-1])/(h[-14:].max()-l[-14:].min()+1e-9)*-100)
        else: f23=-50.0

        # ── CCI 1개 [24] ─────────────────────────────────────────────────────
        if len(c)>=20 and len(h)>=20 and len(l)>=20:
            tp=(h[-20:]+l[-20:]+c[-20:])/3; cci_m=tp.mean(); cci_md=np.mean(np.abs(tp-cci_m))
            f24=float((tp[-1]-cci_m)/(0.015*cci_md+1e-9))
        else: f24=0.0

        # ── MFI 1개 [25] ─────────────────────────────────────────────────────
        if len(c)>=14 and len(v)>=14:
            tp14=(h[-14:]+l[-14:]+c[-14:])/3 if len(h)>=14 else c[-14:]
            mf=tp14*v[-14:]
            pos=mf[np.diff(np.concatenate([[tp14[0]],tp14]))>0].sum()
            neg=mf[np.diff(np.concatenate([[tp14[0]],tp14]))<=0].sum()
            f25=float(100-100/(1+pos/(neg+1e-9)))
        else: f25=50.0

        # ── 거래량 4개 [26-29] ────────────────────────────────────────────────
        vm5 =v[-5:].mean()  if len(v)>=5  else v.mean()
        vm20=v[-20:].mean() if len(v)>=20 else v.mean()
        f26 = float(v[-1]/(vm5 +1e-9))   # 5일 거래량비율
        f27 = float(v[-1]/(vm20+1e-9))   # 20일 거래량비율
        # 거래량 OBV
        obv_arr = np.cumsum(np.where(np.diff(np.concatenate([[c[0]],c]))>0, v, -v))
        f28 = 1.0 if len(obv_arr)>1 and obv_arr[-1]>obv_arr[-2] else -1.0
        f29 = float(np.polyfit(range(min(20,len(obv_arr))),obv_arr[-20:],1)[0]/(abs(obv_arr[-1])+1e-9)) if len(obv_arr)>=5 else 0.0
        # 거래대금 증가율
        tv = v*c  # 거래대금 근사
        f30 = float(tv[-1]/(tv[-5:].mean()+1e-9)-1) if len(tv)>=5 else 0.0

        # ── VWAP 1개 [31] ─────────────────────────────────────────────────────
        if len(v)>=20 and v.sum()>0:
            tp20=(h[-20:]+l[-20:]+c[-20:])/3 if len(h)>=20 else c[-20:]
            vwap=(tp20*v[-20:]).sum()/(v[-20:].sum()+1e-9)
            f31=float((c[-1]-vwap)/vwap*100)
        else: f31=0.0

        # ── 이격도 3개 [32-34] ────────────────────────────────────────────────
        f32=(c[-1]/ma5  -1)*100 if ma5>0  else 0.0
        f33=(c[-1]/ma20 -1)*100 if ma20>0 else 0.0
        f34=(c[-1]/ma60 -1)*100 if ma60>0 else 0.0

        # ── 피보나치 2개 [35-36] ──────────────────────────────────────────────
        if len(c)>=60 and len(h)>=60 and len(l)>=60:
            hi60=h[-60:].max(); lo60=l[-60:].min(); rng=hi60-lo60+1e-9
            f35=float((c[-1]-lo60)/rng)
            levels=[lo60+rng*r for r in [0.236,0.382,0.5,0.618,0.786]]
            dists=[abs(c[-1]-lv) for lv in levels]
            f36=float(levels[np.argmin(dists)]/c[-1]-1)*100
        else: f35=0.5; f36=0.0

        # ── ATR 2개 [37-38] ───────────────────────────────────────────────────
        if len(h)>=15 and len(l)>=15:
            tr_arr=np.array([max(h[-i]-l[-i],abs(h[-i]-c[-i-1]),abs(l[-i]-c[-i-1]))
                             for i in range(1,15)])
            atr=tr_arr.mean()
            f37=float(atr)
            f38=float(atr/c[-1]*100) if c[-1]>0 else 0.0
        else: f37=f38=0.0

        # ── 갭 2개 [39-40] ────────────────────────────────────────────────────
        f39=float((o[-1]-c[-2])/c[-2]*100) if len(c)>=2 and c[-2]>0 else 0.0
        # 갭 지속성 (갭 후 방향)
        f40=float(np.sign(f39)*((c[-1]-o[-1])/o[-1]*100)) if o[-1]>0 else 0.0

        # ── 연속상승 1개 [41] ─────────────────────────────────────────────────
        streak=0
        for i in range(1,min(15,len(c))):
            if c[-i]>c[-i-1]: streak+=1
            else: break
        f41=float(streak/10)

        # ── 변동성돌파 1개 [42] ───────────────────────────────────────────────
        if len(c)>=2 and len(h)>=2 and len(l)>=2:
            k=0.5; target=o[-1]+k*(h[-2]-l[-2])
            f42=1.0 if c[-1]>target else 0.0
        else: f42=0.0

        # ── 52주 위치 1개 [43] ────────────────────────────────────────────────
        if len(c)>=52 and len(h)>=52 and len(l)>=52:
            hi52=h[-252:].max() if len(h)>=252 else h.max()
            lo52=l[-252:].min() if len(l)>=252 else l.min()
            f43=float((c[-1]-lo52)/(hi52-lo52+1e-9))
        else: f43=0.5

        # ── 저항/지지 2개 [44-45] ─────────────────────────────────────────────
        if len(h)>=20:
            res=h[-20:].max(); sup=l[-20:].min()
            f44=1.0 if c[-1]>res*0.99 else 0.0   # 저항선 돌파
            f45=1.0 if c[-1]>sup*1.01 else 0.0   # 지지선 유지
        else: f44=f45=0.0

        # ── 캔들 2개 [46-47] ─────────────────────────────────────────────────
        body=abs(c[-1]-o[-1])/(h[-1]-l[-1]+1e-9)
        f46=float(body)
        f47=1.0 if c[-1]>o[-1] else -1.0

        # ── 고가대비 이격 + 저가대비 이격 2개 [48-49] ───────────────────────
        f48=float((c[-1]-h[-20:].max())/(h[-20:].max()+1e-9)*100) if len(h)>=20 else 0.0
        f49=float((c[-1]-l[-20:].min())/(l[-20:].min()+1e-9)*100) if len(l)>=20 else 0.0

        feats = np.array([
            f0,f1,f2,f3,f4,
            f5,f6,f7,f8,f9,f10,f11,
            f12,f13,f14,
            f15,f16,f17,
            f18,f19,
            f20,f21,f22,
            f23,f24,f25,
            f26,f27,f28,f29,f30,
            f31,
            f32,f33,f34,
            f35,f36,
            f37,f38,
            f39,f40,
            f41,f42,f43,
            f44,f45,
            f46,f47,
            f48,f49,
        ], dtype=np.float32)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # ══════════════════════════════════════════════════════════════════════════
    # 피처 추출 — 주봉 20개
    # ══════════════════════════════════════════════════════════════════════════
    def _extract_weekly_features(self, weekly) -> np.ndarray:
        z = np.zeros(self.N_WEEKLY, dtype=np.float32)
        if weekly is None or len(weekly) < 8:
            return z
        try:
            c = weekly["close"].astype(float).values
            v = weekly["volume"].astype(float).values if "volume" in weekly.columns else np.ones(len(c))
            h = weekly["high"].astype(float).values   if "high"   in weekly.columns else c.copy()
            l = weekly["low"].astype(float).values    if "low"    in weekly.columns else c.copy()

            def mom(n): return float((c[-1]/c[-n]-1)*100) if len(c)>=n and c[-n]>0 else 0.0
            def ma(n):  return c[-n:].mean() if len(c)>=n else c.mean()

            # 모멘텀 3개 [0-2]
            wf0=mom(4); wf1=mom(13); wf2=mom(26)
            # 정배열 2개 [3-4]
            wma5=ma(5); wma13=ma(13); wma26=ma(26)
            wf3=1.0 if wma5>wma13 else -1.0
            wf4=1.0 if wma13>wma26 else -1.0
            # RSI [5]
            if len(c)>=15:
                d=np.diff(c[-15:]); g=np.where(d>0,d,0).mean(); lo=np.where(d<0,-d,0).mean()
                wf5=float(100-100/(1+g/(lo+1e-9)))
            else: wf5=50.0
            # 스토캐스틱 [6]
            if len(c)>=14 and len(h)>=14 and len(l)>=14:
                wf6=float((c[-1]-l[-14:].min())/(h[-14:].max()-l[-14:].min()+1e-9)*100)
            else: wf6=50.0
            # 거래량비율 [7]
            wf7=float(v[-1]/(v[-5:].mean()+1e-9)) if len(v)>=5 else 1.0
            # OBV방향 [8]
            obv=np.cumsum(np.where(np.diff(np.concatenate([[c[0]],c]))>0,v,-v))
            wf8=1.0 if len(obv)>1 and obv[-1]>obv[-2] else -1.0
            # 볼린저 [9-10]
            if len(c)>=20:
                bm=c[-20:].mean(); bs=c[-20:].std()+1e-9
                wf9=float((c[-1]-bm)/(bs*2)); wf10=float(bs*4/bm*100)
            else: wf9=wf10=0.0
            # MACD히스토그램 [11]
            if len(c)>=26:
                def _ema_w(arr, n):
                    if len(arr)==0: return 0.0
                    k=2/(n+1); e=float(arr[0])
                    for x in arr: e=float(x)*k+e*(1-k)
                    return e
                wf11=float(_ema_w(c,12)-_ema_w(c,26))
            else: wf11=0.0
            # 52주위치 [12], 신고가여부 [13]
            if len(c)>=52 and len(h)>=52 and len(l)>=52:
                wf12=float((c[-1]-l[-52:].min())/(h[-52:].max()-l[-52:].min()+1e-9))
                wf13=1.0 if c[-1]>=h[-52:].max()*0.98 else 0.0
            else: wf12=0.5; wf13=0.0
            # ATR정규화 [14]
            if len(h)>=5 and len(l)>=5:
                atr=np.mean([h[-i]-l[-i] for i in range(1,6)])
                wf14=float(atr/c[-1]*100) if c[-1]>0 else 0.0
            else: wf14=0.0
            # 연속상승 [15]
            st=0
            for i in range(1,min(8,len(c))):
                if c[-i]>c[-i-1]: st+=1
                else: break
            wf15=float(st/7)
            # 이격도 [16]
            wf16=float((c[-1]/wma13-1)*100) if wma13>0 else 0.0
            # CCI [17]
            if len(c)>=14 and len(h)>=14 and len(l)>=14:
                tp=(h[-14:]+l[-14:]+c[-14:])/3; cm=tp.mean(); md=np.mean(np.abs(tp-cm))
                wf17=float((tp[-1]-cm)/(0.015*md+1e-9))
            else: wf17=0.0
            # MFI [18]
            if len(c)>=14 and len(v)>=14:
                tp14=(h[-14:]+l[-14:]+c[-14:])/3 if len(h)>=14 else c[-14:]
                mf=tp14*v[-14:]
                dd=np.diff(np.concatenate([[tp14[0]],tp14]))
                pos=mf[dd>0].sum(); neg=mf[dd<=0].sum()
                wf18=float(100-100/(1+pos/(neg+1e-9)))
            else: wf18=50.0
            # 갭 [19]
            wf19=float((c[-1]-c[-2])/c[-2]*100) if len(c)>=2 and c[-2]>0 else 0.0

            return np.nan_to_num(np.array([
                wf0,wf1,wf2,wf3,wf4,wf5,wf6,wf7,wf8,wf9,
                wf10,wf11,wf12,wf13,wf14,wf15,wf16,wf17,wf18,wf19
            ],dtype=np.float32))
        except:
            return z

    # ══════════════════════════════════════════════════════════════════════════
    # 피처 추출 — 월봉 15개
    # ══════════════════════════════════════════════════════════════════════════
    def _extract_monthly_features(self, monthly) -> np.ndarray:
        z = np.zeros(self.N_MONTHLY, dtype=np.float32)
        if monthly is None or len(monthly) < 6:
            return z
        try:
            c = monthly["close"].astype(float).values
            h = monthly["high"].astype(float).values  if "high"   in monthly.columns else c.copy()
            l = monthly["low"].astype(float).values   if "low"    in monthly.columns else c.copy()
            v = monthly["volume"].astype(float).values if "volume" in monthly.columns else np.ones(len(c))

            def mom(n): return float((c[-1]/c[-n]-1)*100) if len(c)>=n and c[-n]>0 else 0.0
            def ma(n):  return c[-n:].mean() if len(c)>=n else c.mean()

            # 모멘텀 4개 [0-3]
            mf0=mom(4); mf1=mom(7); mf2=mom(13); mf3=mom(25)
            # 정배열 2개 [4-5]
            mma3=ma(3); mma6=ma(6); mma12=ma(12)
            mf4=1.0 if mma3>mma6 else -1.0
            mf5=1.0 if mma6>mma12 else -1.0
            # RSI [6]
            if len(c)>=15:
                d=np.diff(c[-15:]); g=np.where(d>0,d,0).mean(); lo=np.where(d<0,-d,0).mean()
                mf6=float(100-100/(1+g/(lo+1e-9)))
            else: mf6=50.0
            # CCI [7]
            if len(c)>=12 and len(h)>=12 and len(l)>=12:
                tp=(h[-12:]+l[-12:]+c[-12:])/3; cm=tp.mean(); md=np.mean(np.abs(tp-cm))
                mf7=float((tp[-1]-cm)/(0.015*md+1e-9))
            else: mf7=0.0
            # 52주위치 [8]
            if len(c)>=12 and len(h)>=12 and len(l)>=12:
                mf8=float((c[-1]-l[-12:].min())/(h[-12:].max()-l[-12:].min()+1e-9))
            else: mf8=0.5
            # 5년위치 [9]
            if len(c)>=60 and len(h)>=60 and len(l)>=60:
                mf9=float((c[-1]-l[-60:].min())/(h[-60:].max()-l[-60:].min()+1e-9))
            else: mf9=0.5
            # 거래량추세 [10]
            if len(v)>=6:
                mf10=float(np.polyfit(range(6),v[-6:],1)[0]/(v[-6:].mean()+1e-9))
            else: mf10=0.0
            # 연속상승 [11]
            st=0
            for i in range(1,min(6,len(c))):
                if c[-i]>c[-i-1]: st+=1
                else: break
            mf11=float(st/5)
            # 장기이격도 [12]
            mf12=float((c[-1]/mma12-1)*100) if mma12>0 else 0.0
            # ATR [13]
            if len(h)>=5 and len(l)>=5:
                atr=np.mean([h[-i]-l[-i] for i in range(1,6)])
                mf13=float(atr/c[-1]*100) if c[-1]>0 else 0.0
            else: mf13=0.0
            # 볼린저위치 [14]
            if len(c)>=12:
                bm=c[-12:].mean(); bs=c[-12:].std()+1e-9
                mf14=float((c[-1]-bm)/(bs*2))
            else: mf14=0.0

            return np.nan_to_num(np.array([
                mf0,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8,
                mf9,mf10,mf11,mf12,mf13,mf14
            ],dtype=np.float32))
        except:
            return z

    # ══════════════════════════════════════════════════════════════════════════
    # 피처 추출 — 고급통계 24개
    # ══════════════════════════════════════════════════════════════════════════
    def _extract_advanced_features(self, ohlcv) -> np.ndarray:
        z = np.zeros(self.N_ADV, dtype=np.float32)
        if ohlcv is None or len(ohlcv) < 30:
            return z
        try:
            c = ohlcv["close"].astype(float).values
            v = ohlcv["volume"].astype(float).values if "volume" in ohlcv.columns else np.ones(len(c))
            h = ohlcv["high"].astype(float).values   if "high"   in ohlcv.columns else c.copy()
            l = ohlcv["low"].astype(float).values    if "low"    in ohlcv.columns else c.copy()
            rets = np.diff(c)/c[:-1]

            # [0] 허스트 지수 (R/S 분석)
            def hurst(ts, min_n=10):
                if len(ts)<min_n*2: return 0.5
                ns=[max(min_n,int(len(ts)/k)) for k in [8,4,2]]
                rs_list=[]
                for n in ns:
                    sub=ts[:n]; m=sub.mean(); dev=np.cumsum(sub-m)
                    R=dev.max()-dev.min(); S=sub.std()
                    if S>0: rs_list.append((n,R/S))
                if len(rs_list)<2: return 0.5
                ns_log=[np.log(r[0]) for r in rs_list]
                rs_log=[np.log(r[1]) for r in rs_list]
                try: return float(np.polyfit(ns_log,rs_log,1)[0])
                except: return 0.5
            af0 = float(np.clip(hurst(c[-60:] if len(c)>=60 else c), 0, 1))

            # [1] 샘플 엔트로피 (역수 → 높을수록 추세 뚜렷)
            def sample_entropy(ts, m=2, r_frac=0.2):
                if len(ts)<10: return 0.5
                r=r_frac*ts.std()
                def count_matches(m_val):
                    cnt=0
                    for i in range(len(ts)-m_val):
                        template=ts[i:i+m_val]
                        for j in range(i+1,len(ts)-m_val):
                            if np.max(np.abs(template-ts[j:j+m_val]))<r: cnt+=1
                    return cnt
                try:
                    B=count_matches(m); A=count_matches(m+1)
                    se=-np.log(A/(B+1e-9)) if B>0 else 2.0
                    return float(np.clip(1/(se+1e-9), 0, 1))
                except: return 0.5
            af1 = sample_entropy(c[-30:] if len(c)>=30 else c)

            # [2-5] 자기상관 lag 1/2/3/5
            def autocorr(ts, lag):
                if len(ts)<lag+5: return 0.0
                return float(np.corrcoef(ts[:-lag],ts[lag:])[0,1]) if lag>0 else 1.0
            r30 = rets[-30:] if len(rets)>=30 else rets
            af2=autocorr(r30,1); af3=autocorr(r30,2)
            af4=autocorr(r30,3); af5=autocorr(r30,5)

            # [6] 분산 비율 (Variance Ratio) — 랜덤워크 검정
            if len(rets)>=20:
                v1=np.var(rets[-20:]); v5=np.var(rets[-20:][::5]+[0])
                af6=float(np.clip(v5/(5*v1+1e-9),0,3))
            else: af6=1.0

            # [7] 왜도
            if len(rets)>=10:
                m3=np.mean((rets-rets.mean())**3); std3=(rets.std()+1e-9)**3
                af7=float(np.clip(m3/std3,-3,3))
            else: af7=0.0

            # [8] 첨도
            if len(rets)>=10:
                m4=np.mean((rets-rets.mean())**4); std4=(rets.std()+1e-9)**4
                af8=float(np.clip(m4/std4-3,-5,5))
            else: af8=0.0

            # [9] 거래 회전율
            if len(v)>=20:
                af9=float(v[-20:].mean()/v.mean()) if v.mean()>0 else 1.0
            else: af9=1.0

            # [10] 가격 가속도 (2차 미분)
            if len(c)>=5:
                d1=np.diff(c[-5:]); d2=np.diff(d1)
                af10=float(d2[-1]/(abs(d1[-1])+1e-9)) if len(d2)>0 else 0.0
            else: af10=0.0

            # [11] 변동성 추세 (단기 vs 장기 변동성)
            sv=rets[-10:].std() if len(rets)>=10 else 0.01
            lv=rets[-30:].std() if len(rets)>=30 else 0.01
            af11=float(np.clip(sv/(lv+1e-9),0,3))

            # [12] 일중 위치 평균 (캔들 상대 위치)
            if len(h)>=10 and len(l)>=10:
                rng=h[-10:]-l[-10:]+1e-9
                pos=(c[-10:]-l[-10:])/rng
                af12=float(pos.mean())
            else: af12=0.5

            # [13] 상승일 비율
            if len(rets)>=20:
                af13=float(np.mean(rets[-20:]>0))
            else: af13=0.5

            # [14] 프랙탈 차원 (Higuchi 근사)
            def fractal_dim(ts, k_max=4):
                if len(ts)<k_max*3: return 1.5
                Lk=[]
                for k in range(1,k_max+1):
                    L=0
                    for m in range(k):
                        sub=ts[m::k]
                        if len(sub)<2: continue
                        L+=np.sum(np.abs(np.diff(sub)))*(len(ts)-1)/(k*(len(sub)-1)+1e-9)
                    Lk.append((np.log(k),np.log(L/(k+1e-9))))
                if len(Lk)<2: return 1.5
                xs,ys=zip(*Lk)
                try: return float(np.clip(-np.polyfit(xs,ys,1)[0],1,2))
                except: return 1.5
            af14=fractal_dim(c[-30:] if len(c)>=30 else c)

            # [15] 가격 효율성 비율 (ER) — 추세 강도
            if len(c)>=10:
                net_move=abs(c[-1]-c[-10])
                total_move=np.sum(np.abs(np.diff(c[-10:])))
                af15=float(net_move/(total_move+1e-9))
            else: af15=0.5

            # [16] 근사 엔트로피 (ApEn)
            def apen(ts, m=2, r_frac=0.2):
                if len(ts)<10: return 0.5
                r=r_frac*ts.std()+1e-9
                def phi(mm):
                    xs=[ts[i:i+mm] for i in range(len(ts)-mm+1)]
                    cnt=np.array([sum(1 for xj in xs if np.max(np.abs(xi-xj))<r)
                                  for xi in xs])
                    return np.log(cnt).mean()
                try: return float(np.clip(abs(phi(m)-phi(m+1)), 0, 3))
                except: return 0.5
            af16=apen(c[-30:] if len(c)>=30 else c)

            # [17] 단기/장기 변동성 비율
            af17=af11  # 재사용(위치상 동일)

            # [18] 가격 충격 지수 (거래량 대비 가격 변화)
            if len(v)>=5 and len(rets)>=5 and v[-5:].mean()>0:
                af18=float(np.abs(rets[-5:]).mean()/(v[-5:].mean()+1e-9)*1e6)
                af18=float(np.clip(af18, 0, 5))
            else: af18=1.0

            # [19] 거래량 가중 변동성
            if len(v)>=20 and len(rets)>=20:
                w=v[-20:]/(v[-20:].sum()+1e-9)
                af19=float(np.sqrt(np.sum(w*rets[-20:]**2))*100)
            else: af19=0.0

            # [20] 베타 (시장 대비 — 코스피 근사: 자기 60일 대 20일 변동비)
            sv20=rets[-20:].std() if len(rets)>=20 else 0.01
            sv60=rets[-60:].std() if len(rets)>=60 else sv20
            af20=float(np.clip(sv20/(sv60+1e-9), 0, 3))

            # [21] 섹터 상대 강도 (자기 변동성 상대 — 근사)
            af21 = af15  # ER 재활용 (실제 섹터 데이터 없으면 근사)

            # [22] 수익률 분포 첨도 (꼬리위험)
            af22=af8  # 첨도 재사용

            # [23] 피처 복합 신호 (MTF 교차 요약)
            af23=float(np.clip(af0*0.4 + af15*0.3 + (1-af11)*0.3, 0, 1))

            result = np.array([
                af0,af1,af2,af3,af4,af5,af6,af7,af8,af9,
                af10,af11,af12,af13,af14,af15,af16,af17,
                af18,af19,af20,af21,af22,af23
            ], dtype=np.float32)
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            return z

    # ══════════════════════════════════════════════════════════════════════════
    # 피처 통합 (109개)
    # ══════════════════════════════════════════════════════════════════════════
    def _extract_features(self, ohlcv, weekly=None, monthly=None) -> np.ndarray:
        df = self._extract_daily_features(ohlcv)
        wf = self._extract_weekly_features(weekly)  if weekly  is not None else np.zeros(self.N_WEEKLY,  dtype=np.float32)
        mf = self._extract_monthly_features(monthly) if monthly is not None else np.zeros(self.N_MONTHLY, dtype=np.float32)
        af = self._extract_advanced_features(ohlcv)
        self._last_adv = af
        return np.concatenate([df, wf, mf, af])

    # ══════════════════════════════════════════════════════════════════════════
    # MTF 교차 신호 스코어
    # ══════════════════════════════════════════════════════════════════════════
    def _mtf_signal_score(self, df, wf, mf) -> float:
        score = 0.0
        d_align = df[9]  if len(df)>9  else 0   # 단기정배열
        w_align = wf[3]  if len(wf)>3  else 0   # 주봉정배열
        m_align = mf[4]  if len(mf)>4  else 0   # 월봉정배열

        if d_align>0 and w_align>0 and m_align>0: score+=20
        elif d_align>0 and w_align>0:              score+=12
        elif d_align>0 and w_align<0:              score-=5

        d_rsi = df[12] if len(df)>12 else 50
        w_rsi = wf[5]  if len(wf)>5  else 50
        m_rsi = mf[6]  if len(mf)>6  else 50
        if 30<=d_rsi<=50 and 30<=w_rsi<=55 and 30<=m_rsi<=60: score+=15
        elif 30<=d_rsi<=50 and 30<=w_rsi<=55:                  score+=8

        if m_align>0 and w_align>0: score+=8

        _adv = self._last_adv if self._last_adv is not None else np.zeros(self.N_ADV, dtype=np.float32)
        hurst = float(_adv[0]) if len(_adv)>0 else 0.5
        er    = float(_adv[15]) if len(_adv)>15 else 0.5
        if hurst>0.6:  score+=8
        elif hurst<0.4: score-=5
        if len(_adv)>1 and _adv[1]>0.7: score+=5
        if er>0.6:     score+=5   # 강한 추세 진행 중
        return float(np.clip(score, -30, 30))

    # ══════════════════════════════════════════════════════════════════════════
    # 룰기반 예측 (ML 없을 때 / 초기)
    # ══════════════════════════════════════════════════════════════════════════
    def _rule_based_predict(self, df, wf, mf, af) -> float:
        score = 50.0
        self._last_adv = af

        # 모멘텀 [0-4]
        score += np.clip(df[0]*1.5,  -15, 15)   # 5일
        score += np.clip(df[2]*0.8,  -10, 10)   # 20일
        score += np.clip(df[3]*0.5,  -8,  8)    # 60일
        score += np.clip(df[4]*0.3,  -6,  6)    # 120일

        # 정배열 [9-11]
        score += 8  if df[9] >0 else -4          # 단기정배열
        score += 5  if df[10]>0 else -2          # 중기정배열
        score += 12 if df[11]>0 else -6          # 완전정배열

        # RSI [12-14]
        rsi = df[12]
        if   30<=rsi<=45: score+=15
        elif 45<rsi<=55:  score+=8
        elif rsi>75:      score-=12
        elif rsi<25:      score+=5
        rsi9 = df[13]
        if 25<=rsi9<=45:  score+=5
        elif rsi9>80:     score-=5

        # 볼린저 [15-17]
        score += np.clip(df[15]*-15, -10, 10)   # %B (낮을수록 매수기회)
        score += np.clip(df[16]*2,   -5,  8)    # 밴드폭

        # MACD [20-22]
        score += 8  if df[22]>0 else -4          # 히스토그램

        # WilliamsR [23]
        wr = df[23]
        if wr<-80: score+=8
        elif wr>-20: score-=6

        # CCI [24]
        score += np.clip(df[24]*0.05, -8, 8)

        # 거래량 [26-30]
        score += np.clip((df[26]-1)*8, -10, 15)  # 5일 거래량비율
        score += np.clip((df[27]-1)*5, -8,  10)  # 20일 거래량비율
        score += 5  if df[28]>0 else -3           # OBV방향 (idx28)
        score += np.clip(df[30]*3,     -5,  5)   # 거래대금증가율

        # VWAP [31]
        score += np.clip(df[31]*1.5, -8, 8)

        # 피보나치 [35-36]
        score += np.clip(df[35]*10, -8, 10)      # 피보위치

        # ATR [37-38]
        # 변동성 높으면 리스크 감점
        score += np.clip(-df[38]*0.5, -5, 0)

        # 갭 [39]
        score += np.clip(df[39]*2, -8, 8)        # 갭 방향

        # 연속상승 [41]
        score += df[41]*8

        # 변동성돌파 [42]
        score += 15 if df[42]>0.5 else 0

        # 52주위치 [43]
        score += 8 if df[43]>=0.95 else 0

        # 저항돌파 [44]
        score += 10 if df[44]>0 else 0

        # 캔들 [46-47]
        score += df[47]*3   # 양봉여부

        # 주봉
        score += np.clip(wf[0]*1.0, -8, 8)   if len(wf)>0  else 0  # 주봉모멘텀3
        score += 8 if len(wf)>3 and wf[3]>0 else (-3 if len(wf)>3 and wf[3]<0 else 0)
        score += 5 if len(wf)>5 and 30<=wf[5]<=50 else 0  # 주봉RSI 과매도권

        # 월봉
        score += 5 if len(mf)>4 and mf[4]>0 else (-2 if len(mf)>4 and mf[4]<0 else 0)
        score += np.clip(mf[0]*0.5, -5, 5) if len(mf)>0 else 0  # 월봉모멘텀3

        # 고급 피처
        if len(af)>0:  score += np.clip((af[0]-0.5)*20,  -10, 15)  # 허스트
        if len(af)>1:  score += np.clip((af[1]-0.5)*15,  -8,  10)  # 역엔트로피
        if len(af)>2:  score += af[2]*8                              # 자기상관lag1
        if len(af)>5:  score += np.clip(af[5]*3,          -8,  8)  # 가격가속도(재)
        if len(af)>12: score += np.clip((af[12]-0.5)*10,  -5,  8)  # 일중위치
        if len(af)>13: score += np.clip((af[13]-0.5)*12,  -8,  10) # 상승일비율
        if len(af)>15: score += np.clip((af[15]-0.5)*12,  -8,  10) # ER
        if len(af)>23: score += np.clip(af[23]*10,         0,  10) # 복합신호

        # MTF 교차
        score += self._mtf_signal_score(df, wf, mf)

        return float(np.clip(score, 0, 100))

    # ══════════════════════════════════════════════════════════════════════════
    # 학습
    # ══════════════════════════════════════════════════════════════════════════
    def _train_models(self, X, y):
        if len(X) < 30: return
        if self.has_xgb:
            try:
                self.xgb_model = self._xgb.XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                    eval_metric='logloss', verbosity=0)
                self.xgb_model.fit(X, y)
                print(f"[OK] XGBoost 학습 완료 ({len(X)}샘플, 피처 {X.shape[1]}개)")
            except Exception as e:
                print(f"[주의] XGBoost 실패: {e}"); self.xgb_model=None
        if self.has_lgbm:
            try:
                self.lgbm_model = self._lgb.LGBMClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                    reg_alpha=0.1, reg_lambda=1.0, verbose=-1)
                self.lgbm_model.fit(X, y)
                print(f"[OK] LightGBM 학습 완료 ({len(X)}샘플, 피처 {X.shape[1]}개)")
            except Exception as e:
                print(f"[주의] LightGBM 실패: {e}"); self.lgbm_model=None
        if self.has_cat:
            try:
                self.cat_model = self._cb.CatBoostClassifier(
                    iterations=300, depth=6, learning_rate=0.05,
                    loss_function='Logloss', verbose=0,
                    random_seed=42, l2_leaf_reg=3.0)
                self.cat_model.fit(X, y, silent=True)
                print(f"[OK] CatBoost 학습 완료 ({len(X)}샘플, 피처 {X.shape[1]}개)")
            except Exception as e:
                print(f"[주의] CatBoost 실패: {e}"); self.cat_model=None
        self._trained = True

    def fit_from_history(self, df):
        if not (self.has_xgb or self.has_lgbm): return
        X_list, y_list = [], []
        for _, row in df.iterrows():
            ohlcv   = row.get("ohlcv")
            weekly  = row.get("ohlcv_weekly")
            monthly = row.get("ohlcv_monthly")
            if ohlcv is None or len(ohlcv)<30: continue
            for i in range(25, len(ohlcv)-1):
                sub_d = ohlcv.iloc[:i]
                sub_w = weekly.iloc[:max(8,i//5)]   if weekly  is not None and len(weekly) >=8 else None
                sub_m = monthly.iloc[:max(6,i//21)] if monthly is not None and len(monthly)>=6 else None
                feats = self._extract_features(sub_d, sub_w, sub_m)
                label = 1 if float(ohlcv["close"].iloc[i])>float(ohlcv["close"].iloc[i-1]) else 0
                X_list.append(feats); y_list.append(label)
        if len(X_list) >= 30:
            self._train_models(np.array(X_list), np.array(y_list))

    # ── ★ 분봉 피처 추출 ─────────────────────────────────────────────────────
    def _extract_minute_features(self, minute_df) -> np.ndarray:
        """
        분봉 데이터 → 8개 피처
        ─ 분봉 모멘텀, 변동성, 거래량 급증, VWAP 위치
        ─ 장중 고저 위치, 분봉 추세 강도
        """
        z = np.zeros(8, dtype=np.float32)
        if minute_df is None or len(minute_df) < 5:
            return z
        try:
            c = minute_df["close"].astype(float).values
            v = minute_df["volume"].astype(float).values if "volume" in minute_df.columns else np.ones(len(c))
            h = minute_df["high"].astype(float).values  if "high"   in minute_df.columns else c.copy()
            l = minute_df["low"].astype(float).values   if "low"    in minute_df.columns else c.copy()

            # [0] 분봉 단기 모멘텀 (마지막 봉 vs 5봉 전)
            mf0 = float((c[-1]/c[-6]-1)*100) if len(c)>=6 and c[-6]>0 else 0.0
            # [1] 분봉 변동성 (표준편차)
            mf1 = float(np.std(c[-20:]/c[-20:].mean()-1)*100) if len(c)>=20 else 0.0
            # [2] 거래량 급증 비율
            vm  = v[-20:].mean() if len(v)>=20 else v.mean()
            mf2 = float(v[-1]/(vm+1e-9))
            # [3] VWAP 위치
            if len(v)>=20 and v[-20:].sum()>0:
                tp  = (h[-20:]+l[-20:]+c[-20:])/3 if len(h)>=20 else c[-20:]
                vwap= (tp*v[-20:]).sum()/(v[-20:].sum()+1e-9)
                mf3 = float((c[-1]-vwap)/vwap*100) if vwap>0 else 0.0
            else: mf3 = 0.0
            # [4] 장중 위치 (현재가의 고저 내 위치)
            hi_d = h.max() if len(h)>0 else c[-1]
            lo_d = l.min() if len(l)>0 else c[-1]
            mf4  = float((c[-1]-lo_d)/(hi_d-lo_d+1e-9))
            # [5] 분봉 추세 강도 (선형회귀 기울기)
            if len(c)>=10:
                slope = float(np.polyfit(range(10), c[-10:]/c[-10:].mean(), 1)[0])
                mf5   = float(np.clip(slope*100, -3, 3))
            else: mf5 = 0.0
            # [6] 매수 체결 비율 근사 (양봉 비율)
            if len(c)>=20:
                mf6 = float(np.mean(np.diff(c[-20:])>0))
            else: mf6 = 0.5
            # [7] 분봉 가격 효율성 (직선 이동 / 실제 이동)
            if len(c)>=10:
                net   = abs(c[-1]-c[-10])
                total = np.sum(np.abs(np.diff(c[-10:])))
                mf7   = float(net/(total+1e-9))
            else: mf7 = 0.5

            return np.nan_to_num(np.array([mf0,mf1,mf2,mf3,mf4,mf5,mf6,mf7], dtype=np.float32))
        except:
            return z

    # ── ★ 옵션 내재변동성 근사 ────────────────────────────────────────────────
    def _calc_implied_vol(self, ohlcv) -> float:
        """
        옵션 시장 없어도 과거 변동성으로 내재변동성 근사
        ─ Parkinson 변동성: 고가/저가 범위 기반 (표준 역사적 변동성보다 정확)
        ─ Yang-Zhang 변동성: 갭 포함
        ─ 반환: 연율화 변동성 (%) — VIX 유사 지표
        """
        try:
            c = ohlcv["close"].astype(float).values
            o = ohlcv["open"].astype(float).values  if "open" in ohlcv.columns else c.copy()
            h = ohlcv["high"].astype(float).values  if "high" in ohlcv.columns else c.copy()
            l = ohlcv["low"].astype(float).values   if "low"  in ohlcv.columns else c.copy()

            n = min(20, len(c)-1)
            if n < 5: return 20.0

            # Parkinson 변동성
            log_hl = np.log(h[-n:]/l[-n:]+1e-9)
            park_var = (1/(4*n*np.log(2))) * np.sum(log_hl**2)
            park_vol = float(np.sqrt(park_var*252)*100)

            # 표준 역사적 변동성
            rets = np.diff(np.log(c[-n-1:]+1e-9))
            hist_vol = float(np.std(rets)*np.sqrt(252)*100)

            # 혼합 (Parkinson 60% + 역사적 40%)
            iv = park_vol*0.6 + hist_vol*0.4
            return float(np.clip(iv, 5, 200))
        except:
            return 20.0

    # ── ★ SHAP 기반 예측 설명 ────────────────────────────────────────────────
    def explain_prediction(self, ohlcv, weekly=None, monthly=None) -> dict:
        """
        예측 근거 설명 (SHAP 또는 룰기반 기여도)
        반환: {"top_pos": [...], "top_neg": [...], "summary": str}
        """
        df_f = self._extract_daily_features(ohlcv)
        wf_f = self._extract_weekly_features(weekly)   if weekly  is not None else np.zeros(self.N_WEEKLY,  dtype=np.float32)
        mf_f = self._extract_monthly_features(monthly) if monthly is not None else np.zeros(self.N_MONTHLY, dtype=np.float32)
        af_f = self._extract_advanced_features(ohlcv)
        feats = np.concatenate([df_f, wf_f, mf_f, af_f])

        # SHAP 시도
        if self._trained and self.xgb_model is not None:
            try:
                import shap
                if self._shap_explainer is None:
                    self._shap_explainer = shap.TreeExplainer(self.xgb_model)
                shap_vals = self._shap_explainer.shap_values(feats.reshape(1,-1))[0]
                feat_names = self._get_feature_names()
                contrib = list(zip(feat_names, shap_vals))
                contrib.sort(key=lambda x: abs(x[1]), reverse=True)
                top_pos = [(n, round(float(v),3)) for n,v in contrib[:10] if v>0][:5]
                top_neg = [(n, round(float(v),3)) for n,v in contrib[:10] if v<0][:5]
                summary = f"상승요인: {','.join([n for n,_ in top_pos[:3]])} | 하락요인: {','.join([n for n,_ in top_neg[:3]])}"
                return {"method":"SHAP","top_pos":top_pos,"top_neg":top_neg,"summary":summary}
            except ImportError:
                pass  # SHAP 없으면 룰기반으로
            except Exception:
                pass

        # 룰기반 기여도 (SHAP 없을 때)
        contributions = []
        # 주요 피처별 기여도 계산
        rules = [
            ("5일모멘텀",   float(df_f[0])*1.5),
            ("20일모멘텀",  float(df_f[2])*0.8),
            ("완전정배열",  float(df_f[11])*12),
            ("RSI14",       50-float(df_f[12])),
            ("거래량비율",  (float(df_f[26])-1)*8),
            ("변동성돌파",  float(df_f[42])*15),
            ("허스트",      (float(af_f[0])-0.5)*20 if len(af_f)>0 else 0),
            ("ER효율성",    (float(af_f[15])-0.5)*12 if len(af_f)>15 else 0),
            ("주봉정배열",  float(wf_f[3])*8 if len(wf_f)>3 else 0),
            ("월봉정배열",  float(mf_f[4])*5 if len(mf_f)>4 else 0),
        ]
        top_pos = [(n,round(v,2)) for n,v in rules if v>0][:5]
        top_neg = [(n,round(v,2)) for n,v in rules if v<0][:5]
        summary = f"상승요인: {','.join([n for n,_ in top_pos[:3]])} | 하락요인: {','.join([n for n,_ in top_neg[:3]])}"
        return {"method":"룰기반","top_pos":top_pos,"top_neg":top_neg,"summary":summary}

    def _get_feature_names(self) -> list:
        daily = [
            "모멘텀5d","모멘텀10d","모멘텀20d","모멘텀60d","모멘텀120d",
            "MA5대비","MA20대비","MA60대비","MA120대비","단기정배열","중기정배열","완전정배열",
            "RSI14","RSI9","RSI21","볼린저%B","볼린저폭","볼린저위치",
            "스토캐스틱K","스토캐스틱D","MACD","시그널","히스토그램",
            "WilliamsR","CCI","MFI","거래량비율5d","거래량비율20d","OBV방향","OBV기울기","거래대금증가",
            "VWAP편차","이격도5","이격도20","이격도60","피보위치","피보레벨",
            "ATR","ATR정규화","갭","갭지속","연속상승","변동성돌파","52주위치",
            "저항돌파","지지확인","캔들몸통","양봉","고가이격","저가이격"
        ]
        weekly  = [f"주봉_{i}" for i in range(self.N_WEEKLY)]
        monthly = [f"월봉_{i}" for i in range(self.N_MONTHLY)]
        adv     = [
            "허스트","샘플엔트로피","자기상관1","자기상관2","자기상관3","자기상관5",
            "분산비율","왜도","첨도","회전율","가속도","변동성추세","일중위치","상승일비율",
            "프랙탈차원","ER효율성","ApEn","변동성비율","가격충격","거래량가중변동성",
            "베타","섹터강도","수익분포첨도","복합신호"
        ]
        return daily + weekly + monthly + adv

    # ── ★ 동적 앙상블 가중치 조정 ─────────────────────────────────────────────
    def _dynamic_weights(self, ohlcv, market_phase: str = "중립") -> tuple:
        """
        시장 국면 + 변동성에 따라 모델 가중치 동적 조정
        ─ 강세장: 모멘텀 강한 XGBoost 비중↑
        ─ 약세장: 안정적인 LSTM 비중↑
        ─ 고변동성: 룰기반 비중↑ (ML 과적합 방지)
        """
        lw = self.lstm_weight_base
        xw = self.xgb_weight_base
        gw = self.lgbm_weight_base
        cw = self.cat_weight_base

        # 변동성 계산
        try:
            c   = ohlcv["close"].astype(float).values
            vol = float(np.std(np.diff(c[-20:])/c[-20:-1])*100) if len(c)>=21 else 2.0
        except:
            vol = 2.0

        # 시장 국면별 조정
        if "강세" in market_phase:
            xw += 0.06; gw += 0.04; lw -= 0.10   # XGBoost 모멘텀 포착
        elif "약세" in market_phase:
            lw += 0.08; xw -= 0.05; gw -= 0.03   # LSTM 안정성
        elif "횡보" in market_phase:
            gw += 0.05; lw += 0.02; xw -= 0.07   # LightGBM 균형

        # 고변동성 환경
        if vol > 3.0:
            lw += 0.05; xw -= 0.03; gw -= 0.02   # 변동성 높으면 LSTM↑

        # 정규화
        total = lw + xw + gw
        lw, xw, gw = lw/total, xw/total, gw/total
        return round(lw,3), round(xw,3), round(gw,3)

    def predict_one(self, ohlcv, weekly=None, monthly=None, lstm_prob=50.0,
                    minute_df=None, market_phase="중립") -> float:
        self._last_adv = None
        df_f = self._extract_daily_features(ohlcv)
        wf_f = self._extract_weekly_features(weekly)   if weekly  is not None else np.zeros(self.N_WEEKLY,  dtype=np.float32)
        mf_f = self._extract_monthly_features(monthly) if monthly is not None else np.zeros(self.N_MONTHLY, dtype=np.float32)
        af_f = self._extract_advanced_features(ohlcv)
        self._last_adv = af_f
        feats = np.concatenate([df_f, wf_f, mf_f, af_f])

        # ★ 동적 가중치
        lw, xw, gw = self._dynamic_weights(ohlcv, market_phase)

        rule_prob = self._rule_based_predict(df_f, wf_f, mf_f, af_f)
        xgb_prob = lgbm_prob = cat_prob = rule_prob
        if self.has_xgb and self.xgb_model is not None:
            try: xgb_prob = float(self.xgb_model.predict_proba(feats.reshape(1,-1))[0][1]*100)
            except: pass
        if self.has_lgbm and self.lgbm_model is not None:
            try: lgbm_prob = float(self.lgbm_model.predict_proba(feats.reshape(1,-1))[0][1]*100)
            except: pass
        if self.has_cat and self.cat_model is not None:
            try: cat_prob = float(self.cat_model.predict_proba(feats.reshape(1,-1))[0][1]*100)
            except: pass

        if self._trained and (self.has_xgb or self.has_lgbm or self.has_cat):
            # CatBoost 있으면 4모델 앙상블
            cw = self.cat_weight if self.has_cat and self.cat_model is not None else 0.0
            tw = lw + xw + gw + cw + 1e-9
            base_score = float(np.clip(
                (lstm_prob*lw + xgb_prob*xw + lgbm_prob*gw + cat_prob*cw) / tw,
                0, 100))
        else:
            base_score = float(np.clip(lstm_prob*0.5 + rule_prob*0.5, 0, 100))

        # ★ 분봉 보정 (있으면)
        if minute_df is not None and len(minute_df) >= 5:
            mf = self._extract_minute_features(minute_df)
            minute_bonus = float(mf[0]*1.5 + mf[2]*2 + (mf[4]-0.5)*5 + mf[6]*3)
            base_score = float(np.clip(base_score + np.clip(minute_bonus,-5,5), 0, 100))

        # ★ 옵션 내재변동성 보정
        iv = self._calc_implied_vol(ohlcv)
        if iv > 60:   base_score = float(np.clip(base_score - 3, 0, 100))  # 고변동성 리스크
        elif iv < 15: base_score = float(np.clip(base_score + 2, 0, 100))  # 저변동성 안정

        return base_score

    def predict_batch(self, df, market_phase="중립") -> pd.DataFrame:
        df = df.copy()
        if not self._trained:
            self.fit_from_history(df)
        scores = []; monthly_used = weekly_used = minute_used = 0
        for _, row in df.iterrows():
            ohlcv      = row.get("ohlcv")
            weekly     = row.get("ohlcv_weekly")
            monthly    = row.get("ohlcv_monthly")
            minute_df  = row.get("ohlcv_minute")        # ★ 분봉
            phase      = str(row.get("market_phase", market_phase))
            lstm_prob  = float(row.get("lstm_score", 50))
            if weekly    is not None and len(weekly)   >=8:  weekly_used  += 1
            if monthly   is not None and len(monthly)  >=6:  monthly_used += 1
            if minute_df is not None and len(minute_df)>=5:  minute_used  += 1
            if ohlcv is None or len(ohlcv)<25:
                scores.append(lstm_prob); continue
            scores.append(self.predict_one(
                ohlcv, weekly, monthly, lstm_prob, minute_df, phase))
        print(f"[트리플+분봉] 주봉 {weekly_used}/{len(df)} / 월봉 {monthly_used}/{len(df)} / 분봉 {minute_used}/{len(df)} (피처 {self.N_TOTAL}개)")
        df["ensemble_score"] = scores
        return df