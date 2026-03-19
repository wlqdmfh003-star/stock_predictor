import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Backtester:
    """
    v5.4 백테스트
    ★ 워크포워드 백테스트 (과적합 방지, 실제 예측력 측정)
    ─ 훈련창 120일 / 검증창 30일 슬라이딩
    ─ IS vs OOS 성과 비교 + 과적합 비율 + 정보비율
    ─ 3전략: TOP K / 동일가중 / 시장평균
    """

    PERIOD_MAP = {
        "2개월 (60일)":  60,
        "6개월 (125일)": 125,
        "1년 (250일)":   250,
        "2년 (500일)":   500,
        "3년 (750일)":   750,
    }

    def __init__(self, lookback_days=250, top_k=5,
                 commission=0.00015, tax=0.002,
                 wf_train=120, wf_test=30):
        self.lookback = lookback_days
        self.top_k    = top_k
        self.cost     = commission + tax
        self.wf_train = wf_train
        self.wf_test  = wf_test

    # ── 일반 백테스트 ─────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) == 0:
            return self._empty_result()

        # ★ 실제 데이터 길이에 맞게 lookback 자동 조정
        lens = [len(row.get("ohlcv")) for _,row in df.iterrows()
                if row.get("ohlcv") is not None]
        if not lens:
            return self._empty_result()
        max_avail = max(lens)
        # 데이터 최대 길이 - 5일 여유, 최소 20일 보장
        auto_lb = max(20, min(self.lookback, max_avail - 5))
        if auto_lb < self.lookback:
            print(f"  [백테스트] lookback {self.lookback}→{auto_lb}일 자동조정 (데이터 최대 {max_avail}일)")

        top_rets, top_log = [], []
        for _, row in df.head(self.top_k).iterrows():
            ohlcv = row.get("ohlcv")
            if ohlcv is None or len(ohlcv) < auto_lb+2: continue
            res = self._bt_single(ohlcv, str(row.get("name","")), auto_lb)
            top_rets.extend(res["rets"]); top_log.append(res["summary"])
        eq_lists = []
        for _, row in df.head(20).iterrows():
            ohlcv = row.get("ohlcv")
            if ohlcv is None or len(ohlcv) < auto_lb+2: continue
            eq_lists.append(self._bt_single(ohlcv, "", auto_lb)["rets"])
        mkt_lists = []
        for _, row in df.iterrows():
            ohlcv = row.get("ohlcv")
            if ohlcv is None or len(ohlcv) < auto_lb+2: continue
            try:
                c = ohlcv["close"].astype(float).values[-(auto_lb+1):]
                mkt_lists.append(((c[1:]/c[:-1])-1)*100)
            except: continue
        return {
            "top":    self._calc_stats(top_rets, top_log),
            "equal":  self._calc_stats(self._avg(eq_lists),  []),
            "market": self._calc_stats(self._avg(mkt_lists), []),
        }

    # ── ★ 워크포워드 백테스트 ─────────────────────────────────────────────────
    def run_walkforward(self, df: pd.DataFrame) -> dict:
        """
        슬라이딩 윈도우 워크포워드
        훈련창(IS) → 검증창(OOS) 반복
        IS 대비 OOS 성과로 실제 예측력 측정
        """
        if df is None or len(df) == 0:
            return self._empty_wf()

        train = self.wf_train
        test  = self.wf_test

        # 충분한 데이터 있는 종목만 (데이터 부족시 훈련창 자동 축소)
        valid = [(row, row.get("ohlcv")) for _, row in df.iterrows()
                 if row.get("ohlcv") is not None
                 and len(row.get("ohlcv")) >= train+test+5]
        if len(valid) < 3:
            # 훈련창 축소 시도
            max_len = max((len(row.get("ohlcv",[])) for _, row in df.iterrows()
                          if row.get("ohlcv") is not None), default=0)
            if max_len >= 50:
                train = max(20, max_len // 5)
                test  = max(10, max_len // 10)
                valid = [(row, row.get("ohlcv")) for _, row in df.iterrows()
                         if row.get("ohlcv") is not None
                         and len(row.get("ohlcv")) >= train+test+5]
                print(f"  [워크포워드] 훈련창 자동 조정: train={train}일, test={test}일")
            if len(valid) < 3:
                return self._empty_wf()

        max_len   = max(len(o) for _, o in valid)
        n_windows = max(1, (max_len - train) // test)

        is_all, oos_all, window_log = [], [], []

        for w in range(n_windows):
            s   = w * test
            ie  = s + train
            oe  = ie + test

            # IS: 모멘텀 점수로 종목 랭킹
            scored = []
            for row, ohlcv in valid:
                if len(ohlcv) < oe: continue
                try:
                    c_is = ohlcv["close"].astype(float).values[s:ie]
                    scored.append((row, ohlcv, self._mom_score(c_is)))
                except: continue

            if len(scored) < 3: continue
            scored.sort(key=lambda x: x[2], reverse=True)
            picks = scored[:self.top_k]

            # IS 수익
            is_w = []
            for row, ohlcv, _ in picks:
                c = ohlcv["close"].astype(float).values[s:ie]
                if len(c)>1:
                    is_w.extend((((c[1:]/c[:-1])-1)*100 - self.cost*100).tolist())

            # OOS 수익
            oos_w = []
            for row, ohlcv, _ in picks:
                c = ohlcv["close"].astype(float).values[ie:oe]
                if len(c)>1:
                    oos_w.extend((((c[1:]/c[:-1])-1)*100 - self.cost*100).tolist())

            if is_w and oos_w:
                is_all.extend(is_w); oos_all.extend(oos_w)
                window_log.append({
                    "윈도우": w+1,
                    "IS수익(%)":  round(float(np.prod(1+np.array(is_w)/100)-1)*100, 2),
                    "OOS수익(%)": round(float(np.prod(1+np.array(oos_w)/100)-1)*100, 2),
                    "선정종목":   ", ".join([r[0].get("name","") for r in picks[:3]]),
                })

        if not oos_all:
            return self._empty_wf()

        is_st  = self._calc_stats(is_all,  [])
        oos_st = self._calc_stats(oos_all, [])
        oos_a  = np.array(oos_all)

        overfit = round(oos_st["total_return"]/(abs(is_st["total_return"])+1e-9), 3)
        ir      = round(float(oos_a.mean()/(oos_a.std()+1e-9)*np.sqrt(252)), 3)
        oos_hit = round(float(np.mean(oos_a>0)*100), 1)

        return {
            "is":            is_st,
            "oos":           oos_st,
            "overfit_ratio": overfit,
            "info_ratio":    ir,
            "oos_hit_rate":  oos_hit,
            "n_windows":     len(window_log),
            "window_log":    pd.DataFrame(window_log) if window_log else pd.DataFrame(),
            "verdict":       self._verdict(overfit, oos_hit, ir),
        }

    def _mom_score(self, c: np.ndarray) -> float:
        if len(c) < 5: return 0.0
        s  = (c[-1]/c[-6]-1)*100  if len(c)>=6  else 0
        s += (c[-1]/c[-21]-1)*100 if len(c)>=21 else 0
        ma5  = c[-5:].mean(); ma20 = c[-20:].mean() if len(c)>=20 else c.mean()
        s += 15 if ma5>ma20 else -5
        if len(c)>=15:
            d = np.diff(c[-15:]); g=np.where(d>0,d,0).mean(); lo=np.where(d<0,-d,0).mean()
            rsi = 100-100/(1+g/(lo+1e-9))
            s += 20 if 30<=rsi<=50 else 10 if rsi<30 else -10 if rsi>70 else 0
        return float(s)

    def _verdict(self, overfit, hit, ir) -> str:
        if   overfit>0.7 and hit>52 and ir>0.5: return "✅ 우수 — 실전 적용 추천"
        elif overfit>0.4 and hit>50:             return "🟡 보통 — 신중하게 적용"
        elif overfit>0:                          return "🟠 주의 — 파라미터 재검토"
        else:                                    return "🔴 위험 — OOS 성과 음수"

    # ── 공통 ─────────────────────────────────────────────────────────────────
    def _bt_single(self, ohlcv, name, lookback=None):
        try:
            lb  = lookback if lookback is not None else self.lookback
            c   = ohlcv["close"].astype(float).values[-(lb+1):]
            net = ((c[1:]/c[:-1]-1)*100 - self.cost*100).tolist()
            tot = float(np.prod(1+np.array(net)/100)-1)*100
            return {"rets": net,
                    "summary": (name, len(net), sum(1 for r in net if r>0),
                                round(tot,2), round(np.mean(net),3))}
        except:
            return {"rets":[], "summary":None}

    def _avg(self, lists):
        if not lists: return []
        ml = max(len(r) for r in lists)
        return np.array([np.pad(r,(0,ml-len(r)),constant_values=0)
                         for r in lists]).mean(axis=0).tolist()

    def _calc_stats(self, rets, log):
        if not rets: return self._empty_stats()
        r   = np.array(rets)
        cum = np.cumprod(1+r/100)
        eq  = ((cum-1)*100).tolist()
        tot = float(cum[-1]-1)*100
        wr  = float(np.mean(r>0)*100)
        sh  = float(r.mean()/(r.std()+1e-9)*np.sqrt(252))
        dn  = r[r<0].std() if len(r[r<0])>0 else 1e-9
        so  = float(r.mean()/(dn+1e-9)*np.sqrt(252))
        mdd = self._mdd(eq)
        cal = float(tot/(abs(mdd)+1e-9))
        aw  = r[r>0].mean() if len(r[r>0])>0 else 0
        al  = abs(r[r<0].mean()) if len(r[r<0])>0 else 1e-9
        pnl = float(aw/(al+1e-9))
        mc = cc = 0
        for v in r:
            if v<0: cc+=1; mc=max(mc,cc)
            else:   cc=0
        monthly = [round(float(np.prod(1+r[i:i+21]/100)-1)*100,2)
                   for i in range(0,len(r),21)]
        log_df = pd.DataFrame([s for s in log if s],
                              columns=["종목","총거래","승","총수익(%)","평균수익(%)"]) \
                 if log else pd.DataFrame()
        return {"total_return":round(tot,2),"win_rate":round(wr,2),
                "sharpe":round(sh,3),"sortino":round(so,3),"calmar":round(cal,3),
                "max_drawdown":round(mdd,2),"pnl_ratio":round(pnl,3),
                "max_cons_loss":mc,"equity_curve":eq,"monthly":monthly,
                "trade_log":log_df}

    def _mdd(self, eq):
        if not eq: return 0.0
        peak=-np.inf; mdd=0.0
        for v in eq:
            peak=max(peak,v); mdd=min(mdd,v-peak)
        return float(mdd)

    def _empty_stats(self):
        return {"total_return":0,"win_rate":0,"sharpe":0,"sortino":0,"calmar":0,
                "max_drawdown":0,"pnl_ratio":0,"max_cons_loss":0,
                "equity_curve":[],"monthly":[],"trade_log":pd.DataFrame()}

    def _empty_result(self):
        return {"top":self._empty_stats(),"equal":self._empty_stats(),
                "market":self._empty_stats()}

    def _empty_wf(self):
        return {"is":self._empty_stats(),"oos":self._empty_stats(),
                "overfit_ratio":0,"info_ratio":0,"oos_hit_rate":0,
                "n_windows":0,"window_log":pd.DataFrame(),"verdict":"데이터 부족"}