import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Backtester:
    """
    백테스트 v7.1
    ★ ATR 손절/목표가 적용 실전 백테스트
    ★ 워크포워드 백테스트 (IS/OOS)
    ★ 몬테카를로 시뮬레이션 (1000회)
       - 수익률 신뢰구간 (5th~95th percentile)
       - 최악/최선/중간 시나리오
       - 파산 확률 (MDD -50% 초과 비율)
    ★ 3전략 비교: TOP K / 동일가중 / 시장평균
    ★ 실전 지표: 샤프/소르티노/칼마/MDD/손익비
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
                 wf_train=120, wf_test=30,
                 use_atr=True, atr_stop_mult=2.0, atr_target_mult=3.0):
        self.lookback        = lookback_days
        self.top_k           = top_k
        self.cost            = commission + tax
        self.wf_train        = wf_train
        self.wf_test         = wf_test
        self.use_atr         = use_atr
        self.atr_stop_mult   = atr_stop_mult
        self.atr_target_mult = atr_target_mult

    # ── 일반 백테스트 ─────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) == 0:
            return self._empty_result()

        lens = [len(row.get("ohlcv")) for _, row in df.iterrows()
                if row.get("ohlcv") is not None]
        if not lens:
            return self._empty_result()
        max_avail = max(lens)
        auto_lb   = max(20, min(self.lookback, max_avail - 5))
        if auto_lb < self.lookback:
            print(f"  [백테스트] lookback {self.lookback}→{auto_lb}일 자동조정")

        top_rets, top_log = [], []
        for _, row in df.head(self.top_k).iterrows():
            ohlcv = row.get("ohlcv")
            if ohlcv is None or len(ohlcv) < auto_lb + 2:
                continue
            res = self._bt_atr(ohlcv, str(row.get("name","")), auto_lb) \
                  if self.use_atr else \
                  self._bt_single(ohlcv, str(row.get("name","")), auto_lb)
            top_rets.extend(res["rets"])
            top_log.append(res["summary"])

        eq_lists, mkt_lists = [], []
        for _, row in df.head(20).iterrows():
            ohlcv = row.get("ohlcv")
            if ohlcv is None or len(ohlcv) < auto_lb + 2:
                continue
            res = self._bt_atr(ohlcv, "", auto_lb) if self.use_atr \
                  else self._bt_single(ohlcv, "", auto_lb)
            eq_lists.append(res["rets"])
        for _, row in df.iterrows():
            ohlcv = row.get("ohlcv")
            if ohlcv is None or len(ohlcv) < auto_lb + 2:
                continue
            try:
                c = ohlcv["close"].astype(float).values[-(auto_lb+1):]
                mkt_lists.append(((c[1:]/c[:-1])-1)*100)
            except:
                continue

        return {
            "top":    self._calc_stats(top_rets, top_log),
            "equal":  self._calc_stats(self._avg(eq_lists),  []),
            "market": self._calc_stats(self._avg(mkt_lists), []),
        }

    # ── ★ 몬테카를로 시뮬레이션 ──────────────────────────────────────────────
    def run_montecarlo(self, df: pd.DataFrame,
                       n_simulations: int = 1000,
                       n_days: int = 60) -> dict:
        """
        몬테카를로 시뮬레이션
        - 과거 수익률을 무작위로 섞어서 n_simulations번 시뮬레이션
        - 수익률 신뢰구간 / 최악·최선·중간 시나리오
        - 파산 확률 (MDD -30% 초과)
        - VaR (Value at Risk) 5%
        """
        if df is None or len(df) == 0:
            return self._empty_mc()

        # TOP K 종목 수익률 수집
        all_rets = []
        for _, row in df.head(self.top_k).iterrows():
            ohlcv = row.get("ohlcv")
            if ohlcv is None or len(ohlcv) < 30:
                continue
            try:
                c    = ohlcv["close"].astype(float).values
                rets = ((c[1:]/c[:-1]) - 1) * 100
                all_rets.extend(rets.tolist())
            except:
                continue

        if len(all_rets) < 20:
            return self._empty_mc()

        rets_arr = np.array(all_rets)
        # 비용 반영
        rets_arr -= self.cost * 100

        # 시뮬레이션 실행
        final_returns   = []
        max_drawdowns   = []
        equity_paths    = []
        bankruptcy_count = 0

        np.random.seed(42)
        for _ in range(n_simulations):
            # 무작위 샘플링 (복원추출)
            sampled = np.random.choice(rets_arr, size=n_days, replace=True)
            cum     = np.cumprod(1 + sampled/100)
            eq      = (cum - 1) * 100

            # MDD 계산
            peak = np.maximum.accumulate(cum)
            dd   = (cum - peak) / (peak + 1e-9) * 100
            mdd  = float(dd.min())

            final_returns.append(float(eq[-1]))
            max_drawdowns.append(mdd)
            equity_paths.append(eq.tolist())

            if mdd < -30:
                bankruptcy_count += 1

        final_arr = np.array(final_returns)
        mdd_arr   = np.array(max_drawdowns)

        # 신뢰구간
        p5  = float(np.percentile(final_arr, 5))
        p25 = float(np.percentile(final_arr, 25))
        p50 = float(np.percentile(final_arr, 50))
        p75 = float(np.percentile(final_arr, 75))
        p95 = float(np.percentile(final_arr, 95))

        # VaR (5%) - 최악 5% 손실
        var_5 = float(np.percentile(final_arr, 5))

        # 대표 경로 3개
        sorted_idx = np.argsort(final_arr)
        worst_path  = equity_paths[sorted_idx[int(n_simulations*0.05)]]
        median_path = equity_paths[sorted_idx[n_simulations//2]]
        best_path   = equity_paths[sorted_idx[int(n_simulations*0.95)]]

        # 승률 (양수 수익 비율)
        win_rate = float((final_arr > 0).mean() * 100)

        return {
            "n_simulations":    n_simulations,
            "n_days":           n_days,
            "mean_return":      round(float(final_arr.mean()), 2),
            "std_return":       round(float(final_arr.std()), 2),
            "p5":               round(p5,  2),
            "p25":              round(p25, 2),
            "p50":              round(p50, 2),
            "p75":              round(p75, 2),
            "p95":              round(p95, 2),
            "var_5":            round(var_5, 2),
            "win_rate":         round(win_rate, 1),
            "avg_mdd":          round(float(mdd_arr.mean()), 2),
            "worst_mdd":        round(float(mdd_arr.min()), 2),
            "bankruptcy_prob":  round(bankruptcy_count/n_simulations*100, 1),
            "worst_path":       worst_path,
            "median_path":      median_path,
            "best_path":        best_path,
            "verdict":          self._mc_verdict(p50, p5, win_rate,
                                                 bankruptcy_count/n_simulations*100),
        }

    def _mc_verdict(self, p50, p5, win_rate, bankruptcy_prob) -> str:
        if p50 > 5 and p5 > -5 and win_rate > 60 and bankruptcy_prob < 5:
            return "✅ 우수 — 기대수익 양호, 리스크 낮음"
        elif p50 > 0 and win_rate > 50:
            return "🟡 보통 — 수익 가능하나 변동성 주의"
        elif p50 < 0 or bankruptcy_prob > 20:
            return "🔴 위험 — 손실 확률 높음, 전략 재검토"
        else:
            return "🟠 주의 — 조건부 적용 권장"

    # ── ATR 실전 백테스트 ─────────────────────────────────────────────────────
    def _bt_atr(self, ohlcv, name: str, lookback: int) -> dict:
        try:
            c = ohlcv["close"].astype(float).values
            h = ohlcv["high"].astype(float).values  if "high" in ohlcv.columns else c.copy()
            l = ohlcv["low"].astype(float).values   if "low"  in ohlcv.columns else c.copy()

            if len(c) < lookback + 2:
                return {"rets": [], "summary": None}

            c = c[-(lookback+1):]
            h = h[-(lookback+1):]
            l = l[-(lookback+1):]

            atr_period = min(14, len(c)-1)
            tr_arr = np.array([
                max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
                for i in range(1, atr_period+1)
            ])
            atr = float(tr_arr.mean())

            rets = []
            i = 0
            while i < len(c) - 1:
                entry  = float(c[i])
                stop   = entry - atr * self.atr_stop_mult
                target = entry + atr * self.atr_target_mult
                exited = False
                for j in range(i+1, min(i+21, len(c))):
                    lo = float(l[j])
                    hi = float(h[j])
                    if lo <= stop:
                        rets.append((stop-entry)/entry*100 - self.cost*100)
                        i = j+1; exited=True; break
                    elif hi >= target:
                        rets.append((target-entry)/entry*100 - self.cost*100)
                        i = j+1; exited=True; break
                if not exited:
                    end_idx = min(i+20, len(c)-1)
                    rets.append((float(c[end_idx])-entry)/entry*100 - self.cost*100)
                    i = end_idx+1

            if not rets:
                return {"rets": [], "summary": None}
            wins    = sum(1 for r in rets if r > 0)
            tot     = float(np.prod(1+np.array(rets)/100)-1)*100
            summary = (name, len(rets), wins, round(tot,2), round(np.mean(rets),3))
            return {"rets": rets, "summary": summary}
        except:
            return {"rets": [], "summary": None}

    # ── 단순 보유 백테스트 ────────────────────────────────────────────────────
    def _bt_single(self, ohlcv, name: str, lookback: int = None) -> dict:
        try:
            lb  = lookback if lookback is not None else self.lookback
            c   = ohlcv["close"].astype(float).values[-(lb+1):]
            net = ((c[1:]/c[:-1]-1)*100 - self.cost*100).tolist()
            tot = float(np.prod(1+np.array(net)/100)-1)*100
            return {"rets": net,
                    "summary": (name, len(net), sum(1 for r in net if r>0),
                                round(tot,2), round(np.mean(net),3))}
        except:
            return {"rets": [], "summary": None}

    # ── 워크포워드 백테스트 ───────────────────────────────────────────────────
    def run_walkforward(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) == 0:
            return self._empty_wf()

        train = self.wf_train
        test  = self.wf_test

        valid = [(row, row.get("ohlcv")) for _, row in df.iterrows()
                 if row.get("ohlcv") is not None
                 and len(row.get("ohlcv")) >= train+test+5]

        if len(valid) < 3:
            max_len = max((len(row.get("ohlcv",[])) for _, row in df.iterrows()
                          if row.get("ohlcv") is not None), default=0)
            if max_len >= 50:
                train = max(20, max_len//5)
                test  = max(10, max_len//10)
                valid = [(row, row.get("ohlcv")) for _, row in df.iterrows()
                         if row.get("ohlcv") is not None
                         and len(row.get("ohlcv")) >= train+test+5]
            if len(valid) < 3:
                return self._empty_wf()

        max_len   = max(len(o) for _, o in valid)
        n_windows = max(1, (max_len-train)//test)

        is_all, oos_all, window_log = [], [], []

        for w in range(n_windows):
            s  = w*test
            ie = s+train
            oe = ie+test

            scored = []
            for row, ohlcv in valid:
                if len(ohlcv) < oe:
                    continue
                try:
                    c_is = ohlcv["close"].astype(float).values[s:ie]
                    scored.append((row, ohlcv, self._mom_score(c_is)))
                except:
                    continue

            if len(scored) < 3:
                continue
            scored.sort(key=lambda x: x[2], reverse=True)
            picks = scored[:self.top_k]

            is_w, oos_w = [], []
            for row, ohlcv, _ in picks:
                c_is = ohlcv["close"].astype(float).values[s:ie]
                if len(c_is) > 1:
                    is_w.extend((((c_is[1:]/c_is[:-1])-1)*100 - self.cost*100).tolist())
                c_oos = ohlcv["close"].astype(float).values[ie:oe]
                if len(c_oos) > 1:
                    if self.use_atr:
                        tmp = ohlcv.iloc[ie:oe]
                        res = self._bt_atr(tmp, "", len(c_oos)-1)
                        oos_w.extend(res["rets"])
                    else:
                        oos_w.extend((((c_oos[1:]/c_oos[:-1])-1)*100 - self.cost*100).tolist())

            if is_w and oos_w:
                is_all.extend(is_w)
                oos_all.extend(oos_w)
                window_log.append({
                    "윈도우":     w+1,
                    "IS수익(%)":  round(float(np.prod(1+np.array(is_w)/100)-1)*100, 2),
                    "OOS수익(%)": round(float(np.prod(1+np.array(oos_w)/100)-1)*100, 2),
                    "선정종목":   ", ".join([r[0].get("name","") for r in picks[:3]]),
                })

        if not oos_all:
            return self._empty_wf()

        is_st  = self._calc_stats(is_all,  [])
        oos_st = self._calc_stats(oos_all, [])
        oos_a  = np.array(oos_all)

        overfit  = round(oos_st["total_return"]/(abs(is_st["total_return"])+1e-9), 3)
        ir       = round(float(oos_a.mean()/(oos_a.std()+1e-9)*np.sqrt(252)), 3)
        oos_hit  = round(float(np.mean(oos_a>0)*100), 1)

        return {
            "is":            is_st,
            "oos":           oos_st,
            "overfit_ratio": overfit,
            "info_ratio":    ir,
            "oos_hit_rate":  oos_hit,
            "n_windows":     len(window_log),
            "window_log":    pd.DataFrame(window_log) if window_log else pd.DataFrame(),
            "verdict":       self._verdict(overfit, oos_hit, ir),
            "atr_used":      self.use_atr,
        }

    # ── 공통 유틸 ─────────────────────────────────────────────────────────────
    def _mom_score(self, c: np.ndarray) -> float:
        if len(c) < 5: return 0.0
        s  = (c[-1]/c[-6]-1)*100  if len(c)>=6  else 0
        s += (c[-1]/c[-21]-1)*100 if len(c)>=21 else 0
        ma5  = c[-5:].mean()
        ma20 = c[-20:].mean() if len(c)>=20 else c.mean()
        s += 15 if ma5>ma20 else -5
        if len(c)>=15:
            d  = np.diff(c[-15:])
            g  = np.where(d>0,d,0).mean()
            lo = np.where(d<0,-d,0).mean()
            rsi = 100-100/(1+g/(lo+1e-9))
            s += 20 if 30<=rsi<=50 else 10 if rsi<30 else -10 if rsi>70 else 0
        return float(s)

    def _verdict(self, overfit, hit, ir) -> str:
        if   overfit>0.7 and hit>52 and ir>0.5: return "✅ 우수 — 실전 적용 추천"
        elif overfit>0.4 and hit>50:             return "🟡 보통 — 신중하게 적용"
        elif overfit>0:                          return "🟠 주의 — 파라미터 재검토"
        else:                                    return "🔴 위험 — OOS 성과 음수"

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
                "n_windows":0,"window_log":pd.DataFrame(),
                "verdict":"데이터 부족","atr_used":self.use_atr}

    def _empty_mc(self):
        return {"n_simulations":0,"n_days":0,"mean_return":0,"std_return":0,
                "p5":0,"p25":0,"p50":0,"p75":0,"p95":0,"var_5":0,
                "win_rate":0,"avg_mdd":0,"worst_mdd":0,"bankruptcy_prob":0,
                "worst_path":[],"median_path":[],"best_path":[],
                "verdict":"데이터 부족"}