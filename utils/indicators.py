import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """
    기술적 지표 계산 v7.0
    ★ 기존: RSI / MACD / 볼린저 / ATR / 모멘텀
    ★ 신규: 스토캐스틱 (과매수/과매도 정밀 감지)
    ★ 신규: CCI (추세 강도 + 이탈 감지)
    ★ 신규: MFI (자금 흐름 지수 — 거래량 반영 RSI)
    ★ 신규: OBV 기울기 + 다이버전스 감지
    ★ 신규: VWAP 이탈 강도 + 위/아래 구분
    ★ 기존: 캔들패턴 15가지 / 거래량프로파일 / 리스크지표
    """

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            ohlcv = row.get("ohlcv")
            if ohlcv is None or len(ohlcv) < 30:
                r = row.copy()
                r.update(self._default_indicators())
                results.append(r)
                continue
            r = row.copy()
            r.update(self._calc_row(ohlcv, row))
            results.append(r)
        return pd.DataFrame(results).reset_index(drop=True)

    # ── 기존 지표 ──────────────────────────────────────────────────

    def _rsi(self, close, period=14):
        delta = close.diff().dropna()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.rolling(period).mean()
        avg_l = loss.rolling(period).mean()
        rs    = avg_g / (avg_l + 1e-9)
        rsi   = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if len(rsi) > 0 else 50.0

    def _macd(self, close):
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - signal
        cross  = int(hist.iloc[-1] > 0 and hist.iloc[-2] <= 0)
        return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1]), cross

    def _bollinger(self, close, period=20):
        ma    = close.rolling(period).mean()
        std   = close.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        cur   = float(close.iloc[-1])
        ub    = float(upper.iloc[-1])
        lb    = float(lower.iloc[-1])
        pct_b = (cur - lb) / (ub - lb + 1e-9)
        width = (ub - lb) / float(ma.iloc[-1] + 1e-9)
        return pct_b, width, ub, lb

    def _momentum(self, close):
        ma5  = float(close.rolling(5).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else ma20
        cur  = float(close.iloc[-1])
        score  = 0
        score += 25 * int(cur > ma5)
        score += 25 * int(ma5 > ma20)
        score += 25 * int(ma20 > ma60)
        score += 25 * int(cur > ma60)
        mom_5  = (cur / float(close.iloc[-6])  - 1) * 100 if len(close) >= 6  else 0.0
        mom_20 = (cur / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
        mom_60 = (cur / float(close.iloc[-61]) - 1) * 100 if len(close) >= 61 else 0.0
        return score, ma5, ma20, ma60, mom_5, mom_20, mom_60

    def _volatility_breakout(self, ohlcv, k=0.5):
        if len(ohlcv) < 2:
            return 0
        prev   = ohlcv.iloc[-2]
        target = float(ohlcv.iloc[-1]["open"]) + \
                 (float(prev["high"]) - float(prev["low"])) * k
        return int(float(ohlcv.iloc[-1]["close"]) > target)

    def _atr(self, ohlcv, period=14):
        h  = ohlcv["high"].astype(float)
        l  = ohlcv["low"].astype(float)
        c  = ohlcv["close"].astype(float).shift(1)
        tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])


    # ── ★ 스토캐스틱 (Stochastic) ────────────────────────────────
    def _stochastic(self, ohlcv: pd.DataFrame,
                    k_period=14, d_period=3) -> tuple:
        """
        스토캐스틱 %K / %D
        %K = (현재가 - 최저가) / (최고가 - 최저가) * 100
        %D = %K의 3일 이동평균
        과매수: %K > 80 / 과매도: %K < 20
        """
        try:
            c = ohlcv["close"].astype(float)
            h = ohlcv["high"].astype(float)  if "high" in ohlcv.columns else c
            l = ohlcv["low"].astype(float)   if "low"  in ohlcv.columns else c

            n = min(k_period, len(c))
            lowest  = l.rolling(n).min()
            highest = h.rolling(n).max()
            stoch_k = (c - lowest) / (highest - lowest + 1e-9) * 100
            stoch_d = stoch_k.rolling(d_period).mean()

            k = float(stoch_k.iloc[-1]) if len(stoch_k) > 0 else 50.0
            d = float(stoch_d.iloc[-1]) if len(stoch_d) > 0 else 50.0

            # 스토캐스틱 점수
            score = 50.0
            if   k < 20:              score += 25   # 과매도 → 반등 기대
            elif k < 30:              score += 15
            elif k > 80:              score -= 20   # 과매수 → 하락 주의
            elif k > 70:              score -= 10
            if k > d and k < 50:      score += 10   # 골든크로스 + 저점
            if k < d and k > 50:      score -= 10   # 데드크로스 + 고점

            return float(k), float(d), float(np.clip(score, 0, 100))
        except Exception:
            return 50.0, 50.0, 50.0

    # ── ★ CCI (Commodity Channel Index) ─────────────────────────
    def _cci(self, ohlcv: pd.DataFrame, period=20) -> tuple:
        """
        CCI = (전형가격 - MA) / (0.015 × 평균편차)
        +100 이상: 과매수 / -100 이하: 과매도
        ±200 이상: 극단적 이탈 (반전 신호 강함)
        """
        try:
            c = ohlcv["close"].astype(float)
            h = ohlcv["high"].astype(float)  if "high" in ohlcv.columns else c
            l = ohlcv["low"].astype(float)   if "low"  in ohlcv.columns else c

            tp  = (h + l + c) / 3   # 전형 가격
            ma  = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (tp - ma) / (0.015 * mad + 1e-9)
            val = float(cci.iloc[-1]) if len(cci) > 0 else 0.0

            # CCI 점수
            score = 50.0
            if   val <= -200:  score += 30   # 극단적 과매도 → 강한 반등
            elif val <= -100:  score += 20
            elif val <= -50:   score += 10
            elif val >= 200:   score -= 25   # 극단적 과매수 → 하락 주의
            elif val >= 100:   score -= 15
            elif val >= 50:    score -= 5
            # 추세 방향
            if len(cci) >= 3:
                trend = cci.iloc[-1] - cci.iloc[-3]
                if   trend > 50:  score += 10   # CCI 급상승
                elif trend < -50: score -= 10

            return float(val), float(np.clip(score, 0, 100))
        except Exception:
            return 0.0, 50.0

    # ── ★ MFI (Money Flow Index) ─────────────────────────────────
    def _mfi(self, ohlcv: pd.DataFrame, period=14) -> tuple:
        """
        MFI = 거래량 반영 RSI (자금 흐름 지수)
        거래량이 많은 날의 방향을 더 중요하게 반영
        과매수: >80 / 과매도: <20
        """
        try:
            c = ohlcv["close"].astype(float)
            h = ohlcv["high"].astype(float)  if "high" in ohlcv.columns else c
            l = ohlcv["low"].astype(float)   if "low"  in ohlcv.columns else c
            v = ohlcv["volume"].astype(float) if "volume" in ohlcv.columns else                 pd.Series(np.ones(len(c)))

            tp   = (h + l + c) / 3          # 전형 가격
            rmf  = tp * v                    # Raw Money Flow
            diff = tp.diff()

            pmf  = rmf.where(diff > 0, 0).rolling(period).sum()   # 양의 자금흐름
            nmf  = rmf.where(diff < 0, 0).rolling(period).sum()   # 음의 자금흐름
            mfi  = 100 - (100 / (1 + pmf / (nmf + 1e-9)))
            val  = float(mfi.iloc[-1]) if len(mfi) > 0 else 50.0

            # MFI 점수
            score = 50.0
            if   val < 20:   score += 25   # 과매도 + 자금 유입 시작
            elif val < 30:   score += 15
            elif val > 80:   score -= 20   # 과매수 + 자금 유출 시작
            elif val > 70:   score -= 10
            # 다이버전스 감지 (간이)
            if len(c) >= period+5 and val < 40:
                price_trend = c.iloc[-1] - c.iloc[-(period//2)]
                if price_trend < 0 and val > 40:
                    score += 15   # 불리시 다이버전스

            return float(val), float(np.clip(score, 0, 100))
        except Exception:
            return 50.0, 50.0

    # ── ★ OBV 기울기 + 다이버전스 ───────────────────────────────
    def _obv_advanced(self, ohlcv: pd.DataFrame) -> dict:
        """
        OBV (On Balance Volume) 고도화
        - OBV 기울기: 상승추세 여부
        - 가격-OBV 다이버전스: 추세 반전 신호
        """
        try:
            c = ohlcv["close"].astype(float)
            v = ohlcv["volume"].astype(float) if "volume" in ohlcv.columns else                 pd.Series(np.ones(len(c)))

            direction = c.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            obv = (v * direction).cumsum()

            # OBV 기울기 (20일)
            if len(obv) >= 20:
                obv_20  = float(obv.iloc[-1] - obv.iloc[-20])
                obv_slope = obv_20 / (abs(float(obv.iloc[-20])) + 1e-9) * 100
            else:
                obv_slope = 0.0

            # 다이버전스 감지 (10일 기준)
            n = min(10, len(c)-1)
            price_dir = float(c.iloc[-1] - c.iloc[-n-1])
            obv_dir   = float(obv.iloc[-1] - obv.iloc[-n-1])
            divergence = 0
            if   price_dir > 0 and obv_dir < 0: divergence = -1  # 베어리시 다이버전스
            elif price_dir < 0 and obv_dir > 0: divergence =  1  # 불리시 다이버전스

            score = 50.0
            if   obv_slope > 10:   score += 20
            elif obv_slope > 5:    score += 12
            elif obv_slope < -10:  score -= 20
            elif obv_slope < -5:   score -= 12
            if   divergence == 1:  score += 15   # 불리시 다이버전스
            elif divergence == -1: score -= 15   # 베어리시 다이버전스

            return {
                "obv_slope":    round(obv_slope, 2),
                "obv_divergence": divergence,
                "obv_score":    float(np.clip(score, 0, 100)),
            }
        except Exception:
            return {"obv_slope": 0.0, "obv_divergence": 0, "obv_score": 50.0}

    # ── ★ VWAP 이탈 강도 ────────────────────────────────────────
    def _vwap_advanced(self, ohlcv: pd.DataFrame) -> dict:
        """
        VWAP (Volume Weighted Average Price) 고도화
        - 현재가 vs VWAP 이탈 강도
        - 위: 강세 / 아래: 약세
        - 이탈 강도: 멀수록 반전 가능성
        """
        try:
            c = ohlcv["close"].astype(float)
            h = ohlcv["high"].astype(float)  if "high" in ohlcv.columns else c
            l = ohlcv["low"].astype(float)   if "low"  in ohlcv.columns else c
            v = ohlcv["volume"].astype(float) if "volume" in ohlcv.columns else                 pd.Series(np.ones(len(c)))

            tp   = (h + l + c) / 3
            vwap = (tp * v).cumsum() / (v.cumsum() + 1e-9)

            cur       = float(c.iloc[-1])
            vwap_cur  = float(vwap.iloc[-1])
            deviation = (cur - vwap_cur) / (vwap_cur + 1e-9) * 100

            score = 50.0
            if   deviation > 5:   score += 15   # VWAP 위 강세
            elif deviation > 2:   score += 8
            elif deviation > 0:   score += 3
            elif deviation < -5:  score -= 15   # VWAP 아래 약세
            elif deviation < -2:  score -= 8
            elif deviation < 0:   score -= 3

            # 극단적 이탈 시 반전 가능성
            if   deviation > 10:  score -= 10   # 과도한 상승 → 반전 주의
            elif deviation < -10: score += 10   # 과도한 하락 → 반등 기대

            return {
                "vwap":          round(vwap_cur, 0),
                "vwap_deviation":round(deviation, 2),
                "vwap_above":    int(cur > vwap_cur),
                "vwap_score":    float(np.clip(score, 0, 100)),
            }
        except Exception:
            return {"vwap": 0.0, "vwap_deviation": 0.0,
                    "vwap_above": 0, "vwap_score": 50.0}

    # ── ★ 캔들 패턴 인식 (15가지) ─────────────────────────────────

    def _candle_patterns(self, ohlcv: pd.DataFrame) -> dict:
        if len(ohlcv) < 3:
            return self._default_candle()

        o = ohlcv["open"].astype(float)
        h = ohlcv["high"].astype(float)
        l = ohlcv["low"].astype(float)
        c = ohlcv["close"].astype(float)

        o1,o2,o3 = float(o.iloc[-3]), float(o.iloc[-2]), float(o.iloc[-1])
        h1,h2,h3 = float(h.iloc[-3]), float(h.iloc[-2]), float(h.iloc[-1])
        l1,l2,l3 = float(l.iloc[-3]), float(l.iloc[-2]), float(l.iloc[-1])
        c1,c2,c3 = float(c.iloc[-3]), float(c.iloc[-2]), float(c.iloc[-1])

        body3  = abs(c3 - o3)
        body2  = abs(c2 - o2)
        body1  = abs(c1 - o1)
        range3 = h3 - l3 + 1e-9
        range2 = h2 - l2 + 1e-9

        upper3 = h3 - max(o3, c3)
        lower3 = min(o3, c3) - l3
        upper2 = h2 - max(o2, c2)
        lower2 = min(o2, c2) - l2

        p = {}

        # 단일 캔들
        p["hammer"]        = int(lower3 >= body3*2 and upper3 <= body3*0.3 and c3 > o3)
        p["inv_hammer"]    = int(upper3 >= body3*2 and lower3 <= body3*0.3 and c3 > o3)
        p["doji"]          = int(body3 <= range3 * 0.05)
        p["marubozu_bull"] = int(c3 > o3 and body3 >= range3*0.85 and body3 > 0)
        p["marubozu_bear"] = -int(c3 < o3 and body3 >= range3*0.85 and body3 > 0)
        p["shooting_star"] = -int(upper3 >= body3*2 and lower3 <= body3*0.3 and c3 < o3)

        # 2개 캔들
        p["bullish_engulf"] = int(c2 < o2 and c3 > o3 and o3 <= c2 and c3 >= o2)
        p["bearish_engulf"] = -int(c2 > o2 and c3 < o3 and o3 >= c2 and c3 <= o2)
        p["bullish_harami"] = int(c2 < o2 and c3 > o3 and o3 > c2 and c3 < o2)
        p["gap_up"]         = int(l3 > h2)
        p["gap_down"]       = -int(h3 < l2)

        # 3개 캔들
        p["morning_star"]    = int(c1 < o1 and body2 <= range2*0.3 and c3 > o3 and c3 > (o1+c1)/2)
        p["evening_star"]    = -int(c1 > o1 and body2 <= range2*0.3 and c3 < o3 and c3 < (o1+c1)/2)
        p["three_soldiers"]  = int(c1>o1 and c2>o2 and c3>o3 and c3>c2>c1 and o2>o1 and o3>o2)
        p["three_crows"]     = -int(c1<o1 and c2<o2 and c3<o3 and c3<c2<c1 and o2<o1 and o3<o2)

        bull = sum(v for v in p.values() if v > 0)
        bear = sum(v for v in p.values() if v < 0)
        candle_score = float(np.clip(50.0 + bull*8 + bear*8, 0, 100))
        detected     = [k for k, v in p.items() if v != 0]

        return {
            **{f"cp_{k}": v for k, v in p.items()},
            "candle_score":   candle_score,
            "candle_pattern": ", ".join(detected) if detected else "없음",
            "bull_patterns":  bull,
            "bear_patterns":  abs(bear),
        }

    # ── ★ 거래량 프로파일 ─────────────────────────────────────────

    def _volume_profile(self, ohlcv: pd.DataFrame) -> dict:
        """POC / VAH / VAL 계산 (60일 기준)"""
        try:
            close  = ohlcv["close"].astype(float)
            volume = ohlcv["volume"].astype(float)
            period = min(60, len(ohlcv))
            c, v   = close.tail(period), volume.tail(period)

            p_min, p_max = c.min(), c.max()
            if p_max == p_min:
                return self._default_volume_profile()

            bins         = np.linspace(p_min, p_max, 21)
            vol_by_price = np.zeros(20)
            for price, vol in zip(c, v):
                idx = min(int((price - p_min) / (p_max - p_min) * 20), 19)
                vol_by_price[idx] += vol

            poc_idx   = np.argmax(vol_by_price)
            poc_price = (bins[poc_idx] + bins[poc_idx+1]) / 2

            # VAH / VAL (70% 거래량 구간)
            sorted_idx = np.argsort(vol_by_price)[::-1]
            cum, va    = 0.0, []
            for idx in sorted_idx:
                cum += vol_by_price[idx]
                va.append(idx)
                if cum >= vol_by_price.sum() * 0.70:
                    break
            vah = (bins[max(va)+1] + bins[max(va)]) / 2
            val = (bins[min(va)]   + bins[min(va)+1]) / 2

            cur_price  = float(close.iloc[-1])
            poc_diff   = (cur_price / poc_price - 1) * 100
            above_poc  = int(cur_price > poc_price)
            in_va      = int(val <= cur_price <= vah)

            vp_score = 50.0
            if above_poc:      vp_score += 15
            if in_va:          vp_score += 10
            if poc_diff > 5:   vp_score += 10
            elif poc_diff < -10: vp_score -= 15

            return {
                "vp_poc":       round(poc_price, 0),
                "vp_vah":       round(vah, 0),
                "vp_val":       round(val, 0),
                "vp_poc_diff":  round(poc_diff, 2),
                "vp_above_poc": above_poc,
                "vp_in_va":     in_va,
                "vp_score":     float(np.clip(vp_score, 0, 100)),
            }
        except Exception:
            return self._default_volume_profile()

    # ── ★ 리스크 관리 지표 ────────────────────────────────────────

    def _risk_metrics(self, ohlcv: pd.DataFrame, atr: float, cur_price: float) -> dict:
        """
        변동성 / MDD / 샤프비율 / ATR 기반 손절익절 / 켈리공식
        """
        try:
            close   = ohlcv["close"].astype(float)
            returns = close.pct_change().dropna()

            # 20일 연환산 변동성
            vol_20d = float(returns.tail(20).std() * np.sqrt(252) * 100)

            # 60일 샤프비율 (무위험 연 3.5% 가정)
            rf_daily  = 0.035 / 252
            ret_60d   = returns.tail(60)
            sharpe_60 = float(
                (ret_60d.mean() - rf_daily) / (ret_60d.std() + 1e-9) * np.sqrt(252)
            ) if len(ret_60d) >= 20 else 0.0

            # 60일 MDD
            p60    = close.tail(60)
            peak   = p60.expanding().max()
            mdd_60 = float(((p60 - peak) / (peak + 1e-9) * 100).min())

            # 20일 승률
            win_rate = float((returns.tail(20) > 0).mean() * 100)

            # ATR 기반 손절(2ATR) / 익절(3ATR)
            atr_stop   = round(cur_price - atr * 2, 0)
            atr_target = round(cur_price + atr * 3, 0)
            atr_ratio  = round((atr_target - cur_price) / (cur_price - atr_stop + 1e-9), 2)

            # 하프 켈리공식
            b      = atr_ratio if atr_ratio > 0 else 1.5
            p      = win_rate / 100
            kelly  = float(np.clip((b*p - (1-p)) / (b + 1e-9), 0, 0.25))
            half_kelly = round(kelly * 0.5 * 100, 1)

            # 리스크 점수
            risk_score = 50.0
            if   sharpe_60 > 1.0:  risk_score += 20
            elif sharpe_60 > 0.5:  risk_score += 10
            elif sharpe_60 < 0:    risk_score -= 15
            if   mdd_60 > -10:     risk_score += 10
            elif mdd_60 < -20:     risk_score -= 15
            if   vol_20d < 20:     risk_score += 10
            elif vol_20d > 40:     risk_score -= 10
            if   win_rate > 55:    risk_score += 10
            elif win_rate < 45:    risk_score -= 10

            return {
                "vol_20d":     round(vol_20d, 2),
                "sharpe_60":   round(sharpe_60, 2),
                "mdd_60":      round(mdd_60, 2),
                "win_rate_20": round(win_rate, 1),
                "atr_stop":    atr_stop,
                "atr_target":  atr_target,
                "atr_ratio":   atr_ratio,
                "half_kelly":  half_kelly,
                "risk_score":  float(np.clip(risk_score, 0, 100)),
            }
        except Exception:
            return self._default_risk_metrics()

    # ── ★ 매수/매도 압력 (OHLCV 기반) ────────────────────────────

    def _order_pressure(self, ohlcv: pd.DataFrame) -> dict:
        """
        실시간 호가 없을 때 OHLCV로 매수압력 추정
        KIS API 연결 시 get_orderbook() 결과로 대체됨
        """
        try:
            close  = ohlcv["close"].astype(float)
            high   = ohlcv["high"].astype(float)
            low    = ohlcv["low"].astype(float)
            open_  = ohlcv["open"].astype(float)
            volume = ohlcv["volume"].astype(float)

            # 매수 압력: (종가-저가)/(고가-저가) → 1에 가까울수록 매수 강
            buy_p  = float(((close - low) / (high - low + 1e-9)).tail(5).mean())
            sell_p = float(((high - close) / (high - low + 1e-9)).tail(5).mean())

            # 거래량 방향성
            up_vol    = float(volume[close > open_].sum())
            down_vol  = float(volume[close < open_].sum())
            vol_ratio = round(up_vol / (down_vol + 1e-9), 2)

            pressure_score = 50.0
            pressure_score += (buy_p - 0.5) * 40
            pressure_score += float(np.clip((vol_ratio - 1) * 10, -15, 15))

            return {
                "buy_pressure":   round(buy_p, 3),
                "sell_pressure":  round(sell_p, 3),
                "vol_ratio":      vol_ratio,
                "pressure_score": float(np.clip(pressure_score, 0, 100)),
                "ob_score":       50.0,   # KIS 호가 없을 때 기본값
                "ob_pressure":    1.0,
            }
        except Exception:
            return {
                "buy_pressure": 0.5, "sell_pressure": 0.5,
                "vol_ratio": 1.0, "pressure_score": 50.0,
                "ob_score": 50.0, "ob_pressure": 1.0,
            }

    # ── 전체 계산 ─────────────────────────────────────────────────

    def _calc_row(self, ohlcv, row):
        close = ohlcv["close"].astype(float)

        rsi                                       = self._rsi(close)
        macd, macd_sig, macd_hist, macd_cross     = self._macd(close)
        bb_pct, bb_width, bb_upper, bb_lower      = self._bollinger(close)
        mom_score, ma5, ma20, ma60, m5, m20, m60  = self._momentum(close)
        vb_signal = self._volatility_breakout(ohlcv)
        atr       = self._atr(ohlcv)

        candle   = self._candle_patterns(ohlcv)
        vp       = self._volume_profile(ohlcv)
        pressure = self._order_pressure(ohlcv)

        # ★ 신규 지표
        stoch_k, stoch_d, stoch_score = self._stochastic(ohlcv)
        cci_val, cci_score            = self._cci(ohlcv)
        mfi_val, mfi_score            = self._mfi(ohlcv)
        obv_data                      = self._obv_advanced(ohlcv)
        vwap_data                     = self._vwap_advanced(ohlcv)

        cur_price  = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else cur_price
        risk       = self._risk_metrics(ohlcv, atr, cur_price)

        # ATR 기반 가격 계산
        buy_price    = round(prev_close * 0.995, 0)
        exp_return   = max(1.5, min(8.0, atr / cur_price * 200)) if cur_price > 0 else 3.0
        target_price = risk["atr_target"] if risk["atr_target"] > buy_price \
                       else round(buy_price * (1 + exp_return / 100), 0)
        stop_price   = risk["atr_stop"] if 0 < risk["atr_stop"] < buy_price \
                       else round(buy_price * 0.97, 0)

        return {
            # 기존
            "rsi": rsi, "macd": macd, "macd_signal": macd_sig,
            "macd_hist": macd_hist, "macd_cross": macd_cross,
            "bb_pct": bb_pct, "bb_width": bb_width,
            "bb_upper": bb_upper, "bb_lower": bb_lower,
            "ma5": ma5, "ma20": ma20, "ma60": ma60,
            "momentum_score": mom_score,
            "mom_5d": m5, "mom_20d": m20, "mom_60d": m60,
            "vb_signal": vb_signal, "atr": atr,
            "buy_price": buy_price, "target_price": target_price,
            "stop_price": stop_price, "expected_return": exp_return,
            # 신규
            **candle, **vp, **pressure, **risk,
            # ★ v7.0 신규 지표
            "stoch_k": stoch_k, "stoch_d": stoch_d, "stoch_score": stoch_score,
            "cci": cci_val, "cci_score": cci_score,
            "mfi": mfi_val, "mfi_score": mfi_score,
            **obv_data, **vwap_data,
        }

    # ── 기본값 ────────────────────────────────────────────────────

    def _default_candle(self):
        keys = ["hammer","inv_hammer","doji","marubozu_bull","marubozu_bear",
                "shooting_star","bullish_engulf","bearish_engulf","bullish_harami",
                "gap_up","gap_down","morning_star","evening_star",
                "three_soldiers","three_crows"]
        return {
            **{f"cp_{k}": 0 for k in keys},
            "candle_score": 50.0, "candle_pattern": "없음",
            "bull_patterns": 0,   "bear_patterns": 0,
        }

    def _default_volume_profile(self):
        return {
            "vp_poc": 0.0, "vp_vah": 0.0, "vp_val": 0.0,
            "vp_poc_diff": 0.0, "vp_above_poc": 0,
            "vp_in_va": 0, "vp_score": 50.0,
        }

    def _default_risk_metrics(self):
        return {
            "vol_20d": 0.0, "sharpe_60": 0.0, "mdd_60": 0.0,
            "win_rate_20": 50.0, "atr_stop": 0.0, "atr_target": 0.0,
            "atr_ratio": 1.5, "half_kelly": 5.0, "risk_score": 50.0,
        }

    def _default_indicators(self):
        d = {
            "rsi": 0.0, "macd": 0.0, "macd_signal": 0.0,
            "macd_hist": 0.0, "macd_cross": 0,
            "bb_pct": 0.5, "bb_width": 0.0, "bb_upper": 0.0, "bb_lower": 0.0,
            "ma5": 0.0, "ma20": 0.0, "ma60": 0.0, "momentum_score": 0,
            "mom_5d": 0.0, "mom_20d": 0.0, "mom_60d": 0.0,
            "vb_signal": 0, "atr": 0.0,
            "buy_price": 0.0, "target_price": 0.0,
            "stop_price": 0.0, "expected_return": 3.0,
            "buy_pressure": 0.5, "sell_pressure": 0.5,
            "vol_ratio": 1.0, "pressure_score": 50.0,
            "ob_score": 50.0, "ob_pressure": 1.0,
        }
        d.update(self._default_candle())
        d.update(self._default_volume_profile())
        d.update(self._default_risk_metrics())
        # ★ v7.0 신규
        d.update({
            "stoch_k":50.0,"stoch_d":50.0,"stoch_score":50.0,
            "cci":0.0,"cci_score":50.0,
            "mfi":50.0,"mfi_score":50.0,
            "obv_slope":0.0,"obv_divergence":0,"obv_score":50.0,
            "vwap":0.0,"vwap_deviation":0.0,"vwap_above":0,"vwap_score":50.0,
        })
        return d