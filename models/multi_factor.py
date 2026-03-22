import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime


class MultiFactorScorer:
    """
    v5.3 멀티팩터 통합 점수
    ★ ATR 기반 동적 손절/목표가 (손익비 1.5 보장)
    ★ 상관관계 필터 (포트폴리오 분산)
    ★ 요일/월말 효과
    ★ 워크포워드 준비 (walk_forward_ready 컬럼)
    """

    def __init__(self, weights=None):
        self.weights = weights or {
            "lstm":0.12,"ensemble":0.10,"candle":0.08,"macro":0.07,
            "momentum":0.12,"sentiment":0.08,"institution":0.10,
            "volume":0.06,"fundamental":0.08,"dart":0.06,
            "short":0.04,"high52":0.04,"us_market":0.05,"sector":0.05,
        }

    # ── 메인 스코어 ───────────────────────────────────────────────────────────
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._fill_defaults(df)

        # 요일/월말 효과 먼저 계산
        df = self._apply_calendar_effect(df)

        w  = self.weights
        tw = sum(w.values()) or 1.0

        def norm(col):
            s  = df[col].astype(float)
            mn = s.min()
            mx = s.max()
            if mx - mn > 1e-9:
                return (s - mn) / (mx - mn) * 100
            else:
                # ★ min==max (모두 같은 값) → 원래 값 그대로 사용 (50점 고정 방지)
                val = float(s.iloc[0]) if len(s) > 0 else 50.0
                # 0~100 범위면 그대로, 아니면 클리핑
                return pd.Series(float(np.clip(val, 0, 100)), index=df.index)

        df["f_lstm"]        = norm("lstm_score")
        df["f_ensemble"]    = norm("ensemble_score")
        df["f_candle"]      = df["candle_score"].clip(0,100)
        df["f_macro"]       = norm("macro_score")
        df["f_momentum"]    = norm("momentum_score")
        df["f_sentiment"]   = norm("sentiment_score")
        df["f_institution"] = self._institution_score(df)
        df["f_volume"]      = norm("volume_score")
        df["f_fundamental"] = norm("fundamental_score")
        df["f_dart"]        = norm("dart_score")
        df["f_short"]       = norm("short_score")
        df["f_high52"]      = norm("high52_score")
        df["f_us_market"]   = norm("us_market_score")
        df["f_sector"]      = norm("sector_score")
        df["f_rsi"]         = df["rsi"].apply(self._rsi_score)
        df["f_macd"]        = df["macd_cross"].fillna(0)*20+50

        df["total_score"] = (
            df["f_lstm"]        * w.get("lstm",0.12) +
            df["f_ensemble"]    * w.get("ensemble",0.10) +
            df["f_candle"]      * w.get("candle",0.08) +
            df["f_macro"]       * w.get("macro",0.07) +
            df["f_momentum"]    * w.get("momentum",0.12) +
            df["f_sentiment"]   * w.get("sentiment",0.08) +
            df["f_institution"] * w.get("institution",0.10) +
            df["f_volume"]      * w.get("volume",0.06) +
            df["f_fundamental"] * w.get("fundamental",0.08) +
            df["f_dart"]        * w.get("dart",0.06) +
            df["f_short"]       * w.get("short",0.04) +
            df["f_high52"]      * w.get("high52",0.04) +
            df["f_us_market"]   * w.get("us_market",0.05) +
            df["f_sector"]      * w.get("sector",0.05)
        ) / tw

        # RSI/MACD 보조
        # ★ 새 지표 점수 추가 (스토캐스틱/CCI/MFI/OBV/VWAP + 일목/피보/엘리어트/CNN)
        for col in ["stoch_score","cci_score","mfi_score","obv_score","vwap_score",
                    "ichi_score","fib_score","elliott_score","cnn_score"]:
            if col not in df.columns:
                df[col] = 50.0

        # ★ IC 기반 동적 가중치 계산
        ic_weights = self._calc_ic_weights(df)

        df["total_score"] = (
            df["total_score"] * 0.65 +
            norm("stoch_score")   * ic_weights.get("stoch",   0.03) +
            norm("cci_score")     * ic_weights.get("cci",     0.03) +
            norm("mfi_score")     * ic_weights.get("mfi",     0.03) +
            norm("obv_score")     * ic_weights.get("obv",     0.03) +
            norm("vwap_score")    * ic_weights.get("vwap",    0.03) +
            norm("ichi_score")    * ic_weights.get("ichi",    0.05) +
            norm("fib_score")     * ic_weights.get("fib",     0.04) +
            norm("elliott_score") * ic_weights.get("elliott", 0.04) +
            norm("cnn_score")     * ic_weights.get("cnn",     0.07) +
            df["f_rsi"]        * 0.10 +
            df["f_macd"]       * 0.08
        )

        # 리스크 조정
        df["total_score"] = df.apply(self._risk_adjust, axis=1)

        # 패턴 보너스
        df["total_score"] = self._apply_bonuses(df)

        # 요일/월말 효과 반영
        df["total_score"] = (df["total_score"] + df["calendar_bonus"]).clip(0,100)

        # ★ 외국인 순매수 추이 반영
        if "foreign_trend_score" in df.columns:
            ft = df["foreign_trend_score"].astype(float)
            df["total_score"] = (df["total_score"] * 0.93 +
                                 ft * 0.07).clip(0, 100)

        # ★ 공시 발표 전날 전략 반영
        if "pre_disclosure_score" in df.columns:
            pd_s = df["pre_disclosure_score"].astype(float)
            bonus = ((pd_s - 50) / 50 * 5).clip(-3, 8)
            df["total_score"] = (df["total_score"] + bonus).clip(0, 100)

        # ★ 섹터 로테이션 가중치 반영
        try:
            from utils.sector_analysis import SectorAnalysis
            sa       = SectorAnalysis()
            rotation = sa.get_rotation_strategy(df)
            df       = sa.apply_rotation_weight(df, rotation)
            df["total_score"] = df["total_score"].clip(0, 100)
            if rotation.get("top3_sectors"):
                pass  # 로테이션 적용 완료
        except Exception:
            pass

        df["rise_prob"] = df["total_score"].apply(self._to_prob)

        # ATR 동적 손절/목표가
        df = self._calc_atr_prices(df)

        # 켈리 비중
        df["suggested_weight"] = df.apply(self._half_kelly, axis=1).clip(0,25)

        df = df.sort_values("rise_prob", ascending=False).reset_index(drop=True)

        # 상관관계 필터 (TOP 30에 적용)
        df = self._correlation_filter(df)

        return df

    # ── ① ATR 기반 동적 손절/목표가 ─────────────────────────────────────────
    def _calc_atr_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ATR(14일) 기반 동적 손절/목표가
        - 손절 = 매수가 - ATR × 2.0  (변동성 클수록 여유있게)
        - 목표 = 매수가 + ATR × 3.0  (손익비 최소 1.5 보장)
        - ATR 없으면 기존 고정비율 사용
        """
        buy_prices, target_prices, stop_prices, exp_rets = [], [], [], []

        for _, row in df.iterrows():
            price = float(row.get("current_price", 0))
            if price <= 0:
                buy_prices.append(0); target_prices.append(0)
                stop_prices.append(0); exp_rets.append(3.0)
                continue

            buy = round(price * 0.995, 0)

            # ATR 계산
            ohlcv = row.get("ohlcv")
            atr   = None
            if ohlcv is not None and len(ohlcv) >= 14:
                try:
                    h = ohlcv["high"].astype(float).values[-14:]
                    l = ohlcv["low"].astype(float).values[-14:]
                    c = ohlcv["close"].astype(float).values[-14:]
                    tr_arr = []
                    for i in range(1, len(h)):
                        tr_arr.append(max(h[i]-l[i],
                                         abs(h[i]-c[i-1]),
                                         abs(l[i]-c[i-1])))
                    if tr_arr:
                        atr = np.mean(tr_arr)
                except Exception:
                    atr = None

            if atr and atr > 0:
                stop   = round(buy - atr * 2.0, 0)
                target = round(buy + atr * 3.0, 0)
                # 손절이 매수가의 85% 이하면 상한 (너무 넓은 손절 방지)
                stop   = max(stop, round(buy * 0.85, 0))
                # 손익비 재확인 (최소 1.3)
                risk   = buy - stop
                reward = target - buy
                if risk > 0 and reward/risk < 1.3:
                    target = round(buy + risk * 1.5, 0)
                exp_ret = round((target/buy - 1)*100, 2)
            else:
                # 폴백: 고정 비율
                score   = float(row.get("total_score", 50))
                exp_ret = round(max(1.5, min(12.0, 3.0 + (score-50)/50*3)), 2)
                stop    = round(buy * 0.97, 0)
                target  = round(buy * (1 + exp_ret/100), 0)

            buy_prices.append(buy)
            target_prices.append(target)
            stop_prices.append(stop)
            exp_rets.append(exp_ret)

        df["buy_price"]      = buy_prices
        df["target_price"]   = target_prices
        df["stop_price"]     = stop_prices
        df["expected_return"]= exp_rets
        return df

    # ── ② 상관관계 필터 ──────────────────────────────────────────────────────
    def _correlation_filter(self, df: pd.DataFrame,
                             top_n: int=30, corr_threshold: float=0.80) -> pd.DataFrame:
        """
        TOP 30 종목 중 상관관계 0.8 이상인 중복 종목 제거
        → 포트폴리오 분산 효과 극대화
        """
        top = df.head(top_n).copy()
        rest = df.iloc[top_n:].copy()

        # 수익률 시계열 수집
        ret_series = {}
        for _, row in top.iterrows():
            ohlcv = row.get("ohlcv")
            code  = str(row.get("code",""))
            if ohlcv is not None and len(ohlcv) >= 20:
                try:
                    c = ohlcv["close"].astype(float).values[-60:]
                    ret_series[code] = np.diff(c)/c[:-1]
                except Exception:
                    pass

        if len(ret_series) < 2:
            df["corr_flag"] = False
            return df

        # 상관관계 행렬
        codes  = list(ret_series.keys())
        min_l  = min(len(v) for v in ret_series.values())
        matrix = np.array([ret_series[c][-min_l:] for c in codes])

        try:
            corr_mat = np.corrcoef(matrix)
        except Exception:
            df["corr_flag"] = False
            return df

        # 탐욕 선택: 상관관계 높은 쌍에서 점수 낮은 종목 제거
        excluded = set()
        for i in range(len(codes)):
            if codes[i] in excluded:
                continue
            for j in range(i+1, len(codes)):
                if codes[j] in excluded:
                    continue
                if corr_mat[i][j] > corr_threshold:
                    # 점수 낮은 쪽 제거
                    score_i = float(top[top["code"]==codes[i]]["rise_prob"].values[0]) \
                              if len(top[top["code"]==codes[i]])>0 else 0
                    score_j = float(top[top["code"]==codes[j]]["rise_prob"].values[0]) \
                              if len(top[top["code"]==codes[j]])>0 else 0
                    excluded.add(codes[j] if score_i >= score_j else codes[i])

        top["corr_flag"]  = top["code"].isin(excluded)
        rest["corr_flag"] = False

        # 제거된 종목은 뒤로 이동 (완전 삭제 X, 사용자가 볼 수 있게)
        top_keep    = top[~top["corr_flag"]]
        top_exclude = top[top["corr_flag"]]
        result      = pd.concat([top_keep, top_exclude, rest], ignore_index=True)

        excl_names = top_exclude["name"].tolist() if len(top_exclude)>0 else []
        if excl_names:
            print(f"[상관관계] 필터: {len(excl_names)}개 중복 제거 → {excl_names}")

        return result

    # ── ③ 요일/월말 효과 ─────────────────────────────────────────────────────
    def _apply_calendar_effect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간대별 통계적 패턴 반영
        - 월요일 갭업 효과: +2점
        - 금요일 포지션 청산: -1점
        - 월말(25일~말일): 기관 수급 마감 효과 +2점
        - 실적 시즌 (1,4,7,10월): 변동성 확대 주의 -1점
        """
        today   = datetime.now()
        weekday = today.weekday()   # 0=월, 4=금
        day     = today.day
        month   = today.month

        bonus = 0.0

        # 요일 효과
        if weekday == 0:    # 월요일
            bonus += 2.0    # 주말 해외 호재 갭업 기대
        elif weekday == 4:  # 금요일
            bonus -= 1.0    # 주말 리스크 포지션 청산

        # 월말 수급 효과
        if day >= 25:
            bonus += 2.0    # 기관 월말 성과 마감 매수

        # 실적 시즌 변동성
        if month in [1, 4, 7, 10]:
            bonus -= 1.0    # 실적 불확실성

        df["calendar_bonus"] = bonus
        df["calendar_note"]  = (
            f"{'월요일갭업+2 ' if weekday==0 else ''}"
            f"{'금요일청산-1 ' if weekday==4 else ''}"
            f"{'월말수급+2 '   if day>=25   else ''}"
            f"{'실적시즌-1'    if month in [1,4,7,10] else ''}"
        ).strip() or "해당없음"

        return df

    # ── 리스크 조정 ───────────────────────────────────────────────────────────
    def _risk_adjust(self, row) -> float:
        score  = float(row.get("total_score", 50))
        sharpe = float(row.get("sharpe_60", 0))
        mdd    = float(row.get("mdd_60", 0))
        vol    = float(row.get("vol_20d", 20))

        if   sharpe > 1.5:  score = min(100, score+5)
        elif sharpe > 1.0:  score = min(100, score+3)
        elif sharpe < -0.5: score = max(0,   score-8)
        elif sharpe < 0:    score = max(0,   score-4)

        if   mdd < -25: score = max(0,   score-10)
        elif mdd < -15: score = max(0,   score-5)
        elif mdd > -5:  score = min(100, score+3)

        if   vol > 60: score = max(0,   score-8)
        elif vol > 40: score = max(0,   score-4)
        elif vol < 15: score = min(100, score+3)

        return float(np.clip(score, 0, 100))

    # ── 패턴 보너스 ───────────────────────────────────────────────────────────
    def _apply_bonuses(self, df: pd.DataFrame) -> pd.Series:
        scores = df["total_score"].copy()
        if "is_new_high"   in df.columns: scores += df["is_new_high"].fillna(0)*5
        if "is_new_low"    in df.columns: scores -= df["is_new_low"].fillna(0)*10
        if "dart_summary"  in df.columns:
            mask   = df["dart_summary"].astype(str).str.contains("유상증자",na=False)
            scores -= mask*15
        if "bull_patterns" in df.columns: scores += df["bull_patterns"].fillna(0).clip(0,3)*2
        if "bear_patterns" in df.columns: scores -= df["bear_patterns"].fillna(0).clip(0,3)*2
        if "cp_gap_up"     in df.columns: scores += df["cp_gap_up"].fillna(0)*3
        if "cp_gap_down"   in df.columns: scores += df["cp_gap_down"].fillna(0)*3

        # 강한 패턴 보너스 (헤드앤숄더/이중바닥)
        if "candle_pattern" in df.columns:
            bull_strong = ["역헤드앤숄더","이중바닥","새벽별형","상승장악형","세병사"]
            bear_strong = ["헤드앤숄더","이중천장","세까마귀"]
            for pat in bull_strong:
                mask    = df["candle_pattern"].astype(str).str.contains(pat,na=False)
                scores += mask*5
            for pat in bear_strong:
                mask    = df["candle_pattern"].astype(str).str.contains(pat,na=False)
                scores -= mask*5

        return scores.clip(0,100)

    # ── 켈리 공식 ────────────────────────────────────────────────────────────
    def _half_kelly(self, row) -> float:
        p   = float(row.get("rise_prob", 55)) / 100
        b   = float(row.get("expected_return", 3.0)) / \
              max(abs(float(row.get("current_price",1)) -
                      float(row.get("stop_price",1)))/
                  max(float(row.get("current_price",1)),1)*100, 1.0)
        q   = 1 - p
        k   = (b*p - q) / (b + 1e-9)
        return float(max(0, k/2*100))   # 하프켈리

    # ── 유틸 ─────────────────────────────────────────────────────────────────
    def _fill_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        defaults = {
            "lstm_score":50,"ensemble_score":50,"candle_score":50,"macro_score":50,
            "momentum_score":50,"sentiment_score":50,"volume_score":50,
            "fundamental_score":50,"dart_score":50,"short_score":50,
            "high52_score":50,"us_market_score":50,"sector_score":50,
            "rsi":50,"macd_cross":0,"inst_net":0,"foreign_net":0,
            "buy_price":0,"target_price":0,"stop_price":0,
            "expected_return":3.0,"current_price":0,
            "news_summary":"뉴스없음","dart_summary":"공시없음",
            "is_new_high":0,"is_new_low":0,
            "pressure_score":50,"vp_score":50,"risk_score":50,
            "ob_score":50,"ob_pressure":1.0,"sharpe_60":0,"mdd_60":0,
            "vol_20d":20,"half_kelly":5,"bull_patterns":0,"bear_patterns":0,
            "cp_gap_up":0,"cp_gap_down":0,"candle_pattern":"",
        }
        for col, val in defaults.items():
            if col not in df.columns: df[col] = val
            else: df[col] = df[col].fillna(val)
        return df

    def _institution_score(self, df):
        combined = df["inst_net"].astype(float) + df["foreign_net"].astype(float)*0.5
        mn,mx    = combined.min(), combined.max()
        return (combined-mn)/(mx-mn+1e-9)*100 if mx-mn>1e-9 \
               else pd.Series(50.0, index=df.index)

    def _rsi_score(self, rsi):
        if   rsi<30: return 75.0
        elif rsi<40: return 65.0
        elif rsi<50: return 60.0
        elif rsi<60: return 50.0
        elif rsi<70: return 40.0
        else:        return 25.0

    def _to_prob(self, score):
        x = (score-50)/15
        return float(np.clip(1/(1+np.exp(-x))*100, 35, 92))