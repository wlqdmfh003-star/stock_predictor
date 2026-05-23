import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MarketPhase:
    """
    시장 국면 감지 v6.0
    ★ 5단계 국면 (극단적공포/공포/중립/탐욕/극단적탐욕)
    ★ 공포·탐욕 지수 (Fear & Greed Index) 자체 계산
    ★ VIX 근사값 (변동성 지수)
    ★ 시장 모멘텀 + RSI + 이평선 + 거래량 종합
    ★ 국면별 최적 가중치 5단계 분리
    """

    # ── 5단계 국면별 최적 가중치 ─────────────────────────────────────────────
    WEIGHTS = {
        "극단적탐욕": {
            "lstm":0.20,"ensemble":0.12,"momentum":0.20,"sentiment":0.10,
            "institution":0.10,"volume":0.08,"fundamental":0.06,
            "dart":0.05,"short":0.02,"high52":0.10,"us_market":0.04,"sector":0.03,
        },
        "탐욕": {
            "lstm":0.18,"ensemble":0.10,"momentum":0.18,"sentiment":0.10,
            "institution":0.10,"volume":0.08,"fundamental":0.08,
            "dart":0.06,"short":0.03,"high52":0.08,"us_market":0.05,"sector":0.04,
        },
        "중립": {
            "lstm":0.13,"ensemble":0.10,"momentum":0.13,"sentiment":0.10,
            "institution":0.10,"volume":0.08,"fundamental":0.12,
            "dart":0.08,"short":0.06,"high52":0.05,"us_market":0.07,"sector":0.06,
        },
        "공포": {
            "lstm":0.10,"ensemble":0.08,"momentum":0.08,"sentiment":0.08,
            "institution":0.13,"volume":0.08,"fundamental":0.18,
            "dart":0.10,"short":0.13,"high52":0.02,"us_market":0.08,"sector":0.04,
        },
        "극단적공포": {
            "lstm":0.08,"ensemble":0.06,"momentum":0.05,"sentiment":0.07,
            "institution":0.15,"volume":0.07,"fundamental":0.22,
            "dart":0.08,"short":0.15,"high52":0.01,"us_market":0.06,"sector":0.00,
        },
    }

    # 기존 호환용
    WEIGHTS_BULL = WEIGHTS["탐욕"]
    WEIGHTS_BEAR = WEIGHTS["공포"]
    WEIGHTS_SIDE = WEIGHTS["중립"]

    def detect(self, market: str = "KOSPI") -> tuple:
        """
        반환: (국면명, 점수, 가중치dict, 상세정보dict)
        """
        try:
            ticker = "^KS11" if "KOSPI" in market else "^KQ11"
            ohlcv  = self._fetch_index(ticker)

            if ohlcv is None or len(ohlcv) < 20:
                return "중립", 50.0, self.WEIGHTS["중립"], self._empty_detail()

            close  = ohlcv["close"].astype(float)
            volume = ohlcv["volume"].astype(float) if "volume" in ohlcv.columns \
                     else pd.Series(np.ones(len(close)))

            detail = self._calc_detail(close, volume)
            score  = detail["fg_index"]
            phase  = self._score_to_phase(score)

            return phase, float(score), self.WEIGHTS[phase], detail

        except Exception:
            return "중립", 50.0, self.WEIGHTS["중립"], self._empty_detail()

    def _fetch_index(self, ticker: str):
        try:
            import yfinance as yf
            end   = datetime.now()
            start = end - timedelta(days=200)
            df    = yf.download(ticker, start=start, end=end,
                                progress=False, auto_adjust=True)
            if df is None or len(df) == 0:
                return None
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                         for c in df.columns]
            return df
        except Exception:
            return None

    def _calc_detail(self, close: pd.Series, volume: pd.Series) -> dict:
        """공포·탐욕 지수 및 세부 지표 계산"""
        scores = {}

        # ── 1. 모멘텀 (20일/60일) ────────────────────────────────────────────
        if len(close) >= 21:
            mom20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100)
            scores["momentum"] = float(np.clip(50 + mom20 * 3, 0, 100))
        else:
            scores["momentum"] = 50.0

        # ── 2. RSI ────────────────────────────────────────────────────────────
        if len(close) >= 15:
            d   = close.diff().dropna()
            g   = d.where(d > 0, 0).rolling(14).mean()
            l   = (-d.where(d < 0, 0)).rolling(14).mean()
            rsi = float(100 - 100 / (1 + g.iloc[-1] / (l.iloc[-1] + 1e-9)))
            scores["rsi"] = float(np.clip(rsi, 0, 100))
        else:
            scores["rsi"] = 50.0

        # ── 3. 이평선 정배열 ─────────────────────────────────────────────────
        ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else float(close.mean())
        ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else ma20
        cur  = float(close.iloc[-1])
        if cur > ma20 > ma60:
            scores["ma"] = 75.0
        elif cur > ma20:
            scores["ma"] = 60.0
        elif cur < ma20 < ma60:
            scores["ma"] = 25.0
        else:
            scores["ma"] = 40.0

        # ── 4. 거래량 추세 ────────────────────────────────────────────────────
        if len(volume) >= 20:
            vm20 = float(volume.rolling(20).mean().iloc[-1])
            vm5  = float(volume.rolling(5).mean().iloc[-1])
            vol_ratio = vm5 / (vm20 + 1e-9)
            scores["volume"] = float(np.clip(50 + (vol_ratio - 1) * 30, 10, 90))
        else:
            scores["volume"] = 50.0

        # ── 5. 52주 신고가 비율 ───────────────────────────────────────────────
        if len(close) >= 52:
            hi52 = float(close.rolling(252).max().iloc[-1]) if len(close) >= 252 \
                   else float(close.max())
            lo52 = float(close.rolling(252).min().iloc[-1]) if len(close) >= 252 \
                   else float(close.min())
            pos52 = (cur - lo52) / (hi52 - lo52 + 1e-9)
            scores["high52"] = float(pos52 * 100)
        else:
            scores["high52"] = 50.0

        # ── 6. VIX 근사 (단기/장기 변동성 비율) ──────────────────────────────
        if len(close) >= 30:
            rets     = close.pct_change().dropna()
            vol_5d   = float(rets.rolling(5).std().iloc[-1]  * np.sqrt(252) * 100)
            vol_20d  = float(rets.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
            vix_approx = vol_5d  # 단기 변동성 = VIX 근사
            # 변동성 높을수록 공포
            vix_score  = float(np.clip(100 - vix_approx * 2, 10, 90))
            scores["vix"] = vix_score
        else:
            vix_approx = 20.0
            scores["vix"] = 50.0

        # ── 공포·탐욕 지수 종합 ───────────────────────────────────────────────
        weights = {
            "momentum": 0.25,
            "rsi":      0.20,
            "ma":       0.20,
            "volume":   0.10,
            "high52":   0.15,
            "vix":      0.10,
        }
        fg_index = sum(scores.get(k, 50) * v for k, v in weights.items())

        return {
            "fg_index":   round(float(fg_index), 1),
            "momentum":   round(scores["momentum"], 1),
            "rsi":        round(scores["rsi"], 1),
            "ma_score":   round(scores["ma"], 1),
            "vol_score":  round(scores["volume"], 1),
            "high52":     round(scores["high52"], 1),
            "vix_approx": round(vix_approx, 1),
            "vix_score":  round(scores["vix"], 1),
        }

    def _score_to_phase(self, score: float) -> str:
        if   score >= 75: return "극단적탐욕"
        elif score >= 60: return "탐욕"
        elif score >= 40: return "중립"
        elif score >= 25: return "공포"
        else:             return "극단적공포"

    def _empty_detail(self) -> dict:
        return {"fg_index":50.0,"momentum":50.0,"rsi":50.0,
                "ma_score":50.0,"vol_score":50.0,"high52":50.0,
                "vix_approx":20.0,"vix_score":50.0}

    def get_phase_description(self, phase: str) -> dict:
        desc = {
            "극단적탐욕": {
                "icon":"🔥","color":"#ef4444",
                "strategy":"과매수 주의 — 일부 차익실현 고려",
                "tip":"거품 가능성 있음, 리스크 관리 철저",
            },
            "탐욕": {
                "icon":"📈","color":"#10b981",
                "strategy":"적극 매수 — 모멘텀·LSTM 중심",
                "tip":"52주 신고가 + 거래량 급증 종목 집중",
            },
            "중립": {
                "icon":"📊","color":"#f59e0b",
                "strategy":"선택적 매수 — 실적 서프라이즈 중심",
                "tip":"섹터 로테이션 활용, 리스크 관리 철저",
            },
            "공포": {
                "icon":"📉","color":"#f97316",
                "strategy":"방어적 — 저PBR 배당주 중심",
                "tip":"공매도 낮고 재무 안정 종목만 선택",
            },
            "극단적공포": {
                "icon":"💀","color":"#7f1d1d",
                "strategy":"역발상 매수 기회 — 극소량만",
                "tip":"공포 극대화 = 반등 기회, 분할 매수",
            },
        }
        return desc.get(phase, desc["중립"])