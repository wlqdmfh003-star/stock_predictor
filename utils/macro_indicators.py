import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MacroIndicators:
    """
    v5.0 매크로 지표 분석
    - 미국 금리 (10년물 국채)
    - 달러인덱스 (DXY)
    - 원자재 (WTI유가, 금, 구리)
    - 공포탐욕 (VIX)
    - 한국 관련 ETF (EWY)
    """

    TICKERS = {
        "미국10년금리": "^TNX",
        "달러인덱스":   "DX-Y.NYB",
        "WTI유가":      "CL=F",
        "금":           "GC=F",
        "구리":         "HG=F",
        "VIX":          "^VIX",
        "한국ETF":      "EWY",
        "나스닥":        "^IXIC",
        "S&P500":       "^GSPC",
    }

    def fetch(self) -> dict:
        """매크로 데이터 수집"""
        result = {}
        for name, ticker in self.TICKERS.items():
            try:
                raw = yf.Ticker(ticker).history(period="10d")
                if raw is None or len(raw) < 2:
                    result[name] = {"value": 0.0, "change": 0.0, "trend": "중립"}
                    continue
                latest = float(raw["Close"].iloc[-1])
                prev   = float(raw["Close"].iloc[-2])
                change = (latest / prev - 1) * 100 if prev > 0 else 0.0
                # 5일 추세
                trend_val = (latest / float(raw["Close"].iloc[0]) - 1) * 100 if len(raw) >= 5 else 0.0
                trend     = "상승" if trend_val > 1 else "하락" if trend_val < -1 else "중립"
                result[name] = {
                    "value":  round(latest, 4),
                    "change": round(change, 2),
                    "trend":  trend,
                }
            except Exception:
                result[name] = {"value": 0.0, "change": 0.0, "trend": "중립"}
        return result

    def calc_score(self, macro: dict) -> float:
        """
        매크로 지표 → 한국 증시 영향 점수 (0~100, 50=중립)
        """
        score = 50.0

        # ① 미국 금리 (금리 상승 = 위험자산 부담)
        rate_chg = macro.get("미국10년금리", {}).get("change", 0)
        if   rate_chg >  0.05: score -= 10  # 금리 급등
        elif rate_chg >  0.02: score -= 5
        elif rate_chg < -0.05: score += 10  # 금리 급락 (완화 신호)
        elif rate_chg < -0.02: score += 5

        # ② 달러인덱스 (달러 강세 = 외국인 이탈)
        dxy_chg = macro.get("달러인덱스", {}).get("change", 0)
        score  += np.clip(-dxy_chg * 3, -15, 15)

        # ③ WTI 유가 (적당히 오르면 경기 호황, 너무 오르면 부담)
        oil_chg = macro.get("WTI유가", {}).get("change", 0)
        oil_val = macro.get("WTI유가", {}).get("value", 70)
        if   oil_val > 100: score -= 10   # 고유가 부담
        elif oil_val < 50:  score -= 5    # 저유가 = 경기침체 우려
        elif 60 <= oil_val <= 90:
            score += np.clip(oil_chg * 1, -5, 5)

        # ④ 금 (안전자산 선호 시 상승 → 위험자산 하락)
        gold_chg   = macro.get("금", {}).get("change", 0)
        gold_trend = macro.get("금", {}).get("trend", "중립")
        if gold_trend == "상승": score -= 8   # 안전자산 선호
        elif gold_trend == "하락": score += 5

        # ⑤ 구리 (경기 선행지표 - 구리 상승 = 경기 호황)
        copper_chg = macro.get("구리", {}).get("change", 0)
        score     += np.clip(copper_chg * 2, -10, 10)

        # ⑥ VIX (공포지수)
        vix = macro.get("VIX", {}).get("value", 20)
        if   vix >= 30: score -= 20
        elif vix >= 25: score -= 10
        elif vix <= 12: score += 15
        elif vix <= 15: score += 8

        # ⑦ 한국 ETF (EWY) - 외국인의 한국 시장 시각
        ewy_chg = macro.get("한국ETF", {}).get("change", 0)
        score  += np.clip(ewy_chg * 4, -15, 15)

        return float(np.clip(score, 0, 100))

    def apply_to_stocks(self, df: pd.DataFrame, macro: dict) -> pd.DataFrame:
        """종목별 매크로 영향 반영"""
        df    = df.copy()
        base  = self.calc_score(macro)

        oil_val    = macro.get("WTI유가",  {}).get("value",  70)
        gold_trend = macro.get("금",       {}).get("trend",  "중립")
        rate_chg   = macro.get("미국10년금리", {}).get("change", 0)
        copper_chg = macro.get("구리",     {}).get("change", 0)

        macro_scores = []
        for _, row in df.iterrows():
            name  = str(row.get("name", ""))
            score = base

            # 에너지주: 유가 수혜
            if any(kw in name for kw in ["S-Oil","SK이노베이션","GS","한국가스","E1"]):
                score += np.clip((oil_val - 70) * 0.3, -10, 15)

            # 금융주: 금리 상승 수혜
            if any(kw in name for kw in ["KB금융","신한","하나금융","우리금융","기업은행"]):
                score += np.clip(rate_chg * 20, -8, 12)

            # 금 관련주: 금 상승 수혜
            if any(kw in name for kw in ["고려아연","풍산","한국금","영풍"]):
                if gold_trend == "상승": score += 10

            # 철강/소재: 구리 상승 = 경기호황 수혜
            if any(kw in name for kw in ["POSCO","현대제철","고려아연","풍산"]):
                score += np.clip(copper_chg * 2, -8, 10)

            # 내수주: 달러 영향 적음 (방어적)
            if any(kw in name for kw in ["롯데","신세계","이마트","CJ","하이트"]):
                score = (score + 50) / 2   # 중립 방향으로

            macro_scores.append(float(np.clip(score, 0, 100)))

        df["macro_score"] = macro_scores
        return df

    def get_summary(self, macro: dict) -> dict:
        """매크로 환경 요약"""
        score = self.calc_score(macro)
        if   score >= 65: env = "🟢 우호적"
        elif score >= 45: env = "🟡 중립"
        else:             env = "🔴 부정적"
        return {
            "score":  round(score, 1),
            "env":    env,
            "detail": macro,
        }