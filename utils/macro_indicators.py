import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MacroIndicators:
    """
    v7.0 매크로 지표 분석
    - 미국 금리 (10년물 국채)
    - 달러인덱스 (DXY)
    - 원자재 (WTI유가, 금, 구리)
    - 공포탐욕 (VIX 실제값)
    ★ 신규: 공포탐욕 지수 5단계 (극단적공포~극단적탐욕)
    ★ 신규: VIX + S&P500 + 나스닥 + 금 + 달러 종합 계산
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
        """종목별 매크로 영향 반영 — 섹터별 차별화 강화"""
        df    = df.copy()
        base  = self.calc_score(macro)

        oil_val    = macro.get("WTI유가",     {}).get("value",  70)
        gold_trend = macro.get("금",          {}).get("trend",  "중립")
        gold_chg   = macro.get("금",          {}).get("change", 0)
        rate_chg   = macro.get("미국10년금리",{}).get("change", 0)
        rate_val   = macro.get("미국10년금리",{}).get("value",  4.0)
        copper_chg = macro.get("구리",        {}).get("change", 0)
        dxy_chg    = macro.get("달러인덱스",  {}).get("change", 0)
        vix        = macro.get("VIX",         {}).get("value",  20)
        ewy_chg    = macro.get("한국ETF",     {}).get("change", 0)

        macro_scores = []
        for _, row in df.iterrows():
            name   = str(row.get("name", ""))
            sector = str(row.get("sector", "기타"))
            score  = base

            # ★ 섹터별 차별화 (점수가 다양하게 나오도록)
            # 반도체: 달러 강세 + 미국 증시 영향
            if sector == "반도체" or any(kw in name for kw in ["삼성전자","SK하이닉스","DB하이텍"]):
                score += np.clip(ewy_chg * 3, -10, 12)
                score += np.clip(-dxy_chg * 2, -8, 8)

            # 2차전지: 구리/원자재 영향
            elif sector == "2차전지" or any(kw in name for kw in ["LG에너지","에코프로","포스코퓨처엠"]):
                score += np.clip(copper_chg * 3, -10, 12)
                score += np.clip(-rate_chg * 15, -10, 8)

            # 에너지주: 유가 수혜
            elif sector == "에너지" or any(kw in name for kw in ["S-Oil","SK이노베이션","GS","한국가스","E1"]):
                score += np.clip((oil_val - 70) * 0.4, -12, 18)

            # 금융주: 금리 상승 수혜
            elif sector == "금융" or any(kw in name for kw in ["KB금융","신한","하나금융","우리금융","기업은행"]):
                score += np.clip(rate_chg * 25, -10, 15)
                if rate_val > 4.5: score += 5

            # 철강/소재: 구리 + 경기
            elif sector == "철강/소재" or any(kw in name for kw in ["POSCO","현대제철","고려아연","풍산"]):
                score += np.clip(copper_chg * 2.5, -10, 12)
                if gold_trend == "상승": score += 8

            # 바이오: VIX 영향 (변동성 낮을수록 유리)
            elif sector == "바이오/제약":
                if vix < 15:   score += 10
                elif vix > 25: score -= 8

            # 방산: 지정학적 리스크 (VIX 상승 시 수혜)
            elif sector == "방산":
                if vix > 25: score += 12
                elif vix > 20: score += 6

            # 자동차: 달러 강세 수혜 (수출)
            elif sector == "자동차/부품" or any(kw in name for kw in ["현대차","기아","현대모비스"]):
                score += np.clip(dxy_chg * 3, -8, 12)

            # 내수주: 달러 영향 적음 (방어적)
            elif sector in ["유통/소비","건설/부동산"] or                  any(kw in name for kw in ["롯데","신세계","이마트","CJ","하이트"]):
                score = (score * 0.6 + 50 * 0.4)  # 중립 방향으로

            # IT/플랫폼: 금리 민감
            elif sector == "IT/플랫폼":
                score += np.clip(-rate_chg * 20, -12, 10)

            # 조선: 달러 강세 + 경기
            elif sector == "조선":
                score += np.clip(dxy_chg * 2, -6, 10)
                score += np.clip(copper_chg * 1.5, -6, 8)

            # 화학: 유가 + 달러
            elif sector == "화학":
                score += np.clip((oil_val - 70) * 0.2, -8, 10)
                score += np.clip(-dxy_chg * 1.5, -6, 8)

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

    # ══════════════════════════════════════════════════════════════
    # ★ 공포탐욕 지수 + VIX 실제값
    # ══════════════════════════════════════════════════════════════
    def get_fear_greed(self, macro: dict) -> dict:
        """
        공포탐욕 지수 계산
        - VIX 실제값 (^VIX yfinance)
        - 시장 모멘텀 (S&P500 52주 대비)
        - 안전자산 선호 (금 vs 주식)
        - 달러 강세 여부
        0~25: 극단적 공포 / 25~45: 공포 / 45~55: 중립
        55~75: 탐욕 / 75~100: 극단적 탐욕
        """
        scores = {}

        # ① VIX (공포지수 역방향)
        vix = macro.get("VIX", {}).get("value", 20)
        if   vix >= 40:  scores["vix"] = 5    # 극단적 공포
        elif vix >= 30:  scores["vix"] = 20
        elif vix >= 25:  scores["vix"] = 35
        elif vix >= 20:  scores["vix"] = 50
        elif vix >= 15:  scores["vix"] = 65
        elif vix >= 12:  scores["vix"] = 80
        else:            scores["vix"] = 95   # 극단적 탐욕

        # ② S&P500 모멘텀
        sp_chg   = macro.get("S&P500", {}).get("change", 0)
        sp_trend = macro.get("S&P500", {}).get("trend",  "중립")
        if   sp_trend == "상승" and sp_chg > 0: scores["sp_mom"] = 75
        elif sp_trend == "상승":                scores["sp_mom"] = 60
        elif sp_trend == "하락" and sp_chg < 0: scores["sp_mom"] = 25
        elif sp_trend == "하락":               scores["sp_mom"] = 40
        else:                                  scores["sp_mom"] = 50

        # ③ 나스닥 모멘텀
        nq_chg = macro.get("나스닥", {}).get("change", 0)
        scores["nq_mom"] = float(np.clip(50 + nq_chg * 5, 10, 90))

        # ④ 금 vs 주식 (안전자산 선호)
        gold_trend = macro.get("금", {}).get("trend", "중립")
        if   gold_trend == "상승": scores["safe_haven"] = 30  # 안전자산 선호 → 공포
        elif gold_trend == "하락": scores["safe_haven"] = 70  # 위험자산 선호 → 탐욕
        else:                      scores["safe_haven"] = 50

        # ⑤ 달러 강세 (달러 강세 → 신흥국 공포)
        dxy_chg = macro.get("달러인덱스", {}).get("change", 0)
        scores["dollar"] = float(np.clip(50 - dxy_chg * 10, 10, 90))

        # 종합 점수 (가중 평균)
        weights = {"vix":0.30,"sp_mom":0.25,"nq_mom":0.20,
                   "safe_haven":0.15,"dollar":0.10}
        fg_score = sum(scores.get(k, 50)*v for k, v in weights.items())

        # 단계 분류
        if   fg_score >= 75: phase = "극단적탐욕"; icon = "🔥"
        elif fg_score >= 55: phase = "탐욕";       icon = "😀"
        elif fg_score >= 45: phase = "중립";       icon = "😐"
        elif fg_score >= 25: phase = "공포";       icon = "😨"
        else:                phase = "극단적공포"; icon = "💀"

        return {
            "score":      round(float(fg_score), 1),
            "phase":      phase,
            "icon":       icon,
            "vix":        round(float(vix), 1),
            "components": {k: round(v,1) for k,v in scores.items()},
            "description": f"{icon} {phase} ({fg_score:.0f}점) | VIX={vix:.1f}",
        }