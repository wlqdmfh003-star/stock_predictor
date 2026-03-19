import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class USMarket:
    """
    v5.0 미국 증시 연동
    - 나스닥/S&P500/다우/환율/VIX/SOX 분석
    - 오버나이트 갭 분석: 전날 종가 대비 당일 시가 갭 계산
    - 반도체 종목에 SOX 갭 추가 반영
    """

    INDICES = {
        "나스닥":        "^IXIC",
        "S&P500":       "^GSPC",
        "다우":          "^DJI",
        "VIX":          "^VIX",
        "원달러":        "USDKRW=X",
        "필라델피아반도체": "^SOX",
    }

    # 반도체 관련 종목 키워드
    SEMI_KEYWORDS = ["삼성전자","SK하이닉스","DB하이텍","하나마이크론","리노공업",
                     "원익IPS","피에스케이","HPSP","에스앤에스텍"]
    # 수출 대형주 (환율 수혜)
    EXPORT_KEYWORDS = ["삼성전자","현대차","기아","LG전자","SK하이닉스","현대모비스"]
    # IT/플랫폼 (나스닥 연동)
    IT_KEYWORDS = ["카카오","NAVER","네이버","크래프톤","넥슨","엔씨소프트"]

    def __init__(self):
        self.start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    def fetch(self) -> dict:
        """미국 증시 데이터 수집 + 오버나이트 갭 계산"""
        result = {}
        for name, ticker in self.INDICES.items():
            try:
                raw = yf.Ticker(ticker).history(period="5d")
                if raw is None or len(raw) < 2:
                    result[name] = {"change": 0.0, "value": 0.0, "gap": 0.0}
                    continue

                latest = float(raw["Close"].iloc[-1])
                prev   = float(raw["Close"].iloc[-2])
                change = (latest / prev - 1) * 100 if prev > 0 else 0.0

                # 오버나이트 갭: 전날 종가 vs 오늘 시가
                gap = 0.0
                if "Open" in raw.columns and len(raw) >= 2:
                    today_open = float(raw["Open"].iloc[-1])
                    prev_close = float(raw["Close"].iloc[-2])
                    gap = (today_open / prev_close - 1) * 100 if prev_close > 0 else 0.0

                result[name] = {
                    "value":  round(latest, 2),
                    "change": round(change, 2),
                    "gap":    round(gap, 2),     # ← 오버나이트 갭
                }
            except Exception:
                result[name] = {"change": 0.0, "value": 0.0, "gap": 0.0}
        return result

    def calc_score(self, us_data: dict) -> float:
        """미국 증시 → 한국 주식 기본 점수 (0~100, 50=중립)"""
        score = 50.0

        nasdaq_chg = us_data.get("나스닥",   {}).get("change", 0)
        nasdaq_gap = us_data.get("나스닥",   {}).get("gap",    0)
        sp_chg     = us_data.get("S&P500",  {}).get("change", 0)
        vix        = us_data.get("VIX",     {}).get("value",  20)
        usd_chg    = us_data.get("원달러",   {}).get("change", 0)
        sox_chg    = us_data.get("필라델피아반도체", {}).get("change", 0)
        sox_gap    = us_data.get("필라델피아반도체", {}).get("gap",    0)

        # 나스닥 등락 + 갭
        score += np.clip(nasdaq_chg * 4, -20, 20)
        score += np.clip(nasdaq_gap * 3, -12, 12)   # 갭 반영

        # S&P500
        score += np.clip(sp_chg * 3, -15, 15)

        # VIX 공포지수
        if   vix >= 30: score -= 20
        elif vix >= 25: score -= 10
        elif vix <= 12: score += 15
        elif vix <= 15: score += 10

        # 원달러 환율 (오르면 외국인 매도 압력)
        score += np.clip(-usd_chg * 3, -15, 15)

        # SOX 등락 + 갭
        score += np.clip(sox_chg * 2, -10, 10)
        score += np.clip(sox_gap * 2, -8,   8)      # SOX 갭 반영

        return float(np.clip(score, 0, 100))

    def apply_to_stocks(self, df: pd.DataFrame, us_data: dict) -> pd.DataFrame:
        """종목별 미국 증시 점수 적용 (섹터별 차등)"""
        df   = df.copy()
        base = self.calc_score(us_data)

        nasdaq_chg = us_data.get("나스닥",         {}).get("change", 0)
        nasdaq_gap = us_data.get("나스닥",         {}).get("gap",    0)
        sox_chg    = us_data.get("필라델피아반도체", {}).get("change", 0)
        sox_gap    = us_data.get("필라델피아반도체", {}).get("gap",    0)
        usd_chg    = us_data.get("원달러",          {}).get("change", 0)

        us_scores = []
        for _, row in df.iterrows():
            name  = str(row.get("name",""))
            score = base

            # 반도체: SOX 등락 + 갭 추가 반영
            if any(kw in name for kw in self.SEMI_KEYWORDS):
                score += np.clip(sox_chg * 3, -15, 15)
                score += np.clip(sox_gap * 3, -12, 12)   # SOX 오버나이트 갭

            # 수출 대형주: 환율 오르면 수혜
            if any(kw in name for kw in self.EXPORT_KEYWORDS):
                score += np.clip(usd_chg * 2, -10, 10)

            # IT/플랫폼: 나스닥 + 갭 강하게 반영
            if any(kw in name for kw in self.IT_KEYWORDS):
                score += np.clip(nasdaq_chg * 3, -15, 15)
                score += np.clip(nasdaq_gap * 2, -10, 10)

            us_scores.append(float(np.clip(score, 0, 100)))

        df["us_market_score"] = us_scores
        return df

    def get_gap_summary(self, us_data: dict) -> dict:
        """오버나이트 갭 요약 반환"""
        return {
            "나스닥 갭":   us_data.get("나스닥",         {}).get("gap", 0),
            "S&P500 갭":  us_data.get("S&P500",         {}).get("gap", 0),
            "SOX 갭":     us_data.get("필라델피아반도체", {}).get("gap", 0),
        }