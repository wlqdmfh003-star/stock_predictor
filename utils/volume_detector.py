import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class VolumeDetector:
    """
    거래량 이상 탐지 v7.0
    ★ 거래량 급증 감지 (평균 대비 2배 이상)
    ★ 거래량 + 가격 방향 일치 여부
    ★ OBV 다이버전스 감지
    ★ 거래량 점수 (0~100)
    """

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df      = df.copy()
        scores  = []
        anomaly = []

        for _, row in df.iterrows():
            ohlcv = row.get("ohlcv")
            if ohlcv is None or len(ohlcv) < 20:
                scores.append(50.0)
                anomaly.append(0)
                continue

            score, is_anomaly = self._analyze(ohlcv)
            scores.append(score)
            anomaly.append(is_anomaly)

        df["volume_score"]   = scores
        df["volume_anomaly"] = anomaly
        return df

    def _analyze(self, ohlcv: pd.DataFrame) -> tuple:
        try:
            close  = ohlcv["close"].astype(float)
            volume = ohlcv["volume"].astype(float)
            open_  = ohlcv["open"].astype(float) if "open" in ohlcv.columns else close

            score = 50.0
            is_anomaly = 0

            # 1. 거래량 급증 감지
            vol_avg20  = float(volume.tail(20).mean())
            vol_recent = float(volume.iloc[-1])
            vol_ratio  = vol_recent / (vol_avg20 + 1e-9)

            if vol_ratio >= 3.0:
                score += 25; is_anomaly = 1
            elif vol_ratio >= 2.0:
                score += 18; is_anomaly = 1
            elif vol_ratio >= 1.5:
                score += 10
            elif vol_ratio < 0.5:
                score -= 10  # 거래량 급감

            # 2. 거래량 + 가격 방향 일치
            price_up = float(close.iloc[-1]) > float(close.iloc[-2])
            if price_up and vol_ratio >= 1.5:
                score += 12  # 상승 + 거래량 증가 = 강한 신호
            elif not price_up and vol_ratio >= 1.5:
                score -= 8   # 하락 + 거래량 증가 = 약한 신호

            # 3. 5일 연속 거래량 추세
            vol_5 = volume.tail(5).values
            if len(vol_5) >= 3:
                vol_trend = float(np.polyfit(range(len(vol_5)), vol_5, 1)[0])
                if vol_trend > 0:
                    score += 5   # 거래량 증가 추세
                else:
                    score -= 3   # 거래량 감소 추세

            # 4. OBV 기울기
            direction = close.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            obv = (volume * direction).cumsum()
            if len(obv) >= 10:
                obv_slope = float(obv.iloc[-1] - obv.iloc[-10])
                if obv_slope > 0:
                    score += 8
                else:
                    score -= 5

            return float(np.clip(score, 0, 100)), is_anomaly

        except Exception:
            return 50.0, 0