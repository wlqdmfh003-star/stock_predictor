import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class HighLow52Week:
    """
    52주 신고가/신저가 분석 v2.0
    ★ 거래량 동반 신고가만 강한 신호 처리
    ★ 신고가 돌파 강도 측정
    ★ 연속 신고가 일수 (추세 지속성)
    ★ 거래량 없는 신고가는 약한 신호로 처리
    """

    VOLUME_SURGE = 1.5   # 거래량 평균 대비 1.5배 = 유효 신고가
    STRONG_BREAK = 2.0   # 2% 이상 돌파 = 강한 돌파

    def fetch_and_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        results = []

        ohlcv_list = df["ohlcv"].tolist() if "ohlcv" in df.columns else [None]*len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            ohlcv = ohlcv_list[i]
            results.append(self._analyze(ohlcv, row))

        df["high52_score"]       = [r["score"]             for r in results]
        df["pct_from_high"]      = [r["pct_from_high"]     for r in results]
        df["pct_from_low"]       = [r["pct_from_low"]      for r in results]
        df["is_new_high"]        = [r["is_new_high"]       for r in results]
        df["is_new_low"]         = [r["is_new_low"]        for r in results]
        df["breakout_strength"]  = [r["breakout_strength"] for r in results]
        df["volume_confirmed"]   = [r["volume_confirmed"]  for r in results]
        df["consec_highs"]       = [r["consec_highs"]      for r in results]
        return df

    def _analyze(self, ohlcv, row):
        default = {
            "score":50.0,"pct_from_high":0.0,"pct_from_low":0.0,
            "is_new_high":0,"is_new_low":0,"breakout_strength":0.0,
            "volume_confirmed":0,"consec_highs":0,
        }
        cur_price = float(row.get("current_price",0) or 0)
        if cur_price <= 0:
            return default

        try:
            if ohlcv is not None and isinstance(ohlcv, pd.DataFrame) and len(ohlcv) >= 20:
                close  = ohlcv["close"].astype(float)
                volume = ohlcv["volume"].astype(float)
                period = min(252, len(ohlcv))
                high52 = float(close.tail(period).max())
                low52  = float(close.tail(period).min())
            else:
                high52 = cur_price * 1.2
                low52  = cur_price * 0.7
                volume = None
                close  = None

            pct_from_high = round((cur_price/high52 - 1)*100, 2)
            pct_from_low  = round((cur_price/low52  - 1)*100, 2)
            is_new_high   = int(cur_price >= high52 * 0.995)
            is_new_low    = int(cur_price <= low52  * 1.005)

            score = 50.0
            if   pct_from_high >= -3:  score += 25
            elif pct_from_high >= -10: score += 15
            elif pct_from_high >= -20: score += 5
            elif pct_from_high <= -30: score -= 10

            breakout_strength = 0.0
            vol_confirmed     = 0
            consec            = 0

            if close is not None and volume is not None and len(close) >= 20:
                vol_avg    = float(volume.tail(20).mean())
                vol_recent = float(volume.iloc[-1])
                vol_ratio  = vol_recent / (vol_avg + 1e-9)

                if is_new_high:
                    vol_confirmed = int(vol_ratio >= self.VOLUME_SURGE)
                    prev_high     = float(close.tail(period).iloc[:-1].max()) \
                                    if len(close) > 1 else high52
                    breakout_strength = round((cur_price/prev_high - 1)*100, 2) \
                                        if prev_high > 0 else 0.0

                    if vol_confirmed:
                        score += 20 if breakout_strength >= self.STRONG_BREAK else 12
                    else:
                        score += 5  # 거래량 없는 신고가는 약한 신호

                # 연속 신고가
                period_max = float(close.tail(period).max())
                for j in range(len(close)-1, max(len(close)-20, 0)-1, -1):
                    if float(close.iloc[j]) >= period_max * 0.99:
                        consec += 1
                    else:
                        break

                if consec >= 3:   score += 10
                elif consec >= 2: score += 5

            if is_new_low:
                score -= 20

            return {
                "score":            float(np.clip(score, 0, 100)),
                "pct_from_high":    pct_from_high,
                "pct_from_low":     pct_from_low,
                "is_new_high":      is_new_high,
                "is_new_low":       is_new_low,
                "breakout_strength":breakout_strength,
                "volume_confirmed": vol_confirmed,
                "consec_highs":     consec,
            }
        except Exception:
            return default