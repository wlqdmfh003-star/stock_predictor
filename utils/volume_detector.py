import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class VolumeDetector:

    def detect(self, df):
        df        = df.copy()
        scores    = []
        anomalies = []

        for _, row in df.iterrows():
            ohlcv      = row.get("ohlcv")
            score, is_anomaly = self._calc(ohlcv, row.get("volume", 0))
            scores.append(score)
            anomalies.append(is_anomaly)

        df["volume_score"]   = scores
        df["volume_anomaly"] = anomalies
        return df

    def _calc(self, ohlcv, cur_volume):
        try:
            if ohlcv is None or len(ohlcv) < 20:
                return 50.0, 0

            vol     = ohlcv["volume"].astype(float)
            mean_20 = vol.rolling(20).mean().iloc[-1]
            std_20  = vol.rolling(20).std().iloc[-1]

            if std_20 == 0 or np.isnan(std_20):
                return 50.0, 0

            z_score    = (float(cur_volume) - mean_20) / std_20
            score      = float(np.clip(50 + z_score * 15, 0, 100))
            is_anomaly = int(z_score > 2.0)

            recent_avg = vol.tail(5).mean()
            ratio      = recent_avg / (mean_20 + 1e-9)
            score      = float(np.clip(score * ratio * 0.5 + score * 0.5, 0, 100))

            return score, is_anomaly

        except Exception:
            return 50.0, 0