import pandas as pd
import numpy as np
import requests
import time
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta


def _fetch_short_naver(code: str) -> dict:
    """네이버 금융 공매도 비율 크롤링"""
    try:
        url     = f"https://finance.naver.com/item/frgn.naver?code={code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp    = requests.get(url, headers=headers, timeout=8)

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        # 공매도 비율은 종목 상세 - 공매도 탭에서
        # 대안: KRX 공매도 CSV API
        return _fetch_short_krx(code)
    except Exception:
        return {"ratio": 0.0, "trend": 0.0}


def _fetch_short_krx(code: str) -> dict:
    """KRX 공매도 잔고 조회 (무료 API)"""
    try:
        today    = datetime.now().strftime("%Y%m%d")
        start_20 = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

        url     = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        payload = {
            "bld":         "dbms/MDC/STAT/standard/MDCSTAT10001",
            "trdDd":       today,
            "isuCd":       code,
            "strtDd":      start_20,
            "endDd":       today,
            "csvxls_isNo": "false",
        }
        headers = {
            "User-Agent":   "Mozilla/5.0",
            "Referer":      "http://data.krx.co.kr/",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = requests.post(url, data=payload, headers=headers, timeout=10)
        data = resp.json()
        rows = data.get("block1", [])

        if not rows:
            return {"ratio": 0.0, "trend": 0.0}

        ratios = []
        for row in rows:
            ratio_str = str(row.get("SHRT_BLNC_RT","0")).replace(",","")
            try:
                ratios.append(float(ratio_str))
            except Exception:
                pass

        if not ratios:
            return {"ratio": 0.0, "trend": 0.0}

        latest  = ratios[-1]
        avg_all = float(np.mean(ratios))
        avg_5   = float(np.mean(ratios[-5:])) if len(ratios) >= 5 else latest
        trend   = avg_5 - avg_all  # 음수면 최근 공매도 감소 (좋은 신호)

        return {"ratio": latest, "trend": trend}

    except Exception:
        return {"ratio": 0.0, "trend": 0.0}


class ShortSelling:
    """
    공매도 분석 v2.0
    ★ pykrx 완전 제거
    ★ KRX 공매도 잔고 API (무료)
    ★ 공매도 비율 + 트렌드(감소/증가) 분석
    """

    def fetch_and_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df    = df.copy()
        codes = df["code"].tolist()

        print(f"📉 공매도 데이터 수집 중... ({len(codes)}개)")
        short_data = {}

        def fetch_one(code):
            data = _fetch_short_krx(code)
            time.sleep(0.05)
            return code, data

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_one, c): c for c in codes}
            done = 0
            for future in as_completed(futures):
                code, data = future.result()
                short_data[code] = data
                done += 1
                if done % 20 == 0:
                    print(f"   공매도 진행: {done}/{len(codes)}")

        scores, ratios = [], []
        for _, row in df.iterrows():
            code  = str(row.get("code",""))
            data  = short_data.get(code, {"ratio":0.0,"trend":0.0})
            ratio = float(data.get("ratio", 0) or 0)
            trend = float(data.get("trend", 0) or 0)
            score = self._calc_score(ratio, trend)
            scores.append(score)
            ratios.append(ratio)

        df["short_score"] = scores
        df["short_ratio"] = ratios
        return df

    def _calc_score(self, ratio: float, trend: float) -> float:
        """공매도 비율 + 트렌드 → 점수"""
        score = self._ratio_to_score(ratio)

        # 트렌드 반영 (공매도 감소 = 호재)
        if trend < -0.5:   score += 15   # 공매도 크게 감소
        elif trend < -0.2: score += 8
        elif trend > 0.5:  score -= 15   # 공매도 크게 증가
        elif trend > 0.2:  score -= 8

        return float(np.clip(score, 0, 100))

    def _ratio_to_score(self, ratio: float) -> float:
        if ratio <= 0:    return 55.0
        elif ratio < 0.5: return 70.0
        elif ratio < 1.0: return 62.0
        elif ratio < 2.0: return 50.0
        elif ratio < 5.0: return 38.0
        else:             return 25.0