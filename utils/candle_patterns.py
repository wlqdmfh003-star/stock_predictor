import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class CandlePatterns:
    """
    v5.1 캔들/차트 패턴 분석 - 19가지
    단일봉 6 + 2봉 4 + 3봉 4 + 중기패턴 5
    """

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        scores, patterns = [], []

        for _, row in df.iterrows():
            ohlcv = row.get("ohlcv")
            if ohlcv is None or len(ohlcv) < 5:
                scores.append(50.0)
                patterns.append("데이터부족")
                continue
            score, found = self._detect(ohlcv)
            scores.append(score)
            patterns.append(found if found else "패턴없음")

        df["candle_score"]   = scores
        df["candle_pattern"] = patterns
        return df

    def _detect(self, ohlcv: pd.DataFrame):
        o = ohlcv["open"].astype(float).values  if "open"   in ohlcv.columns else ohlcv["close"].astype(float).values
        h = ohlcv["high"].astype(float).values  if "high"   in ohlcv.columns else ohlcv["close"].astype(float).values
        l = ohlcv["low"].astype(float).values   if "low"    in ohlcv.columns else ohlcv["close"].astype(float).values
        c = ohlcv["close"].astype(float).values

        score, found = 50.0, []

        # ── 단일봉 (6) ────────────────────────────────────────────────────────
        if self._hammer(o,h,l,c):
            score+=20; found.append("망치형")
        if self._inv_hammer(o,h,l,c):
            score+=15; found.append("역망치형")
        if self._doji(o,h,l,c):
            score+=8;  found.append("도지")
        if self._marubozu_bull(o,h,l,c):
            score+=20; found.append("장대양봉")
        if self._marubozu_bear(o,h,l,c):
            score-=20; found.append("장대음봉")
        if self._spinning_top(o,h,l,c):
            found.append("팽이형")

        # ── 2봉 (4) ──────────────────────────────────────────────────────────
        if self._bull_engulfing(o,h,l,c):
            score+=25; found.append("상승장악형")
        if self._bear_engulfing(o,h,l,c):
            score-=25; found.append("하락장악형")
        if self._piercing(o,h,l,c):
            score+=18; found.append("관통형")
        if self._dark_cloud(o,h,l,c):
            score-=18; found.append("먹구름형")

        # ── 3봉 (4) ──────────────────────────────────────────────────────────
        if self._morning_star(o,h,l,c):
            score+=30; found.append("새벽별형")
        if self._evening_star(o,h,l,c):
            score-=30; found.append("저녁별형")
        if self._three_soldiers(o,h,l,c):
            score+=28; found.append("세병사")
        if self._three_crows(o,h,l,c):
            score-=28; found.append("세까마귀")

        # ── 중기 차트 패턴 (5) ───────────────────────────────────────────────
        if self._cup_handle(c):
            score+=22; found.append("컵핸들")
        if self._double_bottom(c, l):
            score+=28; found.append("이중바닥")
        if self._double_top(c, h):
            score-=28; found.append("이중천장")
        hs = self._head_shoulders(c, h, l)
        if hs == "역헤드앤숄더":
            score+=25; found.append("역헤드앤숄더")
        elif hs == "헤드앤숄더":
            score-=25; found.append("헤드앤숄더")
        if self._ascending_triangle(c, h, l):
            score+=18; found.append("상승삼각형")

        return float(np.clip(score, 0, 100)), ", ".join(found) if found else ""

    # ── 단일봉 ────────────────────────────────────────────────────────────────
    def _hammer(self, o,h,l,c):
        body  = abs(c[-1]-o[-1])
        low_w = min(o[-1],c[-1])-l[-1]
        hi_w  = h[-1]-max(o[-1],c[-1])
        return body>0 and low_w>=2*body and hi_w<=0.3*body and c[-1]>l[-1]

    def _inv_hammer(self, o,h,l,c):
        body  = abs(c[-1]-o[-1])
        hi_w  = h[-1]-max(o[-1],c[-1])
        low_w = min(o[-1],c[-1])-l[-1]
        return body>0 and hi_w>=2*body and low_w<=0.3*body

    def _doji(self, o,h,l,c):
        body  = abs(c[-1]-o[-1])
        range_= h[-1]-l[-1]
        return range_>0 and body/range_<=0.1

    def _marubozu_bull(self, o,h,l,c):
        body  = c[-1]-o[-1]
        range_= h[-1]-l[-1]
        return body>0 and range_>0 and body/range_>=0.9

    def _marubozu_bear(self, o,h,l,c):
        body  = o[-1]-c[-1]
        range_= h[-1]-l[-1]
        return body>0 and range_>0 and body/range_>=0.9

    def _spinning_top(self, o,h,l,c):
        body  = abs(c[-1]-o[-1])
        range_= h[-1]-l[-1]
        return range_>0 and 0.1<body/range_<0.3

    # ── 2봉 ──────────────────────────────────────────────────────────────────
    def _bull_engulfing(self, o,h,l,c):
        if len(c)<2: return False
        return c[-2]<o[-2] and c[-1]>o[-1] and o[-1]<=c[-2] and c[-1]>=o[-2]

    def _bear_engulfing(self, o,h,l,c):
        if len(c)<2: return False
        return c[-2]>o[-2] and c[-1]<o[-1] and o[-1]>=c[-2] and c[-1]<=o[-2]

    def _piercing(self, o,h,l,c):
        if len(c)<2: return False
        mid = (o[-2]+c[-2])/2
        return c[-2]<o[-2] and c[-1]>o[-1] and o[-1]<c[-2] and c[-1]>mid

    def _dark_cloud(self, o,h,l,c):
        if len(c)<2: return False
        mid = (o[-2]+c[-2])/2
        return c[-2]>o[-2] and c[-1]<o[-1] and o[-1]>c[-2] and c[-1]<mid

    # ── 3봉 ──────────────────────────────────────────────────────────────────
    def _morning_star(self, o,h,l,c):
        if len(c)<3: return False
        big_bear = (o[-3]-c[-3]) > abs(o[-3]-c[-3])*0.6
        small    = abs(c[-2]-o[-2]) < abs(o[-3]-c[-3])*0.3
        big_bull = (c[-1]-o[-1]) > abs(c[-1]-o[-1])*0.6
        return big_bear and small and big_bull and max(o[-2],c[-2])<c[-3]

    def _evening_star(self, o,h,l,c):
        if len(c)<3: return False
        big_bull = (c[-3]-o[-3]) > abs(c[-3]-o[-3])*0.6
        small    = abs(c[-2]-o[-2]) < abs(c[-3]-o[-3])*0.3
        big_bear = (o[-1]-c[-1]) > abs(o[-1]-c[-1])*0.6
        return big_bull and small and big_bear and min(o[-2],c[-2])>c[-3]

    def _three_soldiers(self, o,h,l,c):
        if len(c)<4: return False
        return all(c[-3+i]>o[-3+i] and c[-3+i]>c[-4+i] for i in range(3))

    def _three_crows(self, o,h,l,c):
        if len(c)<4: return False
        return all(c[-3+i]<o[-3+i] and c[-3+i]<c[-4+i] for i in range(3))

    # ── 중기 차트 패턴 ────────────────────────────────────────────────────────
    def _cup_handle(self, c):
        """컵핸들 - 최소 30일"""
        if len(c)<30: return False
        w = c[-30:]
        peak1  = w[:5].max()
        trough = w[5:20].min()
        peak2  = w[20:].max()
        if peak1<=0: return False
        depth = (peak1-trough)/peak1
        return 0.10<=depth<=0.35 and peak2>=peak1*0.95

    def _double_bottom(self, c, l):
        """
        이중바닥 (W형) - 상승 반전 신호
        조건: 두 저점이 비슷한 수준, 중간에 반등, 최근 고점 돌파
        """
        if len(c)<20: return False
        window = 20
        lows   = l[-window:] if len(l)>=window else l
        closes = c[-window:] if len(c)>=window else c

        # 저점 2개 찾기
        low_indices = []
        for i in range(2, len(lows)-2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and \
               lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                low_indices.append(i)

        if len(low_indices) < 2:
            return False

        b1_idx = low_indices[-2]
        b2_idx = low_indices[-1]

        # 두 저점 간격 최소 5일
        if b2_idx - b1_idx < 5:
            return False

        b1 = lows[b1_idx]
        b2 = lows[b2_idx]

        # 두 저점이 5% 이내로 비슷
        if b1<=0 or abs(b1-b2)/b1 > 0.05:
            return False

        # 중간 고점 (넥라인)
        neck = closes[b1_idx:b2_idx].max() if b2_idx>b1_idx else closes[-1]

        # 최근 가격이 넥라인 돌파
        return closes[-1] >= neck * 0.98

    def _double_top(self, c, h):
        """
        이중천장 (M형) - 하락 반전 신호
        """
        if len(c)<20: return False
        window = 20
        highs  = h[-window:] if len(h)>=window else h
        closes = c[-window:] if len(c)>=window else c

        # 고점 2개 찾기
        hi_indices = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and \
               highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                hi_indices.append(i)

        if len(hi_indices) < 2:
            return False

        t1_idx = hi_indices[-2]
        t2_idx = hi_indices[-1]

        if t2_idx - t1_idx < 5:
            return False

        t1 = highs[t1_idx]
        t2 = highs[t2_idx]

        if t1<=0 or abs(t1-t2)/t1 > 0.05:
            return False

        # 넥라인 하향 돌파
        neck = closes[t1_idx:t2_idx].min() if t2_idx>t1_idx else closes[-1]
        return closes[-1] <= neck * 1.02

    def _head_shoulders(self, c, h, l):
        """
        헤드앤숄더 / 역헤드앤숄더
        최소 30일, 5개 주요 포인트 탐지
        """
        if len(c)<30: return None
        window = 30
        hi = h[-window:] if len(h)>=window else h
        lo = l[-window:] if len(l)>=window else l
        cl = c[-window:] if len(c)>=window else c

        # ── 역헤드앤숄더 (상승 신호) ─────────────────────────────────
        # 왼쪽어깨(저점) - 머리(최저점) - 오른쪽어깨(저점) - 넥라인 돌파
        lo_idx = []
        for i in range(2, len(lo)-2):
            if lo[i]<lo[i-1] and lo[i]<lo[i+1] and lo[i]<lo[i-2] and lo[i]<lo[i+2]:
                lo_idx.append(i)

        if len(lo_idx) >= 3:
            ls = lo_idx[-3]  # 왼쪽어깨
            hd = lo_idx[-2]  # 머리 (가장 낮아야)
            rs = lo_idx[-1]  # 오른쪽어깨

            if lo[hd] < lo[ls] and lo[hd] < lo[rs]:
                # 두 어깨 높이 비슷 (10% 이내)
                if lo[ls]>0 and abs(lo[ls]-lo[rs])/lo[ls] <= 0.10:
                    # 넥라인: 두 어깨 사이 고점들의 평균
                    neck = hi[ls:rs].max() if rs>ls else cl[-1]
                    # 최근 가격이 넥라인 돌파
                    if cl[-1] >= neck * 0.97:
                        return "역헤드앤숄더"

        # ── 헤드앤숄더 (하락 신호) ───────────────────────────────────
        hi_idx = []
        for i in range(2, len(hi)-2):
            if hi[i]>hi[i-1] and hi[i]>hi[i+1] and hi[i]>hi[i-2] and hi[i]>hi[i+2]:
                hi_idx.append(i)

        if len(hi_idx) >= 3:
            ls = hi_idx[-3]
            hd = hi_idx[-2]
            rs = hi_idx[-1]

            if hi[hd] > hi[ls] and hi[hd] > hi[rs]:
                if hi[ls]>0 and abs(hi[ls]-hi[rs])/hi[ls] <= 0.10:
                    neck = lo[ls:rs].min() if rs>ls else cl[-1]
                    if cl[-1] <= neck * 1.03:
                        return "헤드앤숄더"

        return None

    def _ascending_triangle(self, c, h, l):
        """
        상승삼각형 - 저점은 올라가고 고점은 수평 → 상방 돌파 기대
        최소 15일
        """
        if len(c)<15: return False
        window = 15
        hi = h[-window:] if len(h)>=window else h
        lo = l[-window:] if len(l)>=window else l
        cl = c[-window:] if len(c)>=window else c

        # 고점이 수평 (저항선): 최근 5일 고점들의 표준편차가 작음
        hi_std = hi[-5:].std() / (hi[-5:].mean()+1e-9)
        if hi_std > 0.02:
            return False

        # 저점이 상승 추세
        lo_first = lo[:5].mean()
        lo_last  = lo[-5:].mean()
        if lo_first<=0 or (lo_last-lo_first)/lo_first < 0.02:
            return False

        # 현재가가 저항선 근처
        resistance = hi[-5:].mean()
        return cl[-1] >= resistance * 0.97