import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SectorAnalysis:
    """
    v7.0 섹터/업종 분석
    - pykrx 완전 제거 (자체 섹터 매핑)
    - 종목 수 3배 확장
    - 방산/조선 섹터 추가
    - 5일 vs 20일 모멘텀 비교로 섹터 로테이션 감지
    ★ 신규: 섹터 로테이션 전략 (강한 섹터 → 가중치 자동 UP)
    ★ 신규: 섹터 강도 순위 (TOP3 섹터 집중 추천)
    ★ 신규: 섹터 모멘텀 기반 종목 점수 가중치 자동 조정
    """

    SECTORS = {
        "반도체": [
            "삼성전자","SK하이닉스","DB하이텍","하나마이크론","리노공업",
            "원익IPS","피에스케이","HPSP","에스앤에스텍","ISC",
            "한미반도체","티씨케이","SFA반도체","코미코","에프에스티",
            "심텍","대덕전자","인텍플러스","오로스테크놀로지","네패스",
            "솔브레인","후성","SK실트론","웨이퍼","디아이",
        ],
        "2차전지": [
            "LG에너지솔루션","삼성SDI","SK이노베이션","에코프로","에코프로비엠",
            "포스코퓨처엠","엘앤에프","천보","나노신소재","일진머티리얼즈",
            "엔켐","솔루스첨단소재","코스모신소재","동화기업","피엔티",
            "씨아이에스","에스엠랩","원통형배터리","비에이치","명성티엔에스",
            "에코앤드림","DI동일","세방전지","파워로직스","리튬","이브이첨단소재",
        ],
        "바이오/제약": [
            "삼성바이오로직스","셀트리온","유한양행","한미약품","종근당",
            "대웅제약","HLB","알테오젠","에스티팜","보령","동국제약",
            "일동제약","광동제약","JW중외제약","녹십자","한독","동아에스티",
            "제넥신","메디톡스","코오롱티슈진","테고사이언스","비씨월드제약",
            "파멥신","압타머사이언스","오스코텍","지씨셀","ABL바이오",
        ],
        "자동차/부품": [
            "현대차","기아","현대모비스","한온시스템","만도",
            "현대위아","HL만도","성우하이텍","세종공업","평화정공",
            "모베이스","서연이화","화신","인지컨트롤스","영화테크",
            "S&T모티브","에스엘","삼보모터스","동원금속","코다코",
        ],
        "IT/플랫폼": [
            "카카오","NAVER","네이버","크래프톤","넥슨",
            "엔씨소프트","카카오게임즈","펄어비스","컴투스","위메이드",
            "카카오뱅크","카카오페이","KG모빌리언스","한국전자금융","다날",
            "NHN","NHN한국사이버결제","메가스터디교육","이크레더블",
        ],
        "방산": [
            "한화에어로스페이스","LIG넥스원","현대로템","한국항공우주","빅텍",
            "퍼스텍","스페코","오르비텍","휴니드","한화시스템",
            "풍산","기아","현대차","삼성테크윈","한화","SNT모티브",
            "한화오션","HGH로보틱스","한화비전","이엠코리아",
        ],
        "조선": [
            "한화오션","HD현대중공업","삼성중공업","HD현대미포조선","대우조선해양",
            "HD현대","HD현대인프라코어","케이조선","HJ중공업","세진중공업",
            "성동조선","STX조선해양","현대삼호중공업","한진중공업","조광ILI",
        ],
        "금융": [
            "KB금융","신한지주","하나금융지주","우리금융지주","기업은행",
            "삼성생명","삼성화재","미래에셋","키움증권","NH투자증권",
            "한국금융지주","DB손해보험","현대해상","메리츠금융지주","교보생명",
            "BNK금융지주","DGB금융지주","JB금융지주","카카오뱅크","케이뱅크",
        ],
        "화학": [
            "LG화학","롯데케미칼","금호석유","한화솔루션","OCI",
            "효성화학","SKC","대한유화","한화임팩트","KG케미칼",
            "금양","이수화학","태광산업","삼성정밀화학","동성화인텍",
        ],
        "철강/소재": [
            "POSCO홀딩스","현대제철","고려아연","풍산","세아베스틸",
            "동국제강","KG스틸","포스코스틸리온","세아특수강","성일하이텍",
            "영풍","고려제강","동원시스템즈","화성밸브","태웅",
        ],
        "건설/부동산": [
            "삼성물산","현대건설","GS건설","DL이앤씨","HDC현대산업개발",
            "대우건설","태영건설","호반건설","신세계건설","효성중공업",
            "두산건설","코오롱글로벌","한신공영","계룡건설","동부건설",
        ],
        "에너지": [
            "SK이노베이션","S-Oil","GS","한국전력","한국가스공사",
            "SK가스","E1","한국지역난방공사","STX에너지솔루션","에스에너지",
            "한화솔루션","OCI","신성이엔지","에스와이","SK오케이",
        ],
        "유통/소비": [
            "롯데쇼핑","신세계","이마트","현대백화점","GS리테일",
            "BGF리테일","CJ제일제당","오리온","농심","롯데칠성",
            "하이트진로","무학","신라면세점","호텔신라","파라다이스",
        ],
    }

    def __init__(self):
        self.today = datetime.now().strftime("%Y%m%d")

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sector"] = df["name"].apply(self._get_sector)

        # 섹터별 5일/20일 모멘텀 계산 (로테이션 감지)
        sector_stats = self._calc_sector_stats(df)

        df["sector_score"] = df.apply(
            lambda row: self._calc_score(row, sector_stats), axis=1
        )
        df["sector_rotation"] = df["sector"].apply(
            lambda s: sector_stats.get(s, {}).get("rotation_signal", "중립")
        )
        return df

    def _get_sector(self, name: str) -> str:
        for sector, stocks in self.SECTORS.items():
            if any(stock in name for stock in stocks):
                return sector
        return "기타"

    def _calc_sector_stats(self, df: pd.DataFrame) -> dict:
        """섹터별 5일/20일 모멘텀 비교 → 로테이션 신호 생성"""
        stats = {}
        for sector in list(self.SECTORS.keys()) + ["기타"]:
            sdf = df[df["sector"] == sector]
            if len(sdf) == 0:
                stats[sector] = {
                    "avg_change": 0.0, "avg_momentum": 50.0,
                    "stock_count": 0, "rotation_signal": "중립",
                    "momentum_5d": 0.0, "momentum_20d": 0.0,
                }
                continue

            avg_change   = float(sdf["change_pct"].mean())       if "change_pct"    in sdf.columns else 0.0
            avg_momentum = float(sdf["momentum_score"].mean())   if "momentum_score" in sdf.columns else 50.0

            # 5일 vs 20일 모멘텀 (ohlcv 데이터 있을 때)
            m5, m20 = self._extract_momentum(sdf)

            # 로테이션 신호: 5일이 20일보다 강하면 "가속", 약하면 "둔화"
            if   m5 > m20 + 1.0:  rotation = "🚀 가속"
            elif m5 < m20 - 1.0:  rotation = "🔻 둔화"
            else:                  rotation = "➡️ 중립"

            stats[sector] = {
                "avg_change":      avg_change,
                "avg_momentum":    avg_momentum,
                "stock_count":     len(sdf),
                "rotation_signal": rotation,
                "momentum_5d":     m5,
                "momentum_20d":    m20,
            }
        return stats

    def _extract_momentum(self, sdf: pd.DataFrame):
        """ohlcv에서 5일/20일 수익률 계산"""
        m5_list, m20_list = [], []
        for _, row in sdf.iterrows():
            ohlcv = row.get("ohlcv")
            if ohlcv is None or len(ohlcv) < 21:
                continue
            close = ohlcv["close"].astype(float).values
            if len(close) >= 6:
                m5_list.append((close[-1] / close[-6] - 1) * 100)
            if len(close) >= 21:
                m20_list.append((close[-1] / close[-21] - 1) * 100)

        m5  = float(np.mean(m5_list))  if m5_list  else 0.0
        m20 = float(np.mean(m20_list)) if m20_list else 0.0
        return m5, m20

    def _calc_score(self, row, sector_stats: dict) -> float:
        sector = row.get("sector", "기타")
        data   = sector_stats.get(sector, {})
        score  = 50.0

        avg_mom = data.get("avg_momentum", 50.0)
        avg_chg = data.get("avg_change",   0.0)
        m5      = data.get("momentum_5d",  0.0)
        m20     = data.get("momentum_20d", 0.0)
        count   = data.get("stock_count",  0)

        # 섹터 평균 모멘텀
        score += (avg_mom - 50) * 0.3
        # 섹터 평균 등락률
        score += np.clip(avg_chg * 3, -15, 15)
        # 5일 > 20일 이면 로테이션 가속 보너스
        score += np.clip((m5 - m20) * 1.5, -10, 10)
        # 섹터 내 종목 수 신뢰도
        if count >= 3:
            score += 5
        # 개별 종목이 섹터 평균보다 강하면 보너스
        own_chg = float(row.get("change_pct", 0))
        if own_chg > avg_chg:
            score += 5

        return float(np.clip(score, 0, 100))

    def get_sector_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sector" not in df.columns:
            return pd.DataFrame()
        rows = []
        for sector in list(self.SECTORS.keys()) + ["기타"]:
            sdf = df[df["sector"] == sector]
            if len(sdf) == 0:
                continue
            rows.append({
                "섹터":        sector,
                "종목수":       len(sdf),
                "평균상승확률": round(sdf["rise_prob"].mean(), 1)    if "rise_prob"    in sdf.columns else 0,
                "평균등락률":   round(sdf["change_pct"].mean(), 2)   if "change_pct"   in sdf.columns else 0,
                "5일모멘텀":    round(sdf["momentum_score"].mean(),1) if "momentum_score" in sdf.columns else 0,
                "로테이션":     sdf["sector_rotation"].iloc[0]        if "sector_rotation" in sdf.columns else "-",
                "TOP종목":      str(sdf.sort_values("rise_prob", ascending=False).iloc[0]["name"])
                                if "rise_prob" in sdf.columns and len(sdf) > 0 else "-",
            })
        return pd.DataFrame(rows).sort_values("평균상승확률", ascending=False)

    # ══════════════════════════════════════════════════════════════
    # ★ 섹터 로테이션 전략 v7.0
    # ══════════════════════════════════════════════════════════════
    def get_rotation_strategy(self, df: pd.DataFrame) -> dict:
        """
        섹터 로테이션 전략
        - TOP3 섹터 자동 감지
        - 강한 섹터 종목 가중치 UP
        - 약한 섹터 종목 가중치 DOWN
        반환: {종목코드: 가중치보정값}
        """
        if "sector" not in df.columns:
            df = df.copy()
            df["sector"] = df["name"].apply(self._get_sector)

        sector_stats = self._calc_sector_stats(df)

        # 섹터 강도 점수 계산
        sector_strength = {}
        for sector, stats in sector_stats.items():
            m5  = stats.get("momentum_5d",  0)
            m20 = stats.get("momentum_20d", 0)
            cnt = stats.get("stock_count",  0)
            avg = stats.get("avg_momentum", 50)
            if cnt == 0:
                continue
            # 강도 = 5일모멘텀(40%) + 5vs20차이(30%) + 평균모멘텀(30%)
            strength = m5*0.4 + (m5-m20)*0.3 + (avg-50)*0.3
            sector_strength[sector] = round(float(strength), 2)

        if not sector_strength:
            return {"top3_sectors": [], "weight_adj": {}, "summary": "데이터 없음"}

        # TOP3 / BOTTOM3 섹터 선정
        sorted_sectors = sorted(sector_strength.items(), key=lambda x: x[1], reverse=True)
        top3    = [s[0] for s in sorted_sectors[:3]]
        bottom3 = [s[0] for s in sorted_sectors[-3:]]

        # 종목별 가중치 보정값 계산
        weight_adj = {}
        for _, row in df.iterrows():
            code   = str(row.get("code", ""))
            sector = row.get("sector", "기타")
            if   sector in top3:    weight_adj[code] = +15.0  # TOP 섹터 +15점
            elif sector in bottom3: weight_adj[code] = -10.0  # 하위 섹터 -10점
            else:                   weight_adj[code] =   0.0

        return {
            "top3_sectors":    top3,
            "bottom3_sectors": bottom3,
            "sector_strength": sector_strength,
            "weight_adj":      weight_adj,
            "summary": f"강세섹터: {' > '.join(top3[:3])} | "
                       f"약세섹터: {' > '.join(bottom3[:3])}",
        }

    def apply_rotation_weight(self, df: pd.DataFrame,
                              rotation: dict) -> pd.DataFrame:
        """
        섹터 로테이션 가중치를 종목 점수에 반영
        total_score가 있을 때 호출
        """
        df     = df.copy()
        adj    = rotation.get("weight_adj", {})
        if not adj or "total_score" not in df.columns:
            return df

        new_scores = []
        for _, row in df.iterrows():
            code  = str(row.get("code", ""))
            score = float(row.get("total_score", 50))
            bonus = adj.get(code, 0.0)
            new_scores.append(float(np.clip(score + bonus, 0, 100)))

        df["total_score"] = new_scores
        return df