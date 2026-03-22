import os
import re
import pandas as pd
import numpy as np
import requests
import time
import warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('TRANSFORMERS_VERBOSITY',   'error')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')


# ── 테마주 키워드 사전 ────────────────────────────────────────────────────────
THEME_KEYWORDS = {
    "AI반도체":  ["HBM","AI반도체","엔비디아","GPU","NPU","온디바이스AI","AI칩",
                 "CoWoS","PIM","GDDR","AI가속기","LLM칩","뉴럴프로세서"],
    "2차전지":   ["배터리","전기차","리튬","양극재","음극재","전해질","ESS","EV",
                 "전고체","리튬인산철","LFP","NCM","배터리팩","충전인프라"],
    "바이오":    ["임상","FDA","바이오","신약","항체","mRNA","세포치료","유전자",
                 "ADC","GLP-1","비만치료","알츠하이머","CAR-T","줄기세포"],
    "방산":      ["방산","무기","K방산","미사일","드론","전차","함정","방위산업",
                 "레이더","유도탄","군함","전투기","방공","수출계약"],
    "로봇":      ["로봇","휴머노이드","자동화","협동로봇","산업용로봇",
                 "AMR","물류로봇","서비스로봇","AI로봇","보행로봇"],
    "우주항공":  ["우주","위성","발사체","항공","우주항공","누리호",
                 "소형위성","군집위성","우주인터넷","달탐사","우주여행"],
    "친환경":    ["수소","탄소중립","친환경","태양광","풍력","재생에너지",
                 "탄소저감","그린수소","연료전지","전기선박","ESG"],
    "반도체":    ["반도체","웨이퍼","DRAM","낸드","파운드리","패키징",
                 "칩렛","TSMC","삼성파운드리","반도체장비","EUV"],
    "자율주행":  ["자율주행","ADAS","라이다","V2X","자율차",
                 "자율운항","자율비행","로보택시","완전자율","레벨4"],
    "엔터/IP":   ["아이돌","K팝","웹툰","웹소설","IP","콘텐츠","OTT",
                 "한류","드라마","영화","게임IP","캐릭터"],
    "원전/에너지":["원전","SMR","핵융합","원자력","에너지안보","LNG",
                  "전력망","에너지전환","탈탄소","CCS"],
    "의료기기":  ["의료기기","임플란트","진단기기","내시경","수술로봇",
                 "체외진단","의료영상","디지털헬스","원격의료"],
}

# 섹터 → 테마 자동 매핑 (섹터에 속하면 테마 자동 부여)
# ★ 조선→방산 오류 수정 / IT→AI반도체 오류 수정
SECTOR_THEME_MAP = {
    "반도체":     "반도체",
    "2차전지":    "2차전지",
    "바이오/제약": "바이오",
    "방산":       "방산",
    "로봇/자동화": "로봇",
    "엔터/미디어": "엔터/IP",
    "에너지":     "친환경",
    "자동차/부품": "자율주행",
    "화학":       "친환경",
    "조선":       "조선",
    "금융":       "금융",
    "철강/소재":  "소재",
    "건설/부동산": "건설",
    "유통/소비":  "소비/뷰티",
    "IT/플랫폼":  "플랫폼",
    "운송/물류":  "물류",
    # ETF는 테마 없음
}

# ★ 종목 코드 기반 테마 사전 (가장 정확 - 코드는 절대 안 바뀜)
CODE_THEME_MAP = {
    # AI반도체
    "005930": "AI반도체,반도체",   # 삼성전자
    "000660": "AI반도체,반도체",   # SK하이닉스
    "042700": "AI반도체,반도체",   # 한미반도체
    "278990": "AI반도체,반도체",   # 이수페타시스
    "403870": "AI반도체,반도체",   # HPSP
    "058470": "AI반도체,반도체",   # 리노공업
    "204270": "AI반도체,반도체",   # 에스앤에스텍
    "033640": "AI반도체,반도체",   # 네패스
    "140860": "AI반도체,반도체",   # 파크시스템스
    "357780": "AI반도체,반도체",   # 솔브레인
    "096770": "반도체",            # SK이노베이션(반도체소재)
    "045390": "AI반도체,반도체",   # 대아티아이
    "036830": "반도체",            # 솔브레인홀딩스
    "098460": "반도체",            # 코오롱인더
    # 2차전지
    "086520": "2차전지",           # 에코프로
    "247540": "2차전지",           # 에코프로비엠
    "373220": "2차전지",           # LG에너지솔루션
    "003670": "2차전지",           # 포스코퓨처엠
    "006400": "2차전지",           # 삼성SDI
    "066970": "2차전지",           # 엘앤에프
    "278280": "2차전지",           # 천보
    "051600": "2차전지",           # 한전KPS(ESS)
    "207940": "2차전지",           # 삼성바이오로직스 → 바이오로 변경
    "096530": "2차전지",           # 씨젠
    "272450": "2차전지",           # 솔루스첨단소재
    "009830": "2차전지",           # 한화솔루션
    "006260": "2차전지",           # LS
    "012450": "2차전지",           # 한화에어로스페이스 → 방산
    "004000": "2차전지",           # 롯데정밀화학
    # 방산
    "012450": "방산,우주항공",     # 한화에어로스페이스
    "047810": "방산,우주항공",     # 한국항공우주
    "035610": "방산",              # LIG넥스원
    "064350": "방산",              # 현대로템
    "071050": "방산",              # 한화시스템
    "082270": "방산",              # 제이씨현시스템
    "024900": "방산",              # 덕양산업
    "035890": "방산",              # 서울옥션
    "014620": "방산",              # 성우하이텍
    "079550": "방산",              # LIG넥스원
    # 로봇
    "277810": "로봇",              # 레인보우로보틱스
    "454910": "로봇",              # 두산로보틱스
    "215100": "로봇",              # GS홈쇼핑
    "090460": "로봇",              # 비에이치
    "108490": "로봇",              # 로보티즈
    "319400": "로봇",              # 현대로보틱스
    # 엔터/IP
    "352820": "엔터/IP",           # 하이브
    "041510": "엔터/IP",           # SM엔터테인먼트
    "035900": "엔터/IP",           # JYP Ent.
    "122870": "엔터/IP",           # 와이지엔터테인먼트
    "035760": "엔터/IP",           # CJ ENM
    "425490": "엔터/IP",           # 케이팝 관련
    "032640": "엔터/IP",           # LG유플러스
    # 원전/에너지
    "034020": "원전/에너지",       # 두산에너빌리티
    "051600": "원전/에너지",       # 한전KPS
    "015760": "원전/에너지",       # 한국전력
    "336260": "원전/에너지,친환경", # 두산퓨얼셀
    "298040": "원전/에너지",       # 효성중공업
    "042670": "원전/에너지",       # HD현대인프라코어
    # 바이오
    "207940": "바이오",            # 삼성바이오로직스
    "068270": "바이오",            # 셀트리온
    "000100": "바이오",            # 유한양행
    "128940": "바이오",            # 한미약품
    "185750": "바이오",            # 종근당
    "069620": "바이오",            # 대웅제약
    "028300": "바이오",            # HLB
    "196170": "바이오",            # 알테오젠
    "009420": "바이오",            # 한올바이오파마
    "302440": "바이오",            # SK바이오사이언스
    "145020": "바이오",            # 휴젤
    # 의료기기
    "214150": "의료기기",          # 클래시스
    "048260": "의료기기",          # 오스템임플란트
    "065450": "의료기기",          # 인바디
    "100120": "의료기기",          # 뷰웍스
    "007390": "의료기기",          # 네이처셀
    "307280": "의료기기",          # 원텍
    # 자율주행/전기차
    "005380": "자율주행,전기차",   # 현대차
    "000270": "자율주행,전기차",   # 기아
    "012330": "자율주행,전기차",   # 현대모비스
    "204320": "자율주행",          # HL만도
    "060980": "자율주행",          # 에스에프에이
    "214430": "자율주행",          # 아모텍
    # 친환경/수소
    "009830": "친환경",            # 한화솔루션
    "010060": "친환경",            # OCI
    "322000": "친환경",            # 신성이엔지
    "014820": "친환경",            # 동원시스템즈
    "268280": "친환경",            # 에코앤드림
    "950130": "친환경",            # 엑세스바이오
    # 우주항공
    "099320": "우주항공",          # 쎄트렉아이
    "099220": "우주항공",          # 켄코아에어로스페이스
    "099340": "우주항공",          # 컨텍
    # 플랫폼/게임
    "035420": "플랫폼",            # NAVER
    "035720": "플랫폼",            # 카카오
    "259960": "게임",              # 크래프톤
    "036570": "게임",              # 엔씨소프트
    "251270": "게임",              # 넷마블
    "293490": "게임",              # 카카오게임즈
    "263750": "게임",              # 펄어비스
}

# 종목명 기반 테마 매핑 (코드 없을 때 보완용)
STOCK_THEME_MAP = {
    "삼성전자": "AI반도체,반도체",
    "SK하이닉스": "AI반도체,반도체",
    "한미반도체": "AI반도체,반도체",
    "이수페타시스": "AI반도체,반도체",
    "HPSP": "AI반도체,반도체",
    "리노공업": "반도체",
    "에코프로": "2차전지",
    "에코프로비엠": "2차전지",
    "LG에너지솔루션": "2차전지",
    "포스코퓨처엠": "2차전지",
    "삼성SDI": "2차전지",
    "엘앤에프": "2차전지",
    "천보": "2차전지",
    "한화에어로스페이스": "방산,우주항공",
    "한국항공우주": "방산,우주항공",
    "LIG넥스원": "방산",
    "현대로템": "방산",
    "한화시스템": "방산",
    "빅텍": "방산",
    "레인보우로보틱스": "로봇",
    "두산로보틱스": "로봇",
    "유진로봇": "로봇",
    "로보티즈": "로봇",
    "하이브": "엔터/IP",
    "에스엠": "엔터/IP",
    "JYP": "엔터/IP",
    "와이지엔터테인먼트": "엔터/IP",
    "CJ ENM": "엔터/IP",
    "두산에너빌리티": "원전/에너지",
    "한전KPS": "원전/에너지",
    "한국전력": "원전/에너지",
    "두산퓨얼셀": "원전/에너지,친환경",
    "클래시스": "의료기기",
    "오스템임플란트": "의료기기",
    "인바디": "의료기기",
    "뷰웍스": "의료기기",
    "현대차": "자율주행,전기차",
    "기아": "자율주행,전기차",
    "현대모비스": "자율주행",
    "만도": "자율주행",
    "한화솔루션": "친환경",
    "OCI": "친환경",
    "신성이엔지": "친환경",
    "세트렉아이": "우주항공",
    "쎄트렉아이": "우주항공",
    "켄코아에어로스페이스": "우주항공",
    "삼성바이오로직스": "바이오",
    "셀트리온": "바이오",
    "유한양행": "바이오",
    "한미약품": "바이오",
    "HLB": "바이오",
    "알테오젠": "바이오",
    "NAVER": "플랫폼",
    "네이버": "플랫폼",
    "카카오": "플랫폼",
    "크래프톤": "게임",
    "엔씨소프트": "게임",
    "넷마블": "게임",
    "펄어비스": "게임",
    # ★ 뷰티/화장품
    "에이피알": "뷰티",
    "아모레퍼시픽": "뷰티",
    "LG생활건강": "뷰티",
    "코스맥스": "뷰티",
    "한국콜마": "뷰티",
    "클리오": "뷰티",
    "실리콘투": "뷰티",
    "브이티": "뷰티",
    # ★ 바이오 추가
    "리가켐바이오": "바이오",
    "파마리서치": "바이오",
    "에스티팜": "바이오",
    "오리온바이오로직스": "바이오",
    # ★ 원전
    "한전기술": "원전/에너지",
    "우진": "원전/에너지",
    "비에이치아이": "원전/에너지",
}


class NewsSentiment:
    """
    v6.1 뉴스 감성 분석 + 요약 고도화 + 캐싱
    ★ KoBERT 감성 분석 (금융 특화)
    ★ 핵심 키워드 자동 추출 (상승/하락 근거 명시)
    ★ 테마주 자동 감지 (AI반도체/2차전지/방산 등)
    ★ 어닝 서프라이즈/쇼크 자동 포착
    ★ 뉴스 요약문 생성 (종목별 1~2줄)
    ★ 수치 포함 뉴스 가중 처리
    """

    POS_STRONG = [
        "어닝서프라이즈","깜짝실적","사상최대","역대최대","최대실적","급등",
        "신고가","목표가 상향","수주 대박","흑자전환","특허 획득","FDA 승인",
        "임상 성공","자사주 매입","대규모 수주","영업이익 급증","매출 사상최고",
        "상한가","IPO 흥행","독점 계약","글로벌 수주",
    ]
    POS_MID = [
        "실적개선","상승","호재","매수","상향","수주","계약","성장","돌파",
        "강세","호실적","기대감","매출 증가","이익","배당","호조","신제품",
        "협약","MOU","수출","호황","영업이익 증가","순이익 증가","수익성 개선",
        "점유율 확대","공급 계약","전략적 파트너","독점 공급",
    ]
    POS_WEAK = [
        "반등","저평가","매집","긍정","회복","개선","기대","모멘텀",
        "기회","주목","상향 조정","저점 매수","바닥 형성","턴어라운드",
    ]
    NEG_STRONG = [
        "어닝쇼크","실적쇼크","횡령","분식","상장폐지","파산","검찰 수사",
        "과징금","리콜","임상 실패","FDA 거절","급락","신저가","최대손실",
        "영업정지","거래정지","불성실공시","감사의견 거절",
    ]
    NEG_MID = [
        "하락","악재","매도","하향","적자","손실","소송","규제","약세",
        "부진","부채","경고","매출 감소","적자 전환","우려","리스크",
        "수사","영업이익 감소","순손실","시장점유율 하락",
    ]
    NEG_WEAK = [
        "조정","주의","불확실","하향 조정","감소","축소","둔화",
        "부담","압박","공급과잉","재고 증가",
    ]
    NEGATION = [
        "불구","에도 불구","에도 불구하고","우려에도","에도",
        "우려와 달리","기대 이상","예상 상회","예상 초과","예상 웃돌",
    ]

    def __init__(self, client_id=None, client_secret=None):
        self.client_id     = client_id
        self.client_secret = client_secret
        self._bert_pipe    = None
        self._bert_loaded  = False
        # ★ 당일 뉴스 캐시 (같은 종목 중복 수집 방지)
        self._cache        = {}
        self._try_load_bert()

    def _try_load_bert(self):
        try:
            from transformers import pipeline
            for model_name in [
                "snunlp/KR-FinBert-SC",
                "monologg/koelectra-base-finetuned-sentiment",
                "klue/roberta-base",
            ]:
                try:
                    self._bert_pipe = pipeline(
                        "text-classification", model=model_name,
                        tokenizer=model_name, device=-1,
                        truncation=True, max_length=128,
                    )
                    self._bert_loaded = True
                    print(f"[OK] 뉴스 감성 AI: {model_name} 로드 완료")
                    break
                except Exception:
                    continue
            if not self._bert_loaded:
                print("[주의] KoBERT 없음 → 키워드 감성분석 사용")
        except ImportError:
            print("[주의] transformers 없음 → 키워드 감성분석 사용")

    # ── 메인 ────────────────────────────────────────────────────────────────
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        scores, summaries, themes = [], [], []

        for _, row in df.iterrows():
            name = str(row.get("name", ""))
            code = str(row.get("code", ""))
            if not name or name in ("-", "—"):
                name = code

            # ★ 캐시 확인 (당일 같은 종목 중복 수집 방지)
            cache_key = code or name
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                scores.append(cached["score"])
                summaries.append(cached["summary"])
                themes.append(cached["theme"])
                continue

            articles = self._fetch_news(name)
            if not articles:
                scores.append(50.0)
                summaries.append("뉴스없음")
                # ★ 뉴스 없어도 코드/섹터 기반 테마 감지
                sector_val = str(row.get("sector", "")) if hasattr(row, "get") else ""
                code_val   = str(row.get("code",   "")) if hasattr(row, "get") else ""
                theme_no_news = self._detect_theme([], name, sector_val, code_val)
                themes.append(theme_no_news)
                self._cache[cache_key] = {
                    "score": 50.0,
                    "summary": "뉴스없음",
                    "theme": theme_no_news
                }
                continue

            # 감성 점수
            if self._bert_loaded:
                score, kw_summary = self._bert_score(articles)
            else:
                score, kw_summary = self._keyword_score(articles)

            # ★ 뉴스 요약문 생성
            rich_summary = self._build_summary(articles, kw_summary, score)

            # ★ 테마주 감지 (코드 + 종목명 + 섹터 4단계 감지)
            sector_val = str(row.get("sector", "")) if hasattr(row, "get") else ""
            code_val   = str(row.get("code",   "")) if hasattr(row, "get") else ""
            theme = self._detect_theme(articles, name, sector_val, code_val)

            # ★ 캐시 저장
            self._cache[cache_key] = {
                "score": score, "summary": rich_summary, "theme": theme
            }

            scores.append(score)
            summaries.append(rich_summary)
            themes.append(theme)

        df["sentiment_score"] = scores
        df["news_summary"]    = summaries
        df["theme_tag"]       = themes
        return df

    # ── 뉴스 수집 ───────────────────────────────────────────────────────────
    def _fetch_news(self, name: str) -> list:
        if not self.client_id or not self.client_secret:
            return self._fetch_naver_finance(name)
        try:
            resp = requests.get(
                "https://openapi.naver.com/v1/search/news.json",
                headers={"X-Naver-Client-Id": self.client_id,
                         "X-Naver-Client-Secret": self.client_secret},
                params={"query": name, "display": 10, "sort": "date"},
                timeout=5,
            )
            items = resp.json().get("items", [])
            return [
                f"{i.get('title','')} {i.get('description','')}".replace("<b>","").replace("</b>","")
                for i in items
            ]
        except Exception:
            return self._fetch_naver_finance(name)

    def _fetch_naver_finance(self, name: str) -> list:
        try:
            url  = f"https://finance.naver.com/item/news_news.naver?code={name}&page=1"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            from bs4 import BeautifulSoup
            soup  = BeautifulSoup(resp.text, "html.parser")
            items = soup.select("table.type5 tbody tr td.title a")
            return [a.get_text(strip=True) for a in items[:10]]
        except Exception:
            return []

    # ── KoBERT 감성 ─────────────────────────────────────────────────────────
    def _bert_score(self, articles: list):
        scores = []
        for text in articles[:5]:
            try:
                result = self._bert_pipe(text[:512])
                label  = result[0]["label"].upper()
                conf   = result[0]["score"]
                if   "POS" in label or "긍정" in label: scores.append(conf)
                elif "NEG" in label or "부정" in label: scores.append(-conf)
                else:                                    scores.append(0.0)
            except Exception:
                scores.append(0.0)
        if not scores:
            return 50.0, "분석실패"
        avg   = np.mean(scores)
        score = float(np.clip(50 + avg*40, 10, 90))
        if   avg > 0.3:  kw = "강한 긍정"
        elif avg > 0.1:  kw = "긍정"
        elif avg < -0.3: kw = "강한 부정"
        elif avg < -0.1: kw = "부정"
        else:            kw = "중립"
        return score, kw

    # ── 키워드 감성 (폴백) ───────────────────────────────────────────────────
    def _keyword_score(self, articles: list):
        total   = 0
        pos_kws = []
        neg_kws = []
        for text in articles:
            has_neg = any(n in text for n in self.NEGATION)
            for kw in self.POS_STRONG:
                if kw in text:
                    total += 3 if not has_neg else 1
                    pos_kws.append(kw)
            for kw in self.POS_MID:
                if kw in text:
                    total += 2 if not has_neg else 0
                    pos_kws.append(kw)
            for kw in self.POS_WEAK:
                if kw in text:
                    total += 1
            for kw in self.NEG_STRONG:
                if kw in text:
                    total += -3 if not has_neg else 1
                    neg_kws.append(kw)
            for kw in self.NEG_MID:
                if kw in text:
                    total += -2 if not has_neg else 0
                    neg_kws.append(kw)
            for kw in self.NEG_WEAK:
                if kw in text:
                    total += -1
            if re.search(r'\d+\.?\d*[%억원조]', text):
                total = int(total * 1.3)

        final = float(np.clip(50 + total * 3, 10, 90))
        if total > 3:
            kw = f"긍정({','.join(list(dict.fromkeys(pos_kws))[:2])})"
        elif total < -3:
            kw = f"부정({','.join(list(dict.fromkeys(neg_kws))[:2])})"
        else:
            kw = "중립"
        return final, kw

    # ── ★ 뉴스 요약문 생성 ──────────────────────────────────────────────────
    def _build_summary(self, articles: list, kw_summary: str, score: float) -> str:
        """
        뉴스 전체에서 핵심 문장 추출 + 감성 방향 + 어닝/수주 수치 포함
        """
        if not articles:
            return "뉴스없음"

        highlights = []

        # 수치 포함 문장 우선 추출
        for text in articles[:8]:
            nums = re.findall(r'\d+\.?\d*[%억원조]', text)
            if nums:
                # 가장 짧은 의미 있는 구절 추출
                for sent in re.split(r'[.。…\n]', text):
                    sent = sent.strip()
                    if len(sent) >= 8 and any(n in sent for n in nums):
                        highlights.append(sent[:40])
                        break

        # 강한 키워드 포함 문장 추출
        for text in articles[:5]:
            for kw in self.POS_STRONG + self.NEG_STRONG:
                if kw in text:
                    for sent in re.split(r'[.。…\n]', text):
                        sent = sent.strip()
                        if kw in sent and 5 <= len(sent) <= 50:
                            highlights.append(sent)
                            break

        # 중복 제거 및 최대 2개
        seen = []
        for h in highlights:
            if h not in seen:
                seen.append(h)
        highlights = seen[:2]

        if score >= 70:   direction = "📈"
        elif score >= 60: direction = "🟢"
        elif score <= 30: direction = "📉"
        elif score <= 40: direction = "🔴"
        else:             direction = "⚪"

        if highlights:
            return f"{direction} {kw_summary} | {' / '.join(highlights)}"
        else:
            # 첫 기사 제목 일부
            first = articles[0][:35].strip() if articles else ""
            return f"{direction} {kw_summary}" + (f" | {first}" if first else "")

    # ── ★ 테마주 감지 ────────────────────────────────────────────────────────
    def _detect_theme(self, articles: list, stock_name: str = "",
                      sector: str = "", stock_code: str = "") -> str:
        """
        테마 감지 - 4단계
        1순위: 종목 코드 기반 (CODE_THEME_MAP) - 가장 정확
        2순위: 종목명 직접 매핑 (STOCK_THEME_MAP)
        3순위: 섹터 자동 매핑 (SECTOR_THEME_MAP)
        4순위: 섹터명 그대로 표시
        ★ 뉴스 키워드 감지 제거 (오분류 방지)
        """
        found = []

        # 1순위: 종목 코드 기반 (가장 정확)
        if stock_code and stock_code in CODE_THEME_MAP:
            for t in CODE_THEME_MAP[stock_code].split(","):
                if t and t not in found:
                    found.append(t)

        # 2순위: 종목명 직접 매핑
        if len(found) == 0 and stock_name and stock_name in STOCK_THEME_MAP:
            for t in STOCK_THEME_MAP[stock_name].split(","):
                if t and t not in found:
                    found.append(t)

        # 3순위: 섹터 자동 매핑
        if len(found) == 0 and sector and sector in SECTOR_THEME_MAP:
            t = SECTOR_THEME_MAP[sector]
            if t and t not in found:
                found.append(t)

        # 4순위: 섹터명 그대로 표시 (기타/ETF 제외)
        if len(found) == 0 and sector and sector not in ["기타", "", "ETF/인덱스"]:
            found.append(sector)

        return ",".join(found[:3]) if found else ""