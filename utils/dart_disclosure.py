import pandas as pd
import numpy as np
import requests
import time
import warnings
warnings.filterwarnings('ignore')


class DartDisclosure:
    """
    DART 공시 분석 v2.0
    ★ 어닝 서프라이즈 감지 (분기 실적 추이 비교)
    ★ 흑자전환 / 적자전환 자동 감지
    ★ 기존 자사주/유상증자/배당 감지 유지
    """

    POS_DISCLOSURES = [
        "자기주식취득","자기주식처분","현금배당","주식배당","합병",
        "영업실적","수시공시","공급계약","업무협약","특허","신제품",
        "흑자전환","실적개선","매출증가","영업이익증가","수주","계약체결",
    ]
    NEG_DISCLOSURES = [
        "유상증자","전환사채","신주인수권","불성실공시","조회공시",
        "횡령","배임","소송","적자","감자","부도","워크아웃","기업회생",
        "영업손실","매출감소",
    ]

    def __init__(self, api_key: str = ""):
        self.api_key  = api_key
        self.use_api  = bool(api_key)
        self.base_url = "https://opendart.fss.or.kr/api"

    def fetch_and_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        scores, summaries, surprises = [], [], []

        for _, row in df.iterrows():
            code = str(row.get("code", ""))
            name = str(row.get("name", ""))
            score, summary = self._get_disclosure_score(code, name)
            surprise       = self._get_earnings_surprise(code, name)
            final_score    = float(np.clip(score + surprise * 0.3, 0, 100))
            scores.append(final_score)
            summaries.append(summary)
            surprises.append(surprise)
            time.sleep(0.05)

        df["dart_score"]        = scores
        df["dart_summary"]      = summaries
        df["earnings_surprise"] = surprises
        return df

    def _get_disclosure_score(self, code, name):
        if self.use_api:
            return self._fetch_dart_api(code)
        return self._fetch_dart_crawl(name)

    def _fetch_dart_api(self, code):
        try:
            from datetime import datetime, timedelta
            end    = datetime.now().strftime("%Y%m%d")
            start  = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
            params = {
                "crtfc_key":self.api_key,"corp_code":code,
                "bgn_de":start,"end_de":end,"sort":"date",
                "sort_mth":"desc","page_no":1,"page_count":10,
            }
            resp  = requests.get(f"{self.base_url}/list.json", params=params, timeout=5)
            data  = resp.json()
            if data.get("status") != "000":
                return 50.0, "공시 없음"
            items   = data.get("list", [])
            titles  = [i.get("report_nm","") for i in items]
            score   = self._calc_score(titles)
            summary = titles[0][:40]+"..." if titles else "공시 없음"
            return score, summary
        except Exception:
            return 50.0, "공시 로드 실패"

    def _fetch_dart_crawl(self, name):
        try:
            url     = f"https://dart.fss.or.kr/dsab001/search.ax?textCrpNm={name}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp    = requests.get(url, headers=headers, timeout=5)
            from bs4 import BeautifulSoup
            soup    = BeautifulSoup(resp.text, "html.parser")
            titles  = [td.text.strip() for td in soup.select(".tbList td.tL")][:10]
            if not titles:
                return 50.0, "공시 없음"
            import re
            summary = re.sub(r'<[^>]+>', '', titles[0])[:40] + "..."
            return self._calc_score(titles), summary
        except Exception:
            return 50.0, "공시 로드 실패"

    def _calc_score(self, titles):
        if not titles:
            return 50.0
        score = 50.0
        for title in titles:
            for kw in self.POS_DISCLOSURES:
                if kw in title:
                    score += 20 if "자기주식취득" in title else 10
            for kw in self.NEG_DISCLOSURES:
                if kw in title:
                    score -= 20 if "유상증자" in title else 10
        return float(np.clip(score, 0, 100))

    # ── 어닝 서프라이즈 ───────────────────────────────────────────

    def _get_earnings_surprise(self, code, name):
        if self.use_api:
            return self._fetch_earnings_api(code)
        return self._fetch_earnings_crawl(name)

    def _fetch_earnings_api(self, code):
        try:
            from datetime import datetime
            year    = datetime.now().year
            results = []
            for y in [year, year-1]:
                for reprt in ["11013","11012","11014","11011"]:
                    params = {
                        "crtfc_key":self.api_key,"corp_code":code,
                        "bsns_year":str(y),"reprt_code":reprt,"fs_div":"CFS",
                    }
                    resp = requests.get(f"{self.base_url}/fnlttSinglAcnt.json",
                                        params=params, timeout=5)
                    data = resp.json()
                    if data.get("status") == "000":
                        profit = self._extract_profit(data.get("list",[]))
                        if profit is not None:
                            results.append(profit)
                    if len(results) >= 3:
                        break
                if len(results) >= 3:
                    break

            if len(results) < 2:
                return 0.0
            recent, prev = results[0], results[1]
            if prev == 0:
                return 0.0
            if prev < 0 and recent > 0:
                return 40.0
            if prev > 0 and recent < 0:
                return -40.0
            return float(np.clip((recent-prev)/abs(prev)*100*0.5, -50, 50))
        except Exception:
            return 0.0

    def _fetch_earnings_crawl(self, name):
        try:
            url     = f"https://finance.naver.com/item/main.naver?query={name}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp    = requests.get(url, headers=headers, timeout=5)
            from bs4 import BeautifulSoup
            soup    = BeautifulSoup(resp.text, "html.parser")
            els     = soup.select(".cmp-table-cell.td0301")
            profits = []
            for el in els[:4]:
                text = el.text.strip().replace(",","").replace("-","0")
                try:
                    profits.append(float(text))
                except Exception:
                    continue
            if len(profits) < 2:
                return 0.0
            recent, prev = profits[0], profits[1]
            if prev == 0:
                return 0.0
            return float(np.clip((recent-prev)/abs(prev)*100*0.5, -50, 50))
        except Exception:
            return 0.0

    def _extract_profit(self, items):
        for item in items:
            if "영업이익" in str(item.get("account_nm","")):
                try:
                    return float(str(item.get("thstrm_amount","0") or "0").replace(",",""))
                except Exception:
                    continue
        return None