import sys as _sys
if _sys.platform == "win32":
    try:
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pandas as pd
import numpy as np
import requests
import time
import warnings
warnings.filterwarnings('ignore')

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


# ══════════════════════════════════════════════════════════════════════════════
# 방법 1: 네이버 main.naver — PER/PBR/EPS/ROE (확인된 셀렉터)
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_naver_main(code: str) -> dict:
    result = _empty()
    try:
        from bs4 import BeautifulSoup
        url  = f"https://finance.naver.com/item/main.naver?code={code}"
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        # em#_per, em#_pbr, em#_eps (직접 확인된 셀렉터)
        for field, em_id in [("per","_per"),("pbr","_pbr"),
                              ("eps","_eps"),("bps","_bps")]:
            tag = soup.find("em", id=em_id)
            if tag:
                result[field] = _safe_float(tag.get_text())

        # ROE — th(ROE(%)) 옆 td (직접 확인된 방식)
        for th in soup.find_all("th"):
            th_txt = th.get_text(strip=True)
            if "ROE" in th_txt:
                td = th.find_next_sibling("td")
                if td:
                    val = _safe_float(td.get_text(strip=True))
                    if val != 0:
                        if "%" in th_txt:
                            result["roe"] = val
                            break
                        elif result["roe"] == 0:
                            result["roe"] = val

        # ROE 보완: EPS/BPS 계산
        if result["roe"] == 0 and result["bps"] > 0 and result["eps"] != 0:
            result["roe"] = round(result["eps"] / result["bps"] * 100, 2)

        return result
    except Exception:
        return result


# ══════════════════════════════════════════════════════════════════════════════
# 방법 2: wisereport JSON API — ROA/영업이익률/부채비율/매출증가율
# ★ JavaScript 동적 데이터를 JSON API로 직접 수집
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_wisereport(code: str) -> dict:
    """
    wisereport.co.kr JSON API로 ROA/영업이익률/부채비율/매출증가율 수집
    네이버 재무분석 iframe 데이터 소스
    """
    result = _empty()
    try:
        # wisereport 재무비율 API
        urls = [
            f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}",
            f"https://comp.wisereport.co.kr/company/ajax/cF3002.aspx?cmp_cd={code}&fin_typ=0&freq_typ=Y",
            f"https://navercomp.wisereport.co.kr/v2/company/ajax/cF3002.aspx?cmp_cd={code}&fin_typ=0&freq_typ=Y",
        ]

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}",
            "X-Requested-With": "XMLHttpRequest",
        }

        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=8)
                if resp.status_code != 200:
                    continue

                # JSON 응답 시도
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        # 다양한 키 이름 시도
                        result["roa"]        = _safe_float(data.get("ROA") or data.get("roa"))
                        result["op_margin"]  = _safe_float(data.get("영업이익률") or
                                                           data.get("OPM") or
                                                           data.get("op_margin"))
                        result["debt_ratio"] = _safe_float(data.get("부채비율") or
                                                           data.get("DEBT") or
                                                           data.get("debt_ratio"))
                        result["rev_growth"] = _safe_float(data.get("매출증가율") or
                                                           data.get("REV_GROWTH"))
                        if any(result[k] != 0 for k in ["roa","op_margin","debt_ratio"]):
                            return result
                except Exception:
                    pass

                # HTML 응답 시도 (테이블 파싱)
                from bs4 import BeautifulSoup
                resp.encoding = "utf-8"
                soup = BeautifulSoup(resp.text, "html.parser")

                # 숫자 데이터가 있는 td 파싱
                for tr in soup.find_all("tr"):
                    tds = tr.find_all("td")
                    ths = tr.find_all("th")
                    if not ths or len(tds) < 2:
                        continue
                    label = ths[0].get_text(strip=True)
                    # 첫 번째 숫자 td (가장 최신)
                    for td in tds:
                        val_txt = td.get_text(strip=True).replace(",","").replace("%","")
                        try:
                            val = float(val_txt)
                            if val == 0:
                                continue
                            if   "ROA"    in label and result["roa"]        == 0:
                                result["roa"]        = val; break
                            elif "영업이익률" in label and result["op_margin"] == 0:
                                result["op_margin"]  = val; break
                            elif "부채"   in label and result["debt_ratio"] == 0:
                                result["debt_ratio"] = val; break
                            elif "매출" in label and "증가" in label and result["rev_growth"] == 0:
                                result["rev_growth"] = val; break
                        except Exception:
                            continue

                if any(result[k] != 0 for k in ["roa","op_margin","debt_ratio"]):
                    return result
            except Exception:
                continue

        return result
    except Exception:
        return result


# ══════════════════════════════════════════════════════════════════════════════
# 방법 3: 네이버 재무분석 페이지 크롤링
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_naver_finsum(code: str) -> dict:
    result = _empty()
    try:
        from bs4 import BeautifulSoup
        url  = f"https://finance.naver.com/item/coinfo.naver?code={code}&target=finsum_more"
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        for tbl in soup.select("table"):
            for tr in tbl.select("tr"):
                ths = tr.select("th")
                tds = tr.select("td")
                if not ths or not tds:
                    continue
                label = ths[0].get_text(strip=True)
                for td in tds:
                    val_txt = td.get_text(strip=True).replace(",","").replace("%","").strip()
                    try:
                        val = float(val_txt)
                        if val == 0:
                            continue
                        if   "ROA"    in label and result["roa"]        == 0:
                            result["roa"]        = val; break
                        elif "부채"   in label and result["debt_ratio"] == 0:
                            result["debt_ratio"] = val; break
                        elif "영업이익률" in label and result["op_margin"] == 0:
                            result["op_margin"]  = val; break
                        elif "매출" in label and "증가" in label and result["rev_growth"] == 0:
                            result["rev_growth"] = val; break
                        elif "ROE"    in label and result["roe"]         == 0:
                            result["roe"]        = val; break
                        elif "PER"    in label and result["per"]         == 0:
                            result["per"]        = val; break
                    except Exception:
                        continue
        return result
    except Exception:
        return result


# ══════════════════════════════════════════════════════════════════════════════
# 방법 4: yfinance 최후 폴백
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_yfinance(code: str) -> dict:
    result = _empty()
    try:
        import yfinance as yf
        for suffix in [".KS", ".KQ"]:
            try:
                info = yf.Ticker(f"{code}{suffix}").info
                if not info:
                    continue
                result["per"]        = _safe_float(info.get("trailingPE"))
                result["pbr"]        = _safe_float(info.get("priceToBook"))
                result["roe"]        = _safe_float(info.get("returnOnEquity"),  mul=100)
                result["roa"]        = _safe_float(info.get("returnOnAssets"),  mul=100)
                result["op_margin"]  = _safe_float(info.get("operatingMargins"), mul=100)
                result["rev_growth"] = _safe_float(info.get("revenueGrowth"),   mul=100)
                result["eps"]        = _safe_float(info.get("trailingEps"))
                # ★ 시가총액 수집
                mc = info.get("marketCap")
                if mc and float(mc) > 0:
                    result["market_cap"] = float(mc)
                if result["per"] > 0 or result["pbr"] > 0:
                    return result
            except Exception:
                continue
    except Exception:
        pass
    return result


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def _empty() -> dict:
    return {
        "per": 0.0, "pbr": 0.0, "roe": 0.0, "roa": 0.0,
        "market_cap": 0.0,
        "debt_ratio": 0.0, "op_margin": 0.0, "rev_growth": 0.0,
        "eps": 0.0, "bps": 0.0,
    }

def _safe_float(val, mul: float = 1.0) -> float:
    try:
        if val is None:
            return 0.0
        s = str(val).replace(",","").replace("%","").replace("배","")\
                    .replace("원","").strip()
        if s in ("", "N/A", "-", "null", "nan", "None"):
            return 0.0
        f = float(s)
        if not np.isfinite(f):
            return 0.0
        return round(f * mul, 2)
    except Exception:
        return 0.0

def _merge(base: dict, extra: dict) -> dict:
    for k in base:
        if base.get(k, 0.0) == 0.0 and extra.get(k, 0.0) != 0.0:
            base[k] = extra[k]
    return base


# ══════════════════════════════════════════════════════════════════════════════
# 메인 클래스
# ══════════════════════════════════════════════════════════════════════════════
class FundamentalAnalyzer:
    """
    재무 분석 v6.1
    ★ 1순위: 네이버 main.naver (PER/PBR/ROE/EPS — 직접 확인된 셀렉터)
    ★ 2순위: wisereport JSON API (ROA/영업이익률/부채비율/매출증가율)
    ★ 3순위: 네이버 재무분석 페이지 크롤링 보완
    ★ 4순위: yfinance 최후 폴백
    ★ pykrx 완전 제거
    """

    def fetch_and_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df    = df.copy()
        codes = df["code"].tolist()

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def fetch_one(code):
            # 1순위: 네이버 main (PER/PBR/ROE)
            data = _fetch_naver_main(code)

            # 2순위: wisereport (ROA/영업이익률/부채비율)
            if data["roa"] == 0 or data["op_margin"] == 0 or data["debt_ratio"] == 0:
                wise = _fetch_wisereport(code)
                data = _merge(data, wise)

            # 3순위: 네이버 재무분석 페이지 보완
            if data["roa"] == 0 or data["op_margin"] == 0:
                fin = _fetch_naver_finsum(code)
                data = _merge(data, fin)

            # 4순위: yfinance 최후 폴백
            if data["per"] == 0 and data["pbr"] == 0 and data["roe"] == 0:
                yf_d = _fetch_yfinance(code)
                data = _merge(data, yf_d)

            # ROE 최종 계산
            if data["roe"] == 0 and data["bps"] > 0 and data["eps"] != 0:
                data["roe"] = round(data["eps"] / data["bps"] * 100, 2)

            time.sleep(0.1)
            return code, data

        print(f"[재무] 데이터 수집 중... ({len(codes)}개)")
        fund_data = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(fetch_one, c): c for c in codes}
            done = 0
            for future in as_completed(futures):
                code, data = future.result()
                fund_data[code] = data
                done += 1
                if done % 20 == 0:
                    print(f"  재무 진행: {done}/{len(codes)}")

        col_keys = ["per","pbr","roe","roa","debt_ratio","op_margin","rev_growth","eps","bps"]
        cols = {k: [] for k in col_keys + ["fundamental_score"]}

        for _, row in df.iterrows():
            code = str(row.get("code", ""))
            d    = fund_data.get(code, _empty())

            per  = _safe_float(d.get("per"))
            pbr  = _safe_float(d.get("pbr"))
            roe  = _safe_float(d.get("roe"))
            roa  = _safe_float(d.get("roa"))
            debt = _safe_float(d.get("debt_ratio"))
            opm  = _safe_float(d.get("op_margin"))
            revg = _safe_float(d.get("rev_growth"))
            eps  = _safe_float(d.get("eps"))
            bps  = _safe_float(d.get("bps"))

            cols["per"].append(per);         cols["pbr"].append(pbr)
            cols["roe"].append(roe);         cols["roa"].append(roa)
            cols["debt_ratio"].append(debt); cols["op_margin"].append(opm)
            cols["rev_growth"].append(revg); cols["eps"].append(eps)
            cols["bps"].append(bps)
            cols["fundamental_score"].append(
                self._calc_score(per, pbr, roe, roa, debt, opm, revg)
            )

            # ★ 시가총액 yfinance 보완 (pykrx 실패 시)
            if "market_cap" in d and d["market_cap"] > 0:
                cur_cap = float(row.get("market_cap", 0) or 0)
                if cur_cap <= 0:
                    df.at[row.name, "market_cap"] = d["market_cap"]

        for col, vals in cols.items():
            df[col] = vals

        n     = len(codes)
        per_n = sum(1 for v in cols["per"]       if v > 0)
        roe_n = sum(1 for v in cols["roe"]       if v != 0)
        roa_n = sum(1 for v in cols["roa"]       if v != 0)
        opm_n = sum(1 for v in cols["op_margin"] if v != 0)
        print(f"  [재무완료] PER={per_n}/{n} ROE={roe_n}/{n} "
              f"ROA={roa_n}/{n} 영업이익률={opm_n}/{n}")
        return df

    def _calc_score(self, per, pbr, roe, roa, debt, opm, revg) -> float:
        score = 50.0

        if   0 < per <= 8:   score += 22
        elif 0 < per <= 12:  score += 17
        elif 0 < per <= 15:  score += 12
        elif 0 < per <= 20:  score += 6
        elif per > 60:       score -= 15
        elif per > 40:       score -= 10
        elif per > 30:       score -= 5

        if   0 < pbr <= 0.5: score += 20
        elif 0 < pbr <= 1.0: score += 14
        elif 0 < pbr <= 1.5: score += 6
        elif pbr > 5.0:      score -= 15
        elif pbr > 4.0:      score -= 10
        elif pbr > 3.0:      score -= 5

        if   roe >= 25:      score += 18
        elif roe >= 20:      score += 14
        elif roe >= 15:      score += 9
        elif roe >= 10:      score += 4
        elif roe < -10:      score -= 20
        elif roe < 0:        score -= 12
        elif roe < 5:        score -= 4

        if   roa >= 15:      score += 8
        elif roa >= 10:      score += 5
        elif roa >= 5:       score += 2
        elif roa < -5:       score -= 10
        elif roa < 0:        score -= 6

        if   0 < debt <= 30: score += 8
        elif 0 < debt <= 60: score += 4
        elif debt > 300:     score -= 15
        elif debt > 200:     score -= 10
        elif debt > 150:     score -= 5

        if   opm >= 20:      score += 8
        elif opm >= 15:      score += 5
        elif opm >= 10:      score += 2
        elif opm < -10:      score -= 12
        elif opm < 0:        score -= 7

        if   revg >= 30:     score += 8
        elif revg >= 20:     score += 5
        elif revg >= 10:     score += 2
        elif revg < -20:     score -= 10
        elif revg < -10:     score -= 6

        return float(np.clip(score, 0, 100))