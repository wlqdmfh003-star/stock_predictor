import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class EarningsCalendar:
    """
    실적 발표 캘린더 v6.0
    ★ DART API / 네이버 크롤링으로 실적 발표 예정일 수집
    ★ 발표 전날 → 기대감 점수 가산
    ★ 어닝 서프라이즈 / 쇼크 자동 감지
    ★ 실적 시즌 (1/4/7/10월) 자동 반영
    ★ SQLite DB 캐싱으로 중복 요청 방지
    """

    # 실적 시즌 월
    EARNINGS_MONTHS = [1, 4, 7, 10]

    def __init__(self, dart_api_key: str = ""):
        self.dart_key = dart_api_key
        self.db       = EarningsDB()

    def fetch_and_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df   = df.copy()
        today = datetime.now()
        scores, notes = [], []

        for _, row in df.iterrows():
            code = str(row.get("code", ""))
            name = str(row.get("name", ""))

            base_score = 50.0
            note       = ""

            # ── 1. 실적 시즌 기본 점수 ───────────────────────────────────────
            if today.month in self.EARNINGS_MONTHS:
                base_score += 5
                note += "실적시즌 "

            # ── 2. DB 캐시 확인 ──────────────────────────────────────────────
            cached = self.db.get_earnings(code)
            if cached:
                score, n = self._score_from_cache(cached, today)
                base_score += score
                note       += n
            else:
                # ── 3. 실적 발표일 수집 ──────────────────────────────────────
                earn_date = self._fetch_earnings_date(code, name)
                if earn_date:
                    self.db.save_earnings(code, name, earn_date)
                    score, n = self._score_from_date(earn_date, today)
                    base_score += score
                    note       += n

            # ── 4. 어닝 서프라이즈 기록 확인 ─────────────────────────────────
            surprise = self.db.get_surprise(code)
            if surprise is not None:
                if   surprise > 20:  base_score += 15; note += "어닝서프라이즈↑ "
                elif surprise > 10:  base_score += 8;  note += "실적상회↑ "
                elif surprise < -20: base_score -= 15; note += "어닝쇼크↓ "
                elif surprise < -10: base_score -= 8;  note += "실적하회↓ "

            scores.append(float(np.clip(base_score, 0, 100)))
            notes.append(note.strip() or "해당없음")
            time.sleep(0.03)

        df["earnings_score"] = scores
        df["earnings_note"]  = notes
        return df

    def _score_from_cache(self, cached: dict, today: datetime):
        try:
            earn_dt = datetime.strptime(cached["earn_date"], "%Y-%m-%d")
            return self._score_from_date(earn_dt, today)
        except Exception:
            return 0.0, ""

    def _score_from_date(self, earn_date: datetime, today: datetime):
        diff = (earn_date - today).days
        if   diff == 1:   return 20.0, "내일실적발표! "
        elif diff == 2:   return 15.0, "모레실적발표 "
        elif diff == 3:   return 10.0, "3일후실적발표 "
        elif diff <= 7:   return  5.0, "이번주실적발표 "
        elif diff == 0:   return 12.0, "오늘실적발표! "
        elif -3 <= diff < 0: return  0.0, "최근실적발표완료 "
        return 0.0, ""

    def _fetch_earnings_date(self, code: str, name: str) -> datetime:
        """실적 발표 예정일 수집 (DART API → 네이버 크롤링)"""
        if self.dart_key:
            dt = self._dart_earnings_date(code)
            if dt:
                return dt
        return self._naver_earnings_date(name)

    def _dart_earnings_date(self, code: str):
        """DART API로 다음 실적 발표일 조회"""
        try:
            today = datetime.now()
            end   = (today + timedelta(days=90)).strftime("%Y%m%d")
            start = today.strftime("%Y%m%d")
            params = {
                "crtfc_key": self.dart_key,
                "corp_code": code,
                "bgn_de":    start,
                "end_de":    end,
                "pblntf_ty": "A",  # 정기공시
                "sort":      "date",
                "sort_mth":  "asc",
                "page_no":   1,
                "page_count":10,
            }
            resp = requests.get(
                "https://opendart.fss.or.kr/api/list.json",
                params=params, timeout=5
            )
            data  = resp.json()
            items = data.get("list", [])
            for item in items:
                title = item.get("report_nm", "")
                if any(k in title for k in ["사업보고서","반기보고서","분기보고서"]):
                    date_str = item.get("rcept_dt", "")
                    if date_str:
                        return datetime.strptime(date_str, "%Y%m%d")
        except Exception:
            pass
        return None

    def _naver_earnings_date(self, name: str):
        """네이버 금융 실적 발표 예정 크롤링"""
        try:
            url  = f"https://finance.naver.com/item/main.naver?query={name}"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            resp.encoding = "euc-kr"
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            # 예정 실적 발표일 텍스트 탐색
            for tag in soup.find_all(text=True):
                if "실적발표" in str(tag) or "공시예정" in str(tag):
                    import re
                    m = re.search(r'(\d{4})[./](\d{1,2})[./](\d{1,2})', str(tag))
                    if m:
                        try:
                            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                        except Exception:
                            pass
        except Exception:
            pass
        # 실적 시즌 기반 다음 발표 예정일 추정
        return self._estimate_next_earnings()

    def _estimate_next_earnings(self) -> datetime:
        """실적 시즌 기반 다음 발표 예정일 추정"""
        today = datetime.now()
        for month in self.EARNINGS_MONTHS:
            year = today.year
            est  = datetime(year, month, 15)
            if est > today:
                return est
        return datetime(today.year + 1, 1, 15)

    def update_surprise(self, code: str, surprise_pct: float):
        """어닝 서프라이즈 결과 업데이트"""
        self.db.save_surprise(code, surprise_pct)


# ══════════════════════════════════════════════════════════════════════════════
# SQLite DB (실적 캘린더 캐시)
# ══════════════════════════════════════════════════════════════════════════════
class EarningsDB:
    """실적 발표 캘린더 SQLite 캐시"""

    DB_PATH = ".cache/earnings.db"

    def __init__(self):
        import os
        os.makedirs(".cache", exist_ok=True)
        self._init_db()

    def _conn(self):
        import sqlite3
        return sqlite3.connect(self.DB_PATH)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS earnings_calendar (
                    code       TEXT PRIMARY KEY,
                    name       TEXT,
                    earn_date  TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS earnings_surprise (
                    code        TEXT PRIMARY KEY,
                    surprise    REAL,
                    updated_at  TEXT
                )
            """)
            conn.commit()

    def get_earnings(self, code: str) -> dict:
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT name, earn_date, updated_at FROM earnings_calendar WHERE code=?",
                    (code,)
                ).fetchone()
                if row:
                    # 7일 이상 된 캐시는 무효
                    updated = datetime.strptime(row[2], "%Y-%m-%d")
                    if (datetime.now() - updated).days <= 7:
                        return {"name": row[0], "earn_date": row[1]}
        except Exception:
            pass
        return {}

    def save_earnings(self, code: str, name: str, earn_date: datetime):
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO earnings_calendar
                    (code, name, earn_date, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (code, name, earn_date.strftime("%Y-%m-%d"),
                      datetime.now().strftime("%Y-%m-%d")))
                conn.commit()
        except Exception:
            pass

    def get_surprise(self, code: str):
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT surprise FROM earnings_surprise WHERE code=?",
                    (code,)
                ).fetchone()
                if row:
                    return float(row[0])
        except Exception:
            pass
        return None

    def save_surprise(self, code: str, surprise: float):
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO earnings_surprise
                    (code, surprise, updated_at) VALUES (?, ?, ?)
                """, (code, surprise, datetime.now().strftime("%Y-%m-%d")))
                conn.commit()
        except Exception:
            pass