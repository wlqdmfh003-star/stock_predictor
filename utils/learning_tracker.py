import json
import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR     = ".cache"
DB_PATH       = os.path.join(CACHE_DIR, "learning.db")
RL_STATE_FILE = os.path.join(CACHE_DIR, "rl_state.json")
BAYESIAN_FILE = os.path.join(CACHE_DIR, "bayesian_state.json")

DEFAULT_WEIGHTS = {
    "lstm":0.12,"ensemble":0.10,"candle":0.08,"macro":0.07,
    "momentum":0.12,"sentiment":0.08,"institution":0.10,"volume":0.06,
    "fundamental":0.08,"dart":0.06,"short":0.04,"high52":0.04,
    "us_market":0.05,"sector":0.05,
}
MIN_SAMPLES = 50


class LearningTracker:
    """
    자기학습 시스템 v6.0
    ★ SQLite DB 저장 (JSON → DB 전환, 빠르고 안정적)
    ★ 미결 데이터 자동 처리 (2일 이상 된 것도 자동 업데이트)
    ★ DQN 강화학습 (Deep Q-Network, 연속 상태 공간)
    ★ 베이지안 가중치 최적화
    ★ 기간별 적중률 / 종목별 히스토리 검색
    """

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.db       = LearningDB()
        self.rl       = DQNAgent()
        self.bayesian = BayesianWeightOptimizer()
        self.rl.load()
        self.bayesian.load()

    # ── 예측 저장 ─────────────────────────────────────────────────────────────
    def save_predictions(self, df: pd.DataFrame) -> int:
        today   = datetime.now().strftime("%Y-%m-%d")
        records = []
        for _, row in df.head(50).iterrows():
            records.append({
                "date":           today,
                "code":           str(row.get("code","")),
                "name":           str(row.get("name","")),
                "predicted_prob": float(row.get("rise_prob", 50)),
                "actual_return":  None,
                "hit":            None,
                "market_phase":   str(row.get("market_phase","중립")),
                "factors":        {k: float(row.get(f"{k}_score", 50))
                                   for k in DEFAULT_WEIGHTS},
            })
        saved = self.db.save_predictions(today, records)
        return saved

    # ── 실제 결과 업데이트 (자동 미결 처리 포함) ─────────────────────────────
    def update_results(self, df_today: pd.DataFrame) -> int:
        # 오늘 분석 종목 등락률 (일부만 포함)
        c2chg = {}
        if "code" in df_today.columns and "change_pct" in df_today.columns:
            c2chg = dict(zip(df_today["code"].astype(str),
                             df_today["change_pct"]))

        # ★ 핵심 수정: 미결 종목 전체의 등락률을 yfinance로 직접 조회
        # df_today에 없는 종목도 커버
        updated = 0
        for delta in range(3):
            target_date = (datetime.now() - timedelta(days=delta+1)).strftime("%Y-%m-%d")
            pending     = self.db.get_pending(target_date)
            if not pending:
                continue

            # 미결 종목 중 c2chg에 없는 것들만 yfinance로 조회
            missing_codes = [r["code"] for r in pending
                             if r["code"] not in c2chg]
            if missing_codes:
                yf_chg = self._fetch_changes(missing_codes, target_date)
                c2chg.update(yf_chg)

            for rec in pending:
                code = rec["code"]
                chg  = c2chg.get(code)
                if chg is not None:
                    hit = int(float(chg) > 0)
                    self.db.update_result(rec["id"], float(chg), hit)
                    self.rl.update(
                        state  = self._make_state(rec),
                        action = 1 if rec["predicted_prob"] >= 60 else 0,
                        reward = float(chg),
                    )
                    updated += 1

        if updated > 0:
            self.rl.save()
        return updated

    def _fetch_changes(self, codes: list, date_str: str) -> dict:
        """
        yfinance로 특정 날짜의 등락률 일괄 조회
        미결 종목들의 실제 결과를 가져오기 위한 함수
        """
        result = {}
        try:
            import yfinance as yf
            from datetime import datetime, timedelta

            target = datetime.strptime(date_str, "%Y-%m-%d")
            start  = (target - timedelta(days=3)).strftime("%Y-%m-%d")
            end    = (target + timedelta(days=2)).strftime("%Y-%m-%d")

            # 배치로 한 번에 조회 (빠름)
            batch_size = 20
            for i in range(0, len(codes), batch_size):
                batch = codes[i:i+batch_size]
                # KOSPI/KOSDAQ 둘 다 시도
                tickers_ks = [f"{c}.KS" for c in batch]
                tickers_kq = [f"{c}.KQ" for c in batch]

                for tickers, suffix in [(tickers_ks, ".KS"), (tickers_kq, ".KQ")]:
                    try:
                        data = yf.download(
                            tickers, start=start, end=end,
                            progress=False, auto_adjust=True,
                            group_by="ticker"
                        )
                        if data is None or data.empty:
                            continue
                        for code in batch:
                            if code in result:
                                continue
                            ticker = f"{code}{suffix}"
                            try:
                                if len(batch) == 1:
                                    closes = data["Close"].dropna()
                                else:
                                    closes = data[ticker]["Close"].dropna()                                              if ticker in data.columns.get_level_values(0)                                              else None
                                if closes is None or len(closes) < 2:
                                    continue
                                # 해당 날짜 전후 등락률 계산
                                chg = float(
                                    (closes.iloc[-1] - closes.iloc[-2])
                                    / closes.iloc[-2] * 100
                                )
                                result[code] = chg
                            except Exception:
                                continue
                    except Exception:
                        continue
        except Exception as e:
            pass
        return result

    # ── 강화학습 추천 ─────────────────────────────────────────────────────────
    def rl_recommend(self, row: dict) -> dict:
        state            = self._make_state(row)
        action, q_vals   = self.rl.act(state)
        return {
            "action":     "매수" if action == 1 else "관망",
            "q_buy":      round(float(q_vals[1]), 3),
            "q_hold":     round(float(q_vals[0]), 3),
            "confidence": round(abs(float(q_vals[1]) - float(q_vals[0])) * 10, 1),
        }

    def _make_state(self, rec: dict) -> np.ndarray:
        """연속 상태 벡터 (DQN용)"""
        prob   = float(rec.get("predicted_prob", rec.get("rise_prob", 50)))
        phase  = str(rec.get("market_phase", "중립"))
        factors = rec.get("factors", {})
        state = np.array([
            prob / 100,
            1.0 if "탐욕" in phase else -1.0 if "공포" in phase else 0.0,
            float(factors.get("momentum",   50)) / 100,
            float(factors.get("institution",50)) / 100,
            float(factors.get("sentiment",  50)) / 100,
            float(factors.get("fundamental",50)) / 100,
            float(factors.get("lstm",       50)) / 100,
            float(factors.get("volume",     50)) / 100,
        ], dtype=np.float32)
        return state

    # ── 베이지안 가중치 최적화 ────────────────────────────────────────────────
    def calc_learned_weights(self) -> dict:
        completed = self.db.get_completed()
        if len(completed) < MIN_SAMPLES:
            return DEFAULT_WEIGHTS.copy()
        best_w = self.bayesian.optimize(completed, DEFAULT_WEIGHTS)
        self.db.save_weights(best_w, len(completed))
        return best_w

    def load_learned_weights(self) -> dict:
        w = self.db.load_weights()
        return w if w else DEFAULT_WEIGHTS.copy()

    def get_phase_weights(self, phase: str) -> dict:
        completed = self.db.get_completed()
        if len(completed) < 30:
            return DEFAULT_WEIGHTS.copy()
        phase_data = [d for d in completed
                      if ("강세" in phase and "강세" in str(d.get("market_phase",""))) or
                         ("약세" in phase and "약세" in str(d.get("market_phase",""))) or
                         ("중립" in phase)]
        if len(phase_data) < 15:
            return DEFAULT_WEIGHTS.copy()
        return self.bayesian.optimize(phase_data, DEFAULT_WEIGHTS)

    # ── 통계 ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        stats = self.db.get_stats()
        stats["rl_episodes"]   = self.rl.episodes
        stats["bayesian_iter"] = self.bayesian.iteration
        return stats

    def get_factor_accuracy(self) -> pd.DataFrame:
        completed = self.db.get_completed()
        if not completed:
            return pd.DataFrame()
        rows = []
        for factor in DEFAULT_WEIGHTS:
            scores = np.array([d["factors"].get(factor, 50) for d in completed])
            hits   = np.array([d["hit"] for d in completed])
            med    = np.median(scores)
            hh = hits[scores >= med].mean() * 100 if (scores >= med).sum() > 0 else 0
            lh = hits[scores < med].mean()  * 100 if (scores < med).sum()  > 0 else 0
            rows.append({"팩터":factor,"상위적중률":round(hh,1),
                         "하위적중률":round(lh,1),"예측기여도":round(hh-lh,1),
                         "현재가중치":DEFAULT_WEIGHTS[factor]})
        return pd.DataFrame(rows).sort_values("예측기여도", ascending=False)

    def get_rl_stats(self) -> dict:
        return {
            "episodes":    self.rl.episodes,
            "q_table_size":self.rl.state_dim,
            "epsilon":     round(self.rl.epsilon, 3),
            "avg_reward":  round(self.rl.avg_reward, 3),
            "model_type":  "DQN" if self.rl.has_dqn else "Q-Table",
        }

    # ── DB 히스토리 검색 ──────────────────────────────────────────────────────
    def search_history(self, code: str = None, days: int = 30) -> pd.DataFrame:
        return self.db.search(code=code, days=days)

    def get_period_accuracy(self, days: int = 30) -> dict:
        return self.db.period_accuracy(days=days)


# ══════════════════════════════════════════════════════════════════════════════
# SQLite DB
# ══════════════════════════════════════════════════════════════════════════════
class LearningDB:
    """자기학습 SQLite DB"""

    def __init__(self):
        self._init_db()

    def _conn(self):
        return sqlite3.connect(DB_PATH)

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    date           TEXT,
                    code           TEXT,
                    name           TEXT,
                    predicted_prob REAL,
                    actual_return  REAL,
                    hit            INTEGER,
                    market_phase   TEXT,
                    factors        TEXT,
                    created_at     TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_date ON predictions(date);
                CREATE INDEX IF NOT EXISTS idx_code ON predictions(code);

                CREATE TABLE IF NOT EXISTS learned_weights (
                    id         INTEGER PRIMARY KEY,
                    weights    TEXT,
                    samples    INTEGER,
                    updated_at TEXT
                );
            """)
            conn.commit()
        # 기존 JSON 데이터 마이그레이션
        self._migrate_from_json()

    def _migrate_from_json(self):
        """기존 learning_data.json → DB 마이그레이션"""
        json_path = os.path.join(CACHE_DIR, "learning_data.json")
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return
            with self._conn() as conn:
                count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                if count > 0:
                    return  # 이미 마이그레이션됨
                for rec in data:
                    if not isinstance(rec, dict):
                        continue
                    conn.execute("""
                        INSERT OR IGNORE INTO predictions
                        (date, code, name, predicted_prob, actual_return, hit,
                         market_phase, factors)
                        VALUES (?,?,?,?,?,?,?,?)
                    """, (
                        rec.get("date",""),
                        rec.get("code",""),
                        rec.get("name",""),
                        float(rec.get("predicted_prob", 50)),
                        rec.get("actual_return"),
                        rec.get("hit"),
                        rec.get("market_phase","중립"),
                        json.dumps(rec.get("factors",{}), ensure_ascii=False),
                    ))
                conn.commit()
                print(f"  [DB] JSON→DB 마이그레이션 완료 ({len(data)}건)")
        except Exception as e:
            pass

    def save_predictions(self, date: str, records: list) -> int:
        with self._conn() as conn:
            # 오늘 기존 데이터 삭제 (덮어쓰기)
            conn.execute("DELETE FROM predictions WHERE date=?", (date,))
            for rec in records:
                conn.execute("""
                    INSERT INTO predictions
                    (date, code, name, predicted_prob, market_phase, factors)
                    VALUES (?,?,?,?,?,?)
                """, (
                    date,
                    rec["code"], rec["name"],
                    rec["predicted_prob"],
                    rec.get("market_phase","중립"),
                    json.dumps(rec.get("factors",{}), ensure_ascii=False),
                ))
            conn.commit()
        return len(records)

    def get_pending(self, date: str) -> list:
        """미결(hit=NULL) 데이터 조회"""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, code, predicted_prob, market_phase, factors
                FROM predictions
                WHERE date=? AND hit IS NULL
            """, (date,)).fetchall()
        result = []
        for row in rows:
            try:
                factors = json.loads(row[4]) if row[4] else {}
            except Exception:
                factors = {}
            result.append({
                "id": row[0], "code": row[1],
                "predicted_prob": row[2],
                "market_phase": row[3],
                "factors": factors,
            })
        return result

    def update_result(self, row_id: int, actual_return: float, hit: int):
        with self._conn() as conn:
            conn.execute("""
                UPDATE predictions
                SET actual_return=?, hit=?
                WHERE id=?
            """, (actual_return, hit, row_id))
            conn.commit()

    def get_completed(self) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT code, predicted_prob, actual_return, hit,
                       market_phase, factors
                FROM predictions
                WHERE hit IS NOT NULL
                ORDER BY date DESC
                LIMIT 2000
            """).fetchall()
        result = []
        for row in rows:
            try:
                factors = json.loads(row[5]) if row[5] else {}
            except Exception:
                factors = {}
            result.append({
                "code": row[0], "predicted_prob": row[1],
                "actual_return": row[2], "hit": row[3],
                "market_phase": row[4], "factors": factors,
            })
        return result

    def get_stats(self) -> dict:
        with self._conn() as conn:
            total     = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE hit IS NOT NULL"
            ).fetchone()[0]
            if completed > 0:
                row = conn.execute("""
                    SELECT AVG(hit)*100, AVG(actual_return)
                    FROM predictions WHERE hit IS NOT NULL
                """).fetchone()
                hit_rate   = round(float(row[0] or 0), 1)
                avg_return = round(float(row[1] or 0), 2)
            else:
                hit_rate = avg_return = 0.0
        return {
            "total": total, "completed": completed,
            "hit_rate": hit_rate, "avg_return": avg_return,
            "ready":  completed >= MIN_SAMPLES,
            "needed": max(0, MIN_SAMPLES - completed),
        }

    def save_weights(self, weights: dict, samples: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM learned_weights")
            conn.execute("""
                INSERT INTO learned_weights (weights, samples, updated_at)
                VALUES (?,?,?)
            """, (json.dumps(weights, ensure_ascii=False), samples,
                  datetime.now().strftime("%Y-%m-%d %H:%M")))
            conn.commit()

    def load_weights(self) -> dict:
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT weights FROM learned_weights ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    return json.loads(row[0])
        except Exception:
            pass
        return {}

    def search(self, code: str = None, days: int = 30) -> pd.DataFrame:
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        if code:
            q = "SELECT date,name,predicted_prob,actual_return,hit FROM predictions WHERE code=? AND date>=? ORDER BY date DESC"
            params = (code, since)
        else:
            q = "SELECT date,name,predicted_prob,actual_return,hit FROM predictions WHERE date>=? AND hit IS NOT NULL ORDER BY date DESC LIMIT 200"
            params = (since,)
        with self._conn() as conn:
            rows = conn.execute(q, params).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["날짜","종목","예측확률","실제수익","적중"])
        df["적중"] = df["적중"].apply(lambda x: "✅" if x==1 else "❌" if x==0 else "-")
        return df

    def period_accuracy(self, days: int = 30) -> dict:
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with self._conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*), AVG(hit)*100, AVG(actual_return)
                FROM predictions
                WHERE date>=? AND hit IS NOT NULL
            """, (since,)).fetchone()
        return {
            "period":     f"최근 {days}일",
            "count":      int(row[0] or 0),
            "hit_rate":   round(float(row[1] or 0), 1),
            "avg_return": round(float(row[2] or 0), 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
# DQN 강화학습 에이전트
# ══════════════════════════════════════════════════════════════════════════════
class DQNAgent:
    """
    Deep Q-Network 에이전트
    ─ 상태: 8차원 연속 벡터 (확률/시장국면/팩터 등)
    ─ 행동: 0=관망, 1=매수
    ─ PyTorch 있으면 신경망 DQN
    ─ 없으면 Q-Table 폴백
    """
    STATE_DIM  = 8
    ACTION_DIM = 2

    def __init__(self, lr=0.001, gamma=0.9, epsilon=0.3, epsilon_min=0.05):
        self.lr          = lr
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = 0.995
        self.episodes    = 0
        self.avg_reward  = 0.0
        self._buf        = []
        self.has_dqn     = False
        self.state_dim   = self.STATE_DIM
        self.q_table     = {}  # 폴백용

        # PyTorch DQN 시도
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            self._nn    = nn
            self._build_model()
            self.has_dqn = True
        except ImportError:
            pass

    def _build_model(self):
        nn = self._nn
        self._model = nn.Sequential(
            nn.Linear(self.STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.ACTION_DIM),
        )
        self._target = nn.Sequential(
            nn.Linear(self.STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.ACTION_DIM),
        )
        self._optimizer = self._torch.optim.Adam(
            self._model.parameters(), lr=self.lr)
        self._loss_fn = self._nn.MSELoss()
        self._update_target()

    def _update_target(self):
        if self.has_dqn:
            self._target.load_state_dict(self._model.state_dict())

    def act(self, state) -> tuple:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.ACTION_DIM)
            q_vals = [0.0, 0.0]
        else:
            if self.has_dqn:
                with self._torch.no_grad():
                    s   = self._torch.FloatTensor(state).unsqueeze(0)
                    q   = self._model(s).squeeze().numpy()
                action = int(np.argmax(q))
                q_vals = q.tolist()
            else:
                key    = self._discretize(state)
                q_vals = self.q_table.get(key, [0.0, 0.0])
                action = int(np.argmax(q_vals))
        return action, q_vals

    def update(self, state, action: int, reward: float,
               next_state=None):
        if next_state is None:
            next_state = state

        self._buf.append((state, action, reward, next_state))
        if len(self._buf) > 1000:
            self._buf.pop(0)

        self.episodes   += 1
        self.epsilon     = max(self.epsilon_min, self.epsilon * self.epsilon_dec)
        self._buf_avg    = getattr(self, "_buf_avg", [])
        self._buf_avg.append(reward)
        if len(self._buf_avg) > 100:
            self._buf_avg.pop(0)
        self.avg_reward = float(np.mean(self._buf_avg))

        if self.has_dqn and len(self._buf) >= 32:
            self._train_dqn()
        else:
            self._update_qtable(state, action, reward, next_state)

        if self.episodes % 50 == 0:
            self._update_target()

    def _train_dqn(self):
        """배치 학습"""
        try:
            batch  = [self._buf[i] for i in
                      np.random.randint(0, len(self._buf), 32)]
            states  = self._torch.FloatTensor([b[0] for b in batch])
            actions = self._torch.LongTensor([b[1] for b in batch])
            rewards = self._torch.FloatTensor([b[2] for b in batch])
            nexts   = self._torch.FloatTensor([b[3] for b in batch])

            with self._torch.no_grad():
                next_q  = self._target(nexts).max(1)[0]
                targets = rewards + self.gamma * next_q

            curr_q = self._model(states).gather(1, actions.unsqueeze(1)).squeeze()
            loss   = self._loss_fn(curr_q, targets)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        except Exception:
            pass

    def _update_qtable(self, state, action, reward, next_state):
        key      = self._discretize(state)
        next_key = self._discretize(next_state)
        if key not in self.q_table:
            self.q_table[key] = [0.0, 0.0]
        if next_key not in self.q_table:
            self.q_table[next_key] = [0.0, 0.0]
        old_q   = self.q_table[key][action]
        max_next = max(self.q_table[next_key])
        self.q_table[key][action] = round(
            old_q + self.lr * (reward + self.gamma * max_next - old_q), 4)

    def _discretize(self, state) -> str:
        """연속 상태 → 이산화 (Q-Table 폴백용)"""
        if hasattr(state, '__iter__'):
            arr = np.array(state, dtype=float)
            return "_".join(["H" if v > 0.6 else "L" if v < 0.4 else "M"
                              for v in arr[:4]])
        return str(round(float(state), 1))

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        state = {
            "episodes":   self.episodes,
            "epsilon":    self.epsilon,
            "avg_reward": self.avg_reward,
            "q_table":    self.q_table,
        }
        if self.has_dqn:
            try:
                self._torch.save(
                    self._model.state_dict(),
                    os.path.join(CACHE_DIR, "dqn_model.pt")
                )
            except Exception:
                pass
        with open(RL_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(RL_STATE_FILE):
            try:
                with open(RL_STATE_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.episodes   = d.get("episodes",   0)
                self.epsilon    = d.get("epsilon",     0.3)
                self.avg_reward = d.get("avg_reward",  0.0)
                self.q_table    = d.get("q_table",     {})
            except Exception:
                pass
        if self.has_dqn:
            try:
                mp = os.path.join(CACHE_DIR, "dqn_model.pt")
                if os.path.exists(mp):
                    self._model.load_state_dict(
                        self._torch.load(mp, map_location="cpu"))
                    self._update_target()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# 베이지안 가중치 최적화
# ══════════════════════════════════════════════════════════════════════════════
class BayesianWeightOptimizer:

    def __init__(self, n_iter=20):
        self.n_iter    = n_iter
        self.history   = []
        self.iteration = 0

    def optimize(self, completed: list, base_weights: dict) -> dict:
        factor_names = list(base_weights.keys())
        best_w       = base_weights.copy()
        best_score   = self._evaluate(base_weights, completed)

        for _ in range(self.n_iter):
            if self.history and np.random.random() > 0.4:
                top5 = sorted(self.history, key=lambda x: x[1], reverse=True)[:5]
                avg_w = {k: float(np.mean([h[0].get(k, base_weights[k])
                                           for h in top5]))
                         for k in factor_names}
                candidate = {k: max(0.01, avg_w[k] + np.random.uniform(-0.03, 0.03))
                             for k in factor_names}
            else:
                candidate = {k: max(0.01, base_weights[k] + np.random.uniform(-0.05, 0.05))
                             for k in factor_names}

            total     = sum(candidate.values())
            candidate = {k: round(v/total, 4) for k, v in candidate.items()}
            score     = self._evaluate(candidate, completed)
            self.history.append((candidate, score))
            if score > best_score:
                best_score = score
                best_w     = candidate.copy()

        self.iteration += 1
        total = sum(best_w.values())
        return {k: round(v/total, 4) for k, v in best_w.items()}

    def _evaluate(self, weights: dict, completed: list) -> float:
        if not completed:
            return 0.0
        scores, hits = [], []
        for rec in completed:
            factors = rec.get("factors", {})
            s = sum(weights.get(k, 0) * float(factors.get(k, 50))
                    for k in weights)
            scores.append(s)
            hits.append(rec["hit"])
        scores = np.array(scores)
        hits   = np.array(hits)
        med    = np.median(scores)
        hh = hits[scores >= med].mean() if (scores >= med).sum() > 0 else 0.5
        lh = hits[scores <  med].mean() if (scores <  med).sum() > 0 else 0.5
        try:
            ic = float(np.corrcoef(scores, hits)[0, 1])
        except Exception:
            ic = 0.0
        return float(hh - lh + ic * 0.1)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        top50 = sorted(self.history, key=lambda x: x[1], reverse=True)[:50]
        with open(BAYESIAN_FILE, "w", encoding="utf-8") as f:
            json.dump({"history": top50, "iteration": self.iteration},
                      f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(BAYESIAN_FILE):
            try:
                with open(BAYESIAN_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.history   = d.get("history",   [])
                self.iteration = d.get("iteration", 0)
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════════════════
    # ★ 섹터별 분리 자기학습 (200건 이상 시 활성)
    # ══════════════════════════════════════════════════════════════════════════
    def calc_sector_weights(self, sector: str) -> dict:
        """
        섹터별 최적 가중치 계산
        - 반도체/바이오/방산 등 섹터마다 다른 팩터가 중요
        - 200건 이상 데이터 있을 때 활성화
        - 없으면 전체 가중치 반환
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            df   = pd.read_sql_query(
                "SELECT * FROM predictions WHERE sector=? AND result IS NOT NULL",
                conn, params=(sector,)
            )
            conn.close()

            if len(df) < 30:
                return self.calc_learned_weights()

            # 섹터별 팩터 중요도 계산
            completed = df[df["result"].notna()].copy()
            if len(completed) < 10:
                return self.calc_learned_weights()

            # 각 팩터별 IC 계산 (팩터점수 vs 실제 수익률 상관관계)
            factor_cols = [
                "momentum_score","sentiment_score","fundamental_score",
                "ensemble_score","lstm_score","sector_score","dart_score",
                "short_score","high52_score","macro_score","institution_score",
            ]
            ic_dict = {}
            for col in factor_cols:
                if col in completed.columns and "change_1d" in completed.columns:
                    try:
                        ic = float(completed[col].astype(float).corr(
                            completed["change_1d"].astype(float)))
                        ic_dict[col] = ic if np.isfinite(ic) else 0.0
                    except:
                        ic_dict[col] = 0.0

            if not ic_dict:
                return self.calc_learned_weights()

            # IC 기반 가중치 조정
            base_w = self.calc_learned_weights()
            key_map = {
                "momentum_score":    "momentum",
                "sentiment_score":   "sentiment",
                "fundamental_score": "fundamental",
                "ensemble_score":    "ensemble",
                "lstm_score":        "lstm",
                "sector_score":      "sector",
                "dart_score":        "dart",
                "short_score":       "short",
                "high52_score":      "high52",
                "macro_score":       "macro",
                "institution_score": "institution",
            }

            sector_w = dict(base_w)
            for col, ic in ic_dict.items():
                key = key_map.get(col)
                if key and key in sector_w:
                    # IC 양수 → 가중치 UP / 음수 → 가중치 DOWN (최대 ±40%)
                    adj = 1.0 + np.clip(ic * 2, -0.4, 0.4)
                    sector_w[key] = float(sector_w[key] * adj)

            # 합계 정규화
            total = sum(sector_w.values())
            if total > 0:
                sector_w = {k: round(v/total, 4) for k,v in sector_w.items()}

            print(f"[섹터학습] {sector}: {len(completed)}건 기반 가중치 적용")
            return sector_w

        except Exception as e:
            return self.calc_learned_weights()

    def get_sector_stats(self) -> dict:
        """섹터별 자기학습 현황"""
        try:
            conn   = sqlite3.connect(DB_PATH)
            df_all = pd.read_sql_query(
                "SELECT sector, COUNT(*) as cnt, "
                "AVG(CASE WHEN result='상승' THEN 1 ELSE 0 END) as hit "
                "FROM predictions WHERE sector IS NOT NULL "
                "GROUP BY sector ORDER BY cnt DESC",
                conn
            )
            conn.close()
            result = {}
            for _, row in df_all.iterrows():
                result[str(row["sector"])] = {
                    "count":    int(row["cnt"]),
                    "hit_rate": round(float(row["hit"])*100, 1) if row["hit"] else 0,
                    "active":   int(row["cnt"]) >= 30,
                }
            return result
        except:
            return {}