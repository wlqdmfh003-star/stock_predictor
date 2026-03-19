import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class StockCluster:
    """
    종목 클러스터링 v6.0
    ★ K-Means / 계층적 클러스터링으로 유사 종목 군집화
    ★ 클러스터별 대표 종목 선정 (포트폴리오 진짜 분산)
    ★ 상관관계 필터보다 정교한 방법
    ★ 클러스터 강도 점수 (같은 클러스터 종목들이 오르면 가산점)
    ★ sklearn 없어도 동작 (자체 구현 폴백)
    """

    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self.labels_    = None
        self.centers_   = None

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        종목 특성 벡터로 클러스터링
        반환: df에 cluster_id, cluster_rank, is_cluster_rep 컬럼 추가
        """
        df = df.copy()
        if len(df) < self.n_clusters:
            df["cluster_id"]   = 0
            df["cluster_rank"] = df["rise_prob"].rank(ascending=False).astype(int)
            df["is_cluster_rep"] = True
            return df

        # ── 피처 벡터 구성 ────────────────────────────────────────────────
        feat_cols = [
            "rise_prob", "momentum_score", "fundamental_score",
            "sentiment_score", "institution_score", "volume_score",
            "ensemble_score", "lstm_score",
        ]
        avail = [c for c in feat_cols if c in df.columns]
        if not avail:
            df["cluster_id"]     = 0
            df["cluster_rank"]   = range(1, len(df)+1)
            df["is_cluster_rep"] = True
            return df

        X = df[avail].fillna(50).values.astype(float)

        # 정규화 (0~1)
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        rng  = maxs - mins + 1e-9
        X_norm = (X - mins) / rng

        # ── K-Means 클러스터링 ────────────────────────────────────────────
        labels = self._kmeans(X_norm, self.n_clusters)
        df["cluster_id"] = labels

        # ── 클러스터 내 순위 ──────────────────────────────────────────────
        df["cluster_rank"] = df.groupby("cluster_id")["rise_prob"]\
                               .rank(ascending=False).astype(int)

        # ── 클러스터 대표 종목 (각 클러스터 1위) ─────────────────────────
        df["is_cluster_rep"] = df["cluster_rank"] == 1

        # ── 클러스터 강도 점수 (같은 클러스터 평균 상승확률) ─────────────
        cluster_avg = df.groupby("cluster_id")["rise_prob"].mean()
        df["cluster_strength"] = df["cluster_id"].map(cluster_avg)

        print(f"  [클러스터링] {self.n_clusters}개 클러스터 완료 "
              f"(대표종목 {df['is_cluster_rep'].sum()}개)")
        return df

    def get_diversified_top(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        클러스터별 대표 종목으로 진짜 분산된 TOP N 반환
        각 클러스터에서 최소 1개씩 선정
        """
        if "cluster_id" not in df.columns:
            return df.head(n)

        selected = []
        # 클러스터별 1위 종목 우선 선정
        reps = df[df["is_cluster_rep"]].sort_values("rise_prob", ascending=False)
        selected.extend(reps.index.tolist())

        # 부족하면 클러스터 2위 종목 추가
        if len(selected) < n:
            non_reps = df[~df.index.isin(selected)]\
                         .sort_values("rise_prob", ascending=False)
            selected.extend(non_reps.index.tolist()[:n-len(selected)])

        return df.loc[selected[:n]]

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
        """자체 구현 K-Means (sklearn 없어도 동작)"""
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=min(k, len(X)), random_state=42,
                        n_init=10, max_iter=max_iter)
            return km.fit_predict(X)
        except ImportError:
            return self._simple_kmeans(X, k, max_iter)

    def _simple_kmeans(self, X: np.ndarray, k: int,
                       max_iter: int = 100) -> np.ndarray:
        """sklearn 없을 때 자체 K-Means"""
        n = len(X)
        k = min(k, n)
        # 초기 중심: 균등 간격
        idx = np.linspace(0, n-1, k, dtype=int)
        centers = X[idx].copy()
        labels  = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # 할당
            new_labels = np.array([
                np.argmin([np.sum((x - c)**2) for c in centers])
                for x in X
            ])
            if np.all(new_labels == labels):
                break
            labels = new_labels
            # 중심 업데이트
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    centers[j] = X[mask].mean(axis=0)

        return labels

    def cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """클러스터별 요약 통계"""
        if "cluster_id" not in df.columns:
            return pd.DataFrame()
        summary = df.groupby("cluster_id").agg(
            종목수       = ("name",        "count"),
            평균상승확률  = ("rise_prob",    "mean"),
            대표종목     = ("name",         "first"),
            평균PER      = ("per",          "mean") if "per" in df.columns else ("rise_prob", "count"),
        ).round(1).reset_index()
        summary.columns = ["클러스터", "종목수", "평균상승확률(%)", "대표종목", "평균PER"]
        return summary.sort_values("평균상승확률(%)", ascending=False)