import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    상관관계 기반 포트폴리오 분산 최적화
    ★ 종목간 수익률 상관관계 계산 → 높으면 교체
    ★ 섹터당 최대 2종목 제한
    ★ 켈리공식 기반 비중 + 최대 30% 제한
    ★ 분산도 점수 리포트
    """

    MAX_WEIGHT     = 0.30
    MAX_CORR       = 0.75
    MAX_PER_SECTOR = 2
    MIN_PORTFOLIO  = 5

    def optimize(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return df

        candidates  = df.head(min(top_n * 2, len(df))).copy()
        corr_matrix = self._calc_correlation(candidates)
        portfolio   = self._select_portfolio(candidates, corr_matrix, top_n)
        portfolio   = self._calc_weights(portfolio)
        portfolio   = self._add_corr_score(portfolio, corr_matrix)
        return portfolio.reset_index(drop=True)

    def _calc_correlation(self, df):
        returns_dict = {}
        ohlcv_list   = df["ohlcv"].tolist() if "ohlcv" in df.columns else []
        codes        = df["code"].tolist()   if "code"  in df.columns else []

        for code, ohlcv in zip(codes, ohlcv_list):
            if ohlcv is None or not isinstance(ohlcv, pd.DataFrame) or len(ohlcv) < 20:
                continue
            try:
                close = ohlcv["close"].astype(float)
                ret   = close.pct_change().dropna().tail(60)
                returns_dict[code] = ret.values
            except Exception:
                continue

        if len(returns_dict) < 2:
            n          = len(df)
            codes_list = df["code"].tolist() if "code" in df.columns else [str(i) for i in range(n)]
            return pd.DataFrame(np.eye(n), index=codes_list[:n], columns=codes_list[:n])

        min_len = min(len(v) for v in returns_dict.values())
        aligned = {k: v[-min_len:] for k, v in returns_dict.items()}
        return pd.DataFrame(aligned).corr()

    def _select_portfolio(self, candidates, corr_matrix, top_n):
        selected_codes   = []
        selected_indices = []
        sector_count     = {}

        for idx, row in candidates.iterrows():
            code   = str(row.get("code", ""))
            sector = str(row.get("sector", "기타"))

            if sector_count.get(sector, 0) >= self.MAX_PER_SECTOR:
                continue

            too_corr = False
            if code in corr_matrix.index:
                for sel in selected_codes:
                    if sel in corr_matrix.columns:
                        if abs(float(corr_matrix.loc[code, sel])) > self.MAX_CORR:
                            too_corr = True
                            break
            if too_corr:
                continue

            selected_codes.append(code)
            selected_indices.append(idx)
            sector_count[sector] = sector_count.get(sector, 0) + 1

            if len(selected_codes) >= top_n:
                break

        # 최소 종목 수 보장
        if len(selected_indices) < self.MIN_PORTFOLIO:
            for idx, row in candidates.iterrows():
                if idx not in selected_indices:
                    selected_indices.append(idx)
                if len(selected_indices) >= self.MIN_PORTFOLIO:
                    break

        return candidates.loc[selected_indices].copy()

    def _calc_weights(self, df):
        if len(df) == 0:
            return df
        n = len(df)
        if "half_kelly" in df.columns:
            kelly  = df["half_kelly"].fillna(5.0).clip(0, 25)
            total  = kelly.sum()
            if total > 0:
                w = (kelly / total).clip(0, self.MAX_WEIGHT)
                w = w / w.sum()
            else:
                w = pd.Series([1.0/n]*n, index=df.index)
        else:
            w = pd.Series([1.0/n]*n, index=df.index)

        df["portfolio_weight"] = (w * 100).round(1)
        return df

    def _add_corr_score(self, df, corr_matrix):
        codes     = df["code"].tolist() if "code" in df.columns else []
        avg_corrs = []
        for code in codes:
            if code not in corr_matrix.index:
                avg_corrs.append(0.0)
                continue
            others = [c for c in codes if c != code and c in corr_matrix.columns]
            if not others:
                avg_corrs.append(0.0)
                continue
            avg_corrs.append(round(float(np.mean([abs(float(corr_matrix.loc[code,c])) for c in others])), 3))
        df["avg_correlation"] = avg_corrs
        return df

    def get_portfolio_report(self, df):
        if df is None or len(df) == 0:
            return {}
        sector_dist    = df["sector"].value_counts().to_dict() if "sector" in df.columns else {}
        avg_corr       = float(df["avg_correlation"].mean()) if "avg_correlation" in df.columns else 0.0
        diversification= float(np.clip(100 - avg_corr*100 + len(sector_dist)*5, 0, 100))
        return {
            "종목수":       len(df),
            "섹터수":       len(sector_dist),
            "섹터분포":     sector_dist,
            "평균상관계수": round(avg_corr, 3),
            "분산도점수":   round(diversification, 1),
        }