import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class OptionStrategy:
    """
    옵션 전략 분석 v7.0 — 끝판왕
    ★ Black-Scholes 근사 옵션 가격 계산
    ★ 내재변동성 (Parkinson + Yang-Zhang)
    ★ 그리스 지표 (Delta / Gamma / Theta / Vega)
    ★ 6가지 핵심 전략 자동 추천
       1. 롱콜    — 강한 상승 예상
       2. 롱풋    — 강한 하락 예상
       3. 커버드콜 — 보유 종목 수익 극대화
       4. 프로텍티브풋 — 보유 종목 하락 헤지
       5. 스트래들  — 고변동성 방향 모름
       6. 불스프레드 — 완만한 상승 예상
    ★ 손익분기점 / 최대손실 / 최대수익 자동 계산
    ★ 옵션 점수 → 최종 종목 점수 반영
    """

    # 한국 무위험 이자율 (연 %)
    RISK_FREE_RATE = 0.035

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 종목에 옵션 전략 분석 적용"""
        df = df.copy()
        strategies, scores, greeks_list, iv_list = [], [], [], []

        for _, row in df.iterrows():
            ohlcv = row.get("ohlcv")
            price = float(row.get("current_price", 0) or 0)
            prob  = float(row.get("rise_prob", 50) or 50)
            phase = str(row.get("market_phase", "중립"))

            if ohlcv is None or len(ohlcv) < 20 or price <= 0:
                strategies.append(_empty_strategy())
                scores.append(50.0)
                greeks_list.append(_empty_greeks())
                iv_list.append(20.0)
                continue

            # 내재변동성 계산
            iv = self._calc_iv(ohlcv)

            # 그리스 지표
            greeks = self._calc_greeks(price, iv)

            # 전략 추천
            strategy = self._recommend_strategy(
                price, iv, prob, phase, greeks, row
            )

            # 옵션 점수 (0~100)
            opt_score = self._calc_option_score(strategy, prob, iv)

            strategies.append(strategy)
            scores.append(opt_score)
            greeks_list.append(greeks)
            iv_list.append(iv)

        df["option_strategy"]  = [s["name"]        for s in strategies]
        df["option_detail"]    = [s["detail"]       for s in strategies]
        df["option_breakeven"] = [s["breakeven"]    for s in strategies]
        df["option_max_profit"]= [s["max_profit"]   for s in strategies]
        df["option_max_loss"]  = [s["max_loss"]     for s in strategies]
        df["option_score"]     = scores
        df["implied_vol"]      = iv_list
        df["delta"]            = [g["delta"]  for g in greeks_list]
        df["gamma"]            = [g["gamma"]  for g in greeks_list]
        df["theta"]            = [g["theta"]  for g in greeks_list]
        df["vega"]             = [g["vega"]   for g in greeks_list]

        print(f"  [옵션전략] {len(df)}개 종목 분석완료 "
              f"(평균IV={np.mean(iv_list):.1f}%)")
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # 내재변동성 계산 (Parkinson + Yang-Zhang 혼합)
    # ══════════════════════════════════════════════════════════════════════════
    def _calc_iv(self, ohlcv, n: int = 20) -> float:
        """
        Parkinson(60%) + Yang-Zhang(40%) 혼합 변동성
        → 실제 옵션 IV와 가장 근접한 근사값
        """
        try:
            c = ohlcv["close"].astype(float).values
            h = ohlcv["high"].astype(float).values  if "high" in ohlcv.columns else c.copy()
            l = ohlcv["low"].astype(float).values   if "low"  in ohlcv.columns else c.copy()
            o = ohlcv["open"].astype(float).values  if "open" in ohlcv.columns else c.copy()

            n = min(n, len(c)-1)
            if n < 5:
                return 20.0

            # Parkinson 변동성
            log_hl  = np.log(h[-n:] / (l[-n:] + 1e-9))
            park    = float(np.sqrt((1/(4*n*np.log(2))) * np.sum(log_hl**2) * 252) * 100)

            # Yang-Zhang 변동성
            log_co  = np.log(c[-n:] / (o[-n:] + 1e-9))
            log_oc  = np.log(o[-n:] / (c[-n-1:-1] + 1e-9)) if len(c) > n else log_co
            yz_var  = float(np.var(log_co) + 0.5*np.var(log_hl) - (2*np.log(2)-1)*np.var(log_oc))
            yz      = float(np.sqrt(max(yz_var, 0) * 252) * 100)

            # 역사적 변동성
            rets   = np.diff(np.log(c[-n-1:] + 1e-9))
            hist   = float(np.std(rets) * np.sqrt(252) * 100)

            # 혼합
            iv = park*0.4 + yz*0.3 + hist*0.3
            return float(np.clip(iv, 5, 200))
        except:
            return 20.0

    # ══════════════════════════════════════════════════════════════════════════
    # Black-Scholes 옵션 가격 계산
    # ══════════════════════════════════════════════════════════════════════════
    def _bs_price(self, S, K, T, r, sigma, option_type="call") -> float:
        """
        Black-Scholes 옵션 가격
        S: 현재가, K: 행사가, T: 만기(년), r: 무위험이자율, sigma: 변동성
        """
        try:
            from math import log, sqrt, exp
            from statistics import NormalDist
            nd = NormalDist()

            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return 0.0

            d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
            d2 = d1 - sigma*sqrt(T)

            if option_type == "call":
                price = S*nd.cdf(d1) - K*exp(-r*T)*nd.cdf(d2)
            else:
                price = K*exp(-r*T)*nd.cdf(-d2) - S*nd.cdf(-d1)

            return float(max(0, price))
        except:
            return 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # 그리스 지표 계산 (Delta / Gamma / Theta / Vega)
    # ══════════════════════════════════════════════════════════════════════════
    def _calc_greeks(self, S: float, iv: float,
                     T: float = 1/12) -> dict:
        """
        ATM 콜옵션 기준 그리스 지표
        T: 만기 (기본 1개월)
        """
        try:
            from math import log, sqrt, exp
            from statistics import NormalDist
            nd  = NormalDist()
            r   = self.RISK_FREE_RATE
            K   = S  # ATM (등가격)
            sig = iv / 100

            if T <= 0 or sig <= 0:
                return _empty_greeks()

            d1 = (log(S/K) + (r + 0.5*sig**2)*T) / (sig*sqrt(T))
            d2 = d1 - sig*sqrt(T)

            delta = float(nd.cdf(d1))
            gamma = float(nd.pdf(d1) / (S*sig*sqrt(T)))
            theta = float(-(S*nd.pdf(d1)*sig/(2*sqrt(T)) +
                           r*K*exp(-r*T)*nd.cdf(d2)) / 365)
            vega  = float(S*nd.pdf(d1)*sqrt(T) / 100)

            return {
                "delta": round(delta, 4),
                "gamma": round(gamma, 6),
                "theta": round(theta, 4),
                "vega":  round(vega,  4),
            }
        except:
            return _empty_greeks()

    # ══════════════════════════════════════════════════════════════════════════
    # 전략 추천 엔진
    # ══════════════════════════════════════════════════════════════════════════
    def _recommend_strategy(self, price: float, iv: float,
                            prob: float, phase: str,
                            greeks: dict, row) -> dict:
        """
        상황에 맞는 최적 옵션 전략 자동 추천
        """
        r   = self.RISK_FREE_RATE
        T   = 1/12  # 1개월 만기
        sig = iv / 100

        # 콜/풋 ATM 가격
        call_price = self._bs_price(price, price, T, r, sig, "call")
        put_price  = self._bs_price(price, price, T, r, sig, "put")

        # ── 전략 선택 로직 ────────────────────────────────────────────────────
        # 극단적 상승 예상 (prob >= 70, 강세장)
        if prob >= 70 and "탐욕" in phase:
            return self._long_call(price, call_price, iv)

        # 극단적 하락 예상 (prob <= 35, 공포장)
        elif prob <= 35 and "공포" in phase:
            return self._long_put(price, put_price, iv)

        # 고변동성 + 방향 모름 (IV > 40%)
        elif iv >= 40 and 40 <= prob <= 60:
            return self._straddle(price, call_price, put_price, iv)

        # 완만한 상승 예상 (prob 55~70)
        elif 55 <= prob < 70:
            return self._bull_spread(price, call_price, iv, r, T, sig)

        # 보유 종목 수익 극대화 (횡보+상승 예상)
        elif 50 <= prob < 65 and iv < 30:
            return self._covered_call(price, call_price, iv)

        # 하락 헤지 (prob < 50)
        elif prob < 50:
            return self._protective_put(price, put_price, iv)

        # 기본: 커버드콜
        else:
            return self._covered_call(price, call_price, iv)

    # ══════════════════════════════════════════════════════════════════════════
    # 6가지 전략 구현
    # ══════════════════════════════════════════════════════════════════════════
    def _long_call(self, S, call_price, iv) -> dict:
        """롱콜 — 강한 상승 예상 시"""
        premium      = call_price
        breakeven    = S + premium
        max_profit   = "무제한"
        max_loss     = -premium
        profit_1m    = max(0, S*1.05 - S - premium)  # 5% 상승 시 수익

        return {
            "name":       "📈 롱콜 (Long Call)",
            "detail":     f"행사가 {S:,.0f}원 콜 매수 | 프리미엄 {premium:,.0f}원",
            "breakeven":  f"{breakeven:,.0f}원 ({(breakeven/S-1)*100:+.1f}%)",
            "max_profit": max_profit,
            "max_loss":   f"{max_loss:,.0f}원 (프리미엄)",
            "scenario":   f"5% 상승 시 예상수익: {profit_1m:,.0f}원",
            "iv":         iv,
            "type":       "강세",
            "risk":       "낮음 (프리미엄만 손실)",
        }

    def _long_put(self, S, put_price, iv) -> dict:
        """롱풋 — 강한 하락 예상 시"""
        premium   = put_price
        breakeven = S - premium
        max_profit= S - premium  # 주가 0이면 최대
        max_loss  = -premium

        return {
            "name":       "📉 롱풋 (Long Put)",
            "detail":     f"행사가 {S:,.0f}원 풋 매수 | 프리미엄 {premium:,.0f}원",
            "breakeven":  f"{breakeven:,.0f}원 ({(breakeven/S-1)*100:+.1f}%)",
            "max_profit": f"{max_profit:,.0f}원",
            "max_loss":   f"{max_loss:,.0f}원 (프리미엄)",
            "scenario":   f"5% 하락 시 예상수익: {max(0,S*0.05-premium):,.0f}원",
            "iv":         iv,
            "type":       "약세",
            "risk":       "낮음 (프리미엄만 손실)",
        }

    def _straddle(self, S, call_price, put_price, iv) -> dict:
        """스트래들 — 고변동성, 방향 모를 때"""
        total_premium = call_price + put_price
        be_up   = S + total_premium
        be_down = S - total_premium

        return {
            "name":       "⚡ 스트래들 (Straddle)",
            "detail":     f"콜+풋 동시 매수 | 총 프리미엄 {total_premium:,.0f}원 | IV={iv:.1f}%",
            "breakeven":  f"상단 {be_up:,.0f}원 / 하단 {be_down:,.0f}원",
            "max_profit": "무제한 (양방향)",
            "max_loss":   f"{-total_premium:,.0f}원 (횡보 시)",
            "scenario":   f"±{total_premium/S*100:.1f}% 이상 움직이면 수익",
            "iv":         iv,
            "type":       "변동성",
            "risk":       "중간 (프리미엄×2)",
        }

    def _bull_spread(self, S, call_price, iv, r, T, sig) -> dict:
        """불스프레드 — 완만한 상승 예상"""
        K1   = S
        K2   = S * 1.05  # 5% 위 행사가
        buy  = call_price
        sell = self._bs_price(S, K2, T, r, sig, "call")
        net  = buy - sell
        max_profit = K2 - K1 - net
        breakeven  = K1 + net

        return {
            "name":       "🐂 불스프레드 (Bull Spread)",
            "detail":     f"콜 매수({K1:,.0f}) + 콜 매도({K2:,.0f}) | 순비용 {net:,.0f}원",
            "breakeven":  f"{breakeven:,.0f}원 ({(breakeven/S-1)*100:+.1f}%)",
            "max_profit": f"{max_profit:,.0f}원 (5% 이상 상승 시)",
            "max_loss":   f"{-net:,.0f}원",
            "scenario":   f"3% 상승 시 예상수익: {min(S*0.03,max_profit):,.0f}원",
            "iv":         iv,
            "type":       "완만한 강세",
            "risk":       "낮음",
        }

    def _covered_call(self, S, call_price, iv) -> dict:
        """커버드콜 — 보유 종목 추가 수익"""
        premium   = call_price
        K         = S * 1.03  # 3% 위 행사가 (OTM)
        breakeven = S - premium
        max_profit= premium + (K - S)

        return {
            "name":       "💰 커버드콜 (Covered Call)",
            "detail":     f"주식 보유 + 콜 매도({K:,.0f}원) | 프리미엄 수취 {premium:,.0f}원",
            "breakeven":  f"{breakeven:,.0f}원 ({(breakeven/S-1)*100:+.1f}%)",
            "max_profit": f"{max_profit:,.0f}원 (3% 상승까지)",
            "max_loss":   f"주가 하락분 - {premium:,.0f}원",
            "scenario":   f"월 {premium/S*100:.1f}% 추가 수익 (횡보 시)",
            "iv":         iv,
            "type":       "수익 극대화",
            "risk":       "낮음 (주식 보유 기준)",
        }

    def _protective_put(self, S, put_price, iv) -> dict:
        """프로텍티브풋 — 하락 헤지"""
        premium   = put_price
        K         = S * 0.97  # 3% 아래 행사가
        breakeven = S + premium

        return {
            "name":       "🛡️ 프로텍티브풋 (Protective Put)",
            "detail":     f"주식 보유 + 풋 매수({K:,.0f}원) | 보험료 {premium:,.0f}원",
            "breakeven":  f"{breakeven:,.0f}원 ({(breakeven/S-1)*100:+.1f}%)",
            "max_profit": "무제한 (상승 시)",
            "max_loss":   f"{-(S-K+premium):,.0f}원 (3% 하락까지 보호)",
            "scenario":   f"10% 하락해도 손실 {(S*0.03+premium)/S*100:.1f}%로 제한",
            "iv":         iv,
            "type":       "헤지",
            "risk":       "매우 낮음",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 옵션 점수 계산
    # ══════════════════════════════════════════════════════════════════════════
    def _calc_option_score(self, strategy: dict,
                           prob: float, iv: float) -> float:
        """옵션 전략 기반 최종 점수 보정"""
        score = 50.0
        stype = strategy.get("type", "")

        # 전략 유형별 기본 점수
        if   stype == "강세"       and prob >= 65: score += 20
        elif stype == "완만한 강세" and prob >= 55: score += 15
        elif stype == "수익 극대화" and prob >= 50: score += 10
        elif stype == "변동성"     and iv >= 40:   score += 8
        elif stype == "헤지":                       score += 5
        elif stype == "약세"       and prob <= 35: score += 15

        # IV 보정
        if   iv > 60:  score -= 5   # 고변동성 리스크
        elif iv < 15:  score += 5   # 저변동성 안정
        elif 20<=iv<=35: score += 3  # 적정 변동성

        return float(np.clip(score, 0, 100))

    # ══════════════════════════════════════════════════════════════════════════
    # 전체 시장 옵션 시그널 요약
    # ══════════════════════════════════════════════════════════════════════════
    def market_option_signal(self, df: pd.DataFrame) -> dict:
        """시장 전체 옵션 시그널 분석"""
        if "implied_vol" not in df.columns:
            return {}
        avg_iv  = float(df["implied_vol"].mean())
        avg_delta = float(df["delta"].mean()) if "delta" in df.columns else 0.5
        high_iv = df[df["implied_vol"] > 40]
        low_iv  = df[df["implied_vol"] < 15]

        if avg_iv > 50:
            market_signal = "🚨 시장 공포 — 스트래들/풋 전략 유리"
        elif avg_iv > 35:
            market_signal = "⚡ 변동성 높음 — 스트래들 유효"
        elif avg_iv < 15:
            market_signal = "😴 변동성 낮음 — 커버드콜 유리"
        else:
            market_signal = "✅ 정상 변동성 — 방향성 전략 유효"

        return {
            "avg_iv":        round(avg_iv, 1),
            "avg_delta":     round(avg_delta, 3),
            "high_iv_count": len(high_iv),
            "low_iv_count":  len(low_iv),
            "market_signal": market_signal,
            "best_strategy": "스트래들" if avg_iv > 35 else "커버드콜" if avg_iv < 20 else "불스프레드",
        }


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def _empty_strategy() -> dict:
    return {
        "name": "-", "detail": "-", "breakeven": "-",
        "max_profit": "-", "max_loss": "-",
        "scenario": "-", "iv": 20.0, "type": "-", "risk": "-",
    }

def _empty_greeks() -> dict:
    return {"delta": 0.5, "gamma": 0.0, "theta": 0.0, "vega": 0.0}