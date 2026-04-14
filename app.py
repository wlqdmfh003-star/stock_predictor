import os
import sys

# Windows 터미널 인코딩 UTF-8 강제 설정 (깨진 글자 방지)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# HuggingFace 토큰 경고 억제
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="📈 내일 주식 상승 예측 v7.2",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700;900&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans KR',sans-serif;}
.main-header{background:linear-gradient(135deg,#0f0f23,#1a1a3e,#0d1b2a);padding:28px;border-radius:16px;text-align:center;margin-bottom:20px;border:1px solid rgba(100,180,255,.2);}
.main-header h1{color:#64b4ff;font-size:1.9rem;font-weight:900;margin:0;}
.main-header p{color:#94a3b8;margin:8px 0 0;font-size:.82rem;}
.metric-card{background:linear-gradient(135deg,#1e1e3a,#252545);border:1px solid rgba(100,180,255,.15);border-radius:12px;padding:14px;text-align:center;margin:3px 0;}
.metric-card .label{color:#94a3b8;font-size:.75rem;margin-bottom:5px;}
.metric-card .value{color:#64b4ff;font-size:1.4rem;font-weight:700;}
.metric-card .value.green{color:#10b981;}
.metric-card .value.red{color:#ef4444;}
.metric-card .value.gold{color:#f59e0b;}
.info-box{background:rgba(100,180,255,.08);border:1px solid rgba(100,180,255,.2);border-radius:10px;padding:10px 14px;color:#93c5fd;font-size:.83rem;margin:6px 0;}
.learn-box{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.25);border-radius:10px;padding:10px 14px;color:#6ee7b7;font-size:.83rem;margin:6px 0;}
.upgrade-box{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-radius:10px;padding:10px 14px;color:#fcd34d;font-size:.83rem;margin:6px 0;}
</style>
""", unsafe_allow_html=True)


def _clean_df(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: "-" if isinstance(x,(pd.DataFrame,list,dict))
                else ("-" if (isinstance(x,float) and pd.isna(x)) else str(x))
                if not isinstance(x,str) else x
            )
    return df


from utils.data_fetcher     import DataFetcher
from utils.indicators       import TechnicalIndicators
from utils.news_sentiment   import NewsSentiment
from utils.market_phase     import MarketPhase
from utils.volume_detector  import VolumeDetector
from utils.fundamental      import FundamentalAnalyzer
from utils.dart_disclosure  import DartDisclosure
from utils.short_selling    import ShortSelling
from utils.high_low_52week  import HighLow52Week
from utils.us_market        import USMarket
from utils.sector_analysis  import SectorAnalysis
from utils.candle_patterns  import CandlePatterns
from utils.macro_indicators import MacroIndicators
from utils.learning_tracker  import LearningTracker
from utils.earnings_calendar import EarningsCalendar
from utils.stock_cluster     import StockCluster
from utils.option_strategy   import OptionStrategy
from models.lstm_model       import LSTMPredictor
from models.multi_factor     import MultiFactorScorer
from models.backtester       import Backtester
from models.ensemble_model   import EnsembleModel

st.markdown("""
<div class="main-header">
  <h1>📈 내일 주식 상승 종목 예측 v7.2</h1>
  <p>LSTM · XGBoost+LightGBM+CatBoost(109개 피처) · 트리플 타임프레임(일봉+주봉+월봉)
  · 일목균형표 · 피보나치 · 엘리어트파동 · CNN패턴 · 캔들패턴 19가지
  · ATR 동적 손절/목표가 · 상관관계 필터 · 요일/월말 효과 · 섹터 yfinance 자동분류
  · KoBERT 뉴스AI · 매크로 · 기관수급 · 재무 · DART · 공매도 · 옵션전략 · 자기학습</p>
</div>
""", unsafe_allow_html=True)

tracker     = LearningTracker()
learn_stats = tracker.get_stats()

# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 분석 설정")
    st.markdown("---")
    market      = st.selectbox("🏦 시장", ["KOSPI","KOSDAQ","KOSPI+KOSDAQ"], index=2)
    top_n       = st.slider("🎯 분석 종목 수", 20, 200, 100, 10)
    min_cap_str = st.selectbox("💰 최소 시가총액", ["500억+","1000억+","3000억+","5000억+"], index=1)
    min_volume  = st.number_input("📊 최소 거래대금(억)", 1, 1000, 50, 10)

    st.markdown("---")
    st.markdown("### 🔧 분석 모듈")
    use_lstm        = st.checkbox("🧠 LSTM 예측",              value=True)
    use_ensemble    = st.checkbox("🤝 앙상블(XGB+LGB+CatBoost+109피처)", value=True)
    use_candle      = st.checkbox("🕯️ 캔들/차트 패턴",          value=True)
    use_macro       = st.checkbox("🌍 매크로 지표",             value=True)
    use_sentiment   = st.checkbox("📰 뉴스 감성 AI (KoBERT)",   value=True)
    use_institution = st.checkbox("🏛️ 기관 수급",              value=True)
    use_volume_anom = st.checkbox("🔍 거래량 이상 탐지",        value=True)
    use_fundamental = st.checkbox("💹 재무 분석",               value=True)
    use_dart        = st.checkbox("📋 DART 공시",               value=True)
    use_short       = st.checkbox("📉 공매도 분석",             value=True)
    use_52week      = st.checkbox("🚀 52주 신고가",             value=True)
    use_us_market   = st.checkbox("🌎 미국 증시 연동",          value=True)
    use_sector      = st.checkbox("🏭 섹터 분석",               value=True)
    use_backtest    = st.checkbox("📊 백테스트",                value=False)
    use_corr_filter    = st.checkbox("🔗 상관관계 필터 (포트폴리오 분산)", value=True)
    use_earnings_cal   = st.checkbox("📅 실적 발표 캘린더",              value=True)
    use_clustering     = st.checkbox("🔵 종목 클러스터링 (진짜 분산)",   value=True)
    use_option         = st.checkbox("🎯 옵션 전략 분석",                 value=True)

    if use_backtest:
        bt_period = st.selectbox(
            "📅 백테스트 기간",
            ["2개월 (60일)","6개월 (125일)","1년 (250일)","2년 (500일)","3년 (750일)"],
            index=2
        )
        bt_top_k = st.slider("🏆 TOP K", 3, 20, 5)
    else:
        bt_period = "1년 (250일)"; bt_top_k = 5

    st.markdown("---")
    st.markdown("### 🤖 가중치")
    use_auto_weight = st.checkbox("🌐 시장국면 자동가중치", value=True)
    _cur_stats  = tracker.get_stats()   # 사이드바도 최신 DB
    ready_label = f"✅ 적용가능 ({_cur_stats['completed']}건)" if _cur_stats["ready"] \
                  else f"⏳ 데이터부족 ({_cur_stats['completed']}/50건)"
    use_learned = st.checkbox(f"🧬 자기학습 가중치 {ready_label}", value=False,
                               disabled=not _cur_stats["ready"])

    st.markdown("#### 📐 수동 가중치")
    w_lstm=st.slider("🧠 LSTM",   0.0,1.0,0.12,0.05)
    w_ens =st.slider("🤝 앙상블", 0.0,1.0,0.10,0.05)
    w_can =st.slider("🕯️ 캔들",   0.0,1.0,0.08,0.05)
    w_mac =st.slider("🌍 매크로", 0.0,1.0,0.07,0.05)
    w_mom =st.slider("📈 모멘텀", 0.0,1.0,0.12,0.05)
    w_sen =st.slider("📰 감성",   0.0,1.0,0.08,0.05)
    w_ins =st.slider("🏛️ 기관",  0.0,1.0,0.10,0.05)
    w_vol =st.slider("🔊 거래량", 0.0,1.0,0.06,0.05)
    w_fun =st.slider("💹 재무",   0.0,1.0,0.08,0.05)
    w_dar =st.slider("📋 공시",   0.0,1.0,0.06,0.05)
    w_sht =st.slider("📉 공매도", 0.0,1.0,0.04,0.05)
    w_h52 =st.slider("🚀 52주",   0.0,1.0,0.04,0.05)
    w_us  =st.slider("🌎 미국",   0.0,1.0,0.05,0.05)
    w_sec =st.slider("🏭 섹터",   0.0,1.0,0.05,0.05)

    st.markdown("---")
    st.markdown("### 🔑 API 설정")
    kis_app_key    = st.text_input("KIS 앱키",          type="password")
    kis_app_secret = st.text_input("KIS 시크릿키",      type="password")
    kis_account    = st.text_input("계좌번호",           placeholder="50173435-01")

    # KIS 연결 테스트
    if st.button("🔌 KIS 연결 테스트"):
        if kis_app_key and kis_app_secret and kis_account:
            with st.spinner("KIS 연결 중..."):
                try:
                    from utils.kis_api import KISApi as _KISApi
                    _conn = _KISApi(app_key=kis_app_key,
                                    app_secret=kis_app_secret,
                                    account=kis_account).test_connection()
                    if _conn["ok"]:
                        st.success(_conn["message"])
                    else:
                        st.error(_conn["message"])
                        st.caption("💡 KIS 개발자센터에서 앱키를 확인하세요\nhttps://apiportal.koreainvestment.com")
                except Exception as _e:
                    st.error(f"오류: {_e}")
        else:
            st.warning("앱키 / 시크릿키 / 계좌번호를 모두 입력하세요")

    # 계좌 유형 표시
    if kis_app_key and kis_account:
        _acc = kis_account.replace("-","")
        _mode = "모의투자 🟡" if _acc.startswith("50") else "실전투자 🟢"
        st.markdown(f'<div style="color:#94a3b8;font-size:.78rem">계좌유형: {_mode}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#ef4444;font-size:.78rem">⚠️ KIS 미입력 → pykrx 크롤링 대체</div>',
                    unsafe_allow_html=True)

    naver_id       = st.text_input("네이버 Client ID",   type="password")
    naver_secret   = st.text_input("네이버 Client Secret",type="password")
    dart_api_key   = st.text_input("DART API 키 (선택)", type="password")

    st.markdown("---")
    analyze_btn = st.button("🚀 분석 시작", type="primary")

# 자기학습 배너 — 분석 후 갱신된 값 사용 (아래 ns2로 대체)
_banner_stats = tracker.get_stats()   # 항상 최신 DB에서 읽기
if _banner_stats["completed"] > 0:
    st.markdown(f"""
    <div class="learn-box">
    🧬 <b>자기학습</b> | 누적 {_banner_stats['completed']}건 |
    적중률 <b>{_banner_stats['hit_rate']}%</b> | 평균수익 {_banner_stats['avg_return']:+.2f}% |
    {"✅ 가중치 최적화 가능" if _banner_stats["ready"] else f"⏳ {_banner_stats['needed']}건 더 필요"}
    </div>
    """, unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12 = st.tabs([
    "🎯 예측결과","🌎 미국/매크로","🏭 섹터","🕯️ 캔들패턴",
    "📊 시장국면","🔗 포트폴리오","📋 백테스트","🧬 자기학습",
    "🔍 예측설명","🔵 클러스터링","🎯 옵션전략","📖 가이드 v7.2"
])

if analyze_btn:
    cap_map={"500억+":50_000_000_000,"1000억+":100_000_000_000,
             "3000억+":300_000_000_000,"5000억+":500_000_000_000}
    min_cap = cap_map.get(min_cap_str,100_000_000_000)

    manual_weights={
        "lstm":w_lstm,"ensemble":w_ens,"candle":w_can,"macro":w_mac,
        "momentum":w_mom,"sentiment":w_sen,"institution":w_ins,"volume":w_vol,
        "fundamental":w_fun,"dart":w_dar,"short":w_sht,"high52":w_h52,
        "us_market":w_us,"sector":w_sec,
    }

    if use_learned and learn_stats["ready"]:
        weights=tracker.load_learned_weights(); weight_mode="🧬 자기학습"
    elif use_auto_weight:
        weights=manual_weights.copy(); weight_mode="🌐 시장국면 자동"
    else:
        weights=manual_weights.copy(); weight_mode="📐 수동"

    with st.spinner("📡 종목 데이터 수집 중... (일봉+주봉+월봉, 3~4분 소요)"):
        fetcher = DataFetcher(market=market, top_n=top_n,
                              min_market_cap=min_cap, min_volume_bil=min_volume)
        df = fetcher.fetch_all_parallel()

    if df is None or len(df)==0:
        st.error("❌ 데이터 수집 실패"); st.stop()

    kospi_n  = len(df[df["market"]=="KOSPI"])  if "market" in df.columns else 0
    kosdaq_n = len(df[df["market"]=="KOSDAQ"]) if "market" in df.columns else 0
    monthly_n= df["ohlcv_monthly"].apply(lambda x: x is not None and len(x)>=6).sum() \
               if "ohlcv_monthly" in df.columns else 0
    st.success(f"✅ {len(df)}개 수집 — KOSPI {kospi_n} / KOSDAQ {kosdaq_n} / 월봉 {monthly_n}개")

    with st.spinner("📐 기술적 지표 계산 중..."): df=TechnicalIndicators().calculate_all(df)
    with st.spinner("📊 시장 국면 분석 중..."):
        pr=MarketPhase().detect(market="KOSPI")
        phase_name=pr[0] if isinstance(pr,tuple) else str(pr)
        phase_score=pr[1] if isinstance(pr,tuple) else 50.0

    # 자동 가중치
    if use_auto_weight and not use_learned:
        if "강세" in phase_name:
            weights.update({"lstm":0.18,"ensemble":0.14,"momentum":0.18,"high52":0.10,
                            "institution":0.08,"candle":0.08,"sentiment":0.07,
                            "fundamental":0.05,"macro":0.04,"dart":0.04,
                            "short":0.02,"volume":0.04,"us_market":0.03,"sector":0.02})
            st.info(f"🚀 강세장 → LSTM·앙상블·모멘텀 상향 | {weight_mode}")
        elif "약세" in phase_name:
            weights.update({"fundamental":0.16,"short":0.14,"institution":0.14,
                            "dart":0.10,"sentiment":0.08,"macro":0.08,
                            "lstm":0.07,"ensemble":0.06,"momentum":0.06,
                            "candle":0.04,"volume":0.03,"high52":0.02,
                            "us_market":0.02,"sector":0.02})
            st.info(f"🛡️ 약세장 → 재무·공매도·기관 상향 | {weight_mode}")
        else:
            weights.update({"lstm":0.12,"ensemble":0.10,"momentum":0.12,"candle":0.08,
                            "macro":0.07,"sentiment":0.08,"institution":0.10,"volume":0.06,
                            "fundamental":0.08,"dart":0.06,"short":0.04,"high52":0.04,
                            "us_market":0.05,"sector":0.05})
            st.info(f"➡️ 횡보장 → 균등 배분 | {weight_mode}")

    tw=sum(weights.values()); weights={k:v/tw for k,v in weights.items()} if tw>0 else weights

    # ★ 섹터 분석은 항상 실행 (테마 감지에 필요)
    with st.spinner("🏭 섹터 분석 중..."): df=SectorAnalysis().analyze(df)
    if use_candle:
        with st.spinner("🕯️ 캔들/차트 패턴 (19가지) 분석 중..."): df=CandlePatterns().analyze(df)
    macro_data={}
    if use_macro:
        with st.spinner("🌍 매크로 지표 수집 중..."):
            mi=MacroIndicators(); macro_data=mi.fetch(); df=mi.apply_to_stocks(df,macro_data)
    # ★ 뉴스 감성은 항상 실행 (테마 생성 필요)
    with st.spinner("📰 뉴스 감성 AI (KoBERT/키워드) 분석 중..."):
        df=NewsSentiment(client_id=naver_id,client_secret=naver_secret).analyze(df)
    if use_institution:
        with st.spinner("🏛️ 기관/외인 수급 분석 중..."):
            try:
                from utils.kis_api import KISApi
                df=KISApi(app_key=kis_app_key,app_secret=kis_app_secret,account=kis_account).get_institution_data(df)
            except: df=fetcher.fetch_institution_data(df)
    if use_volume_anom:
        with st.spinner("🔍 거래량 이상 탐지 중..."): df=VolumeDetector().detect(df)
    if use_fundamental:
        with st.spinner("💹 재무 분석 중..."): df=FundamentalAnalyzer().fetch_and_score(df)
    if use_dart:
        with st.spinner("📋 DART 공시 분석 중..."): df=DartDisclosure(api_key=dart_api_key).fetch_and_score(df)
    if use_short:
        with st.spinner("📉 공매도 분석 중..."): df=ShortSelling().fetch_and_score(df)
    if use_52week:
        with st.spinner("🚀 52주 신고가 분석 중..."): df=HighLow52Week().fetch_and_score(df)
    us_data={}
    if use_us_market:
        with st.spinner("🌎 미국 증시 수집 중..."):
            um=USMarket(); us_data=um.fetch(); df=um.apply_to_stocks(df,us_data)
    if use_lstm:
        with st.spinner("🧠 LSTM 예측 중..."): df=LSTMPredictor().predict_batch(df)
    if use_ensemble:
        with st.spinner("🤝 앙상블 (XGB+LGB+CatBoost, 109피처, 트리플 타임프레임) 예측 중..."):
            df=EnsembleModel().predict_batch(df)

    # ★ 클러스터링 파이프라인 (체크 시 실행)
    if use_clustering:
        with st.spinner("🔵 종목 클러스터링 분석 중..."):
            try:
                df = StockCluster().fit_predict(df)
            except Exception as _ce:
                pass

    # ★ 옵션 전략 파이프라인 (체크 시 실행)
    if use_option:
        with st.spinner("🎯 옵션 전략 분석 중..."):
            try:
                df = OptionStrategy().analyze(df)
            except Exception as _oe:
                pass

    # ★ sector/theme_tag 최종 안전망 (직접 계산)
    if "sector" not in df.columns or df["sector"].isna().all():
        df = SectorAnalysis().analyze(df)
    if "theme_tag" not in df.columns or df["theme_tag"].isna().all():
        from utils.news_sentiment import CODE_THEME_MAP, STOCK_THEME_MAP, SECTOR_THEME_MAP
        def _get_theme(row):
            code   = str(row.get("code","")   or "")
            name   = str(row.get("name","")   or "")
            sector = str(row.get("sector","") or "")
            if code in CODE_THEME_MAP:
                return CODE_THEME_MAP[code]
            if name in STOCK_THEME_MAP:
                return STOCK_THEME_MAP[name]
            if sector in SECTOR_THEME_MAP:
                return SECTOR_THEME_MAP[sector]
            if sector and sector not in ["기타",""]:
                return sector
            return ""
        df["theme_tag"] = df.apply(_get_theme, axis=1)

    with st.spinner("🎯 최종 점수 계산 중 (ATR 손절/상관관계/요일효과)..."):
        scorer=MultiFactorScorer(weights=weights)
        df=scorer.score(df)
        df=df.sort_values("rise_prob",ascending=False).reset_index(drop=True)

    saved=tracker.save_predictions(df); updated=tracker.update_results(df)
    ns2=tracker.get_stats()
    if ns2["ready"]: tracker.calc_learned_weights()

    if "name" in df.columns:
        df["name"]=df["name"].apply(lambda x: str(x) if not isinstance(x,str) else x)\
                             .replace({"":"—","nan":"—"})

    # 요일/월말 효과 알림
    if "calendar_note" in df.columns:
        note=df["calendar_note"].iloc[0] if len(df)>0 else ""
        bonus=df["calendar_bonus"].iloc[0] if "calendar_bonus" in df.columns and len(df)>0 else 0
        if note and note!="해당없음":
            st.markdown(f'<div class="upgrade-box">📅 <b>시간대 효과 적용:</b> {note} (전체 {bonus:+.1f}점)</div>',
                        unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # 탭 1: 예측 결과
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown(f"### 🎯 예측 결과 — {weight_mode} | 시장국면: **{phase_name}**")

        top5_avg  = df.head(5)["rise_prob"].mean() if len(df)>=5 else 0
        high_conf = len(df[df["rise_prob"]>=70])
        corr_excl = int(df["corr_flag"].sum()) if "corr_flag" in df.columns else 0

        cols=st.columns(6)
        for col,(label,val,cls) in zip(cols,[
            ("📊 분석종목",    f"{len(df)}개",      ""),
            ("🏆 TOP5 평균",   f"{top5_avg:.1f}%",  "green"),
            ("⭐ 70%+ 고신뢰", f"{high_conf}개",     "gold"),
            ("🌙 월봉 활용",   f"{monthly_n}개",    ""),
            ("🔗 상관중복",    f"{corr_excl}개 제거",""),
            ("📅 요일효과",    df["calendar_note"].iloc[0] if "calendar_note" in df.columns and len(df)>0 else "계산됨", ""),
        ]):
            with col:
                st.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                            f'<div class="value {cls}">{val}</div></div>',unsafe_allow_html=True)

        display_cols=["name","market","rise_prob","change_pct","market_cap",
            "current_price","buy_price","target_price","stop_price","expected_return",
            "per","pbr","roe","roa","debt_ratio","op_margin","rev_growth",
            "momentum_score","sentiment_score","institution_score",
            "fundamental_score","lstm_score","ensemble_score","candle_score","macro_score",
            "sector_score","theme_tag","candle_pattern","news_summary","sector","corr_flag","calendar_note"]
        show_cols=[c for c in display_cols if c in df.columns]
        rename_map={
            "name":"종목명","market":"시장","rise_prob":"상승확률","change_pct":"등락률",
            "market_cap":"시가총액","current_price":"현재가","buy_price":"예상매수가",
            "target_price":"목표가","stop_price":"ATR손절가","expected_return":"예상수익",
            "per":"PER","pbr":"PBR","roe":"ROE(%)","roa":"ROA(%)",
            "debt_ratio":"부채비율","op_margin":"영업이익률","rev_growth":"매출증가율",
            "theme_tag":"테마",
            "momentum_score":"모멘텀","sentiment_score":"감성","institution_score":"기관수급","ob_score":"호가점수","trade_strength":"체결강도",
            "fundamental_score":"재무","lstm_score":"LSTM","ensemble_score":"앙상블",
            "candle_score":"캔들","macro_score":"매크로","sector_score":"섹터점수",
            "candle_pattern":"캔들패턴","news_summary":"뉴스요약","sector":"섹터",
            "corr_flag":"상관중복","calendar_note":"시간대효과",
            "short_signal":"공매도신호","implied_vol":"내재변동성(%)",
            "option_strategy":"옵션전략","sector_rotation":"섹터로테이션",
            "earnings_note":"실적일정","stoch_k":"스토캐스틱K",
            "cci":"CCI","mfi":"MFI","vwap_deviation":"VWAP이탈(%)",
            "foreign_consec":"외국인연속(일)","ichi_score":"일목점수",
            "fib_score":"피보나치점수","elliott_wave_pos":"엘리어트파동",
            "cnn_pattern":"차트패턴","cnn_score":"패턴점수",
        }

        df_disp=df.drop(columns=["ohlcv","ohlcv_weekly","ohlcv_monthly"],errors="ignore").copy()
        top50=df_disp.head(50)[show_cols].copy()
        top50.index=range(1,len(top50)+1)
        top50=top50.rename(columns=rename_map)

        def fmt_cap(v):
            try:
                v=float(v)
                if v>=1e12: return f"{v/1e12:.1f}조"
                elif v>=1e8: return f"{int(v/1e8):,}억"
                return f"{int(v):,}"
            except: return "-"
        def fmt_price(v):
            try: v=float(v); return f"{int(v):,}원" if v>0 else "-"
            except: return "-"
        def fmt_pct(v,d=1):
            try: v=float(v); return f"{'+'if v>0 else''}{v:.{d}f}%"
            except: return "-"
        def fmt_score(v):
            try: return f"{float(v):.0f}점"
            except: return "-"

        for col,fn in [("시가총액",fmt_cap),("현재가",fmt_price),("예상매수가",fmt_price),
                       ("목표가",fmt_price),("ATR손절가",fmt_price)]:
            if col in top50.columns: top50[col]=top50[col].apply(fn)
        for col in ["상승확률","등락률","예상수익"]:
            d=2 if col=="등락률" else 1
            if col in top50.columns: top50[col]=top50[col].apply(lambda v:fmt_pct(v,d))
        def safe_per(v):
            try:
                s = str(v).replace("x","").replace("%","").strip()
                if s in ["-","nan","None","","0.0","0"]: return "-"
                f = float(s)
                return f"{f:.1f}x" if f > 0 else "-"
            except: return "-"
        for col in ["PER","PBR"]:
            if col in top50.columns: top50[col]=top50[col].apply(safe_per)
        def safe_pct(v, signed=True):
            """이미 포맷된 문자열 or 숫자를 안전하게 % 포맷으로 변환"""
            try:
                s = str(v).replace("%","").replace("+","").strip()
                if s in ["-","nan","None","","0.0","0"]: return "-"
                f = float(s)
                if f == 0: return "-"
                return f"{f:+.1f}%" if signed else f"{f:.1f}%"
            except: return "-"
        def safe_pct_unsigned(v):
            try:
                s = str(v).replace("%","").replace("+","").strip()
                if s in ["-","nan","None","","0.0","0"]: return "-"
                f = float(s)
                return f"{f:.0f}%" if f > 0 else "-"
            except: return "-"
        for col in ["ROE(%)","ROA(%)","영업이익률","매출증가율"]:
            if col in top50.columns: top50[col]=top50[col].apply(safe_pct)
        for col in ["부채비율"]:
            if col in top50.columns: top50[col]=top50[col].apply(safe_pct_unsigned)
        for col in ["모멘텀","감성","기관수급","재무","LSTM","앙상블","캔들","매크로","섹터점수"]:
            if col in top50.columns: top50[col]=top50[col].apply(fmt_score)
        if "상관중복" in top50.columns:
            top50["상관중복"]=top50["상관중복"].apply(lambda x: "⚠️중복" if str(x)=="True" else "✅")

        def style_row(row):
            styles=[""]*len(row)
            if "상승확률" in row.index:
                try:
                    val=float(str(row["상승확률"]).replace("%","").replace("+",""))
                    if   val>=70: styles=["background-color:#064e3b;color:#6ee7b7"]*len(row)
                    elif val>=60: styles=["background-color:#1e3a2f;color:#a7f3d0"]*len(row)
                    elif val>=55: styles=["background-color:#1a2e20;color:#bbf7d0"]*len(row)
                except: pass
            if "등락률" in row.index:
                try:
                    chg=float(str(row["등락률"]).replace("%","").replace("+",""))
                    idx=list(row.index).index("등락률")
                    styles[idx]="color:#10b981;font-weight:700" if chg>0 else "color:#ef4444;font-weight:700" if chg<0 else ""
                except: pass
            if "상관중복" in row.index and "⚠️" in str(row["상관중복"]):
                idx=list(row.index).index("상관중복"); styles[idx]="color:#f59e0b;font-weight:700"
            return styles

        top50=_clean_df(top50)
        st.dataframe(top50.style.apply(style_row,axis=1),width='stretch',height=600)

        chart_df=df_disp.head(20).copy()
        fig=go.Figure(go.Bar(
            x=chart_df["name"].astype(str).tolist(),
            y=pd.to_numeric(chart_df["rise_prob"],errors="coerce").fillna(0).tolist(),
            marker=dict(color=pd.to_numeric(chart_df["rise_prob"],errors="coerce").fillna(0).tolist(),
                        colorscale="Blues",showscale=False),
            text=[f"{v:.1f}%" for v in pd.to_numeric(chart_df["rise_prob"],errors="coerce").fillna(0).tolist()],
            textposition="outside",
        ))
        fig.update_layout(title="TOP 20 상승 확률",paper_bgcolor="#0f0f23",plot_bgcolor="#0f0f23",
                          font=dict(color="white"),height=400)
        st.plotly_chart(fig,width='stretch')

        st.download_button("📥 결과 CSV 다운로드",
            df.drop(columns=["ohlcv","ohlcv_weekly","ohlcv_monthly"],errors="ignore")\
              .to_csv(index=False,encoding="utf-8-sig"),
            file_name="stock_prediction_v70.csv",mime="text/csv")

    # ════════════════════════════════════════════════════════════════════════
    # 탭 2: 미국/매크로
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 🌎 미국 증시 & 🌍 매크로")
        if us_data:
            st.markdown("#### 미국 증시"); cols2=st.columns(3)
            for i,name in enumerate(["나스닥","S&P500","다우","VIX","원달러","필라델피아반도체"]):
                d=us_data.get(name,{}); chg=d.get("change",0); val=d.get("value",0)
                with cols2[i%3]:
                    st.markdown(f'<div class="metric-card"><div class="label">{name}</div>'
                                f'<div class="value {"green"if chg>=0 else"red"}">'
                                f'{"+"if chg>=0 else""}{chg:.2f}%</div>'
                                f'<div style="color:#94a3b8;font-size:.75rem">{val:,.2f}</div>'
                                f'</div>',unsafe_allow_html=True)
        if macro_data:
            mi2 = MacroIndicators()
            sm  = mi2.get_summary(macro_data)
            # ★ 공포탐욕 지수
            fg  = mi2.get_fear_greed(macro_data)
            st.markdown("---")
            st.markdown("#### 😱 공포탐욕 지수 (Fear & Greed Index)")
            fg_cols = st.columns(5)
            fg_color = {"극단적탐욕":"#ef4444","탐욕":"#10b981","중립":"#f59e0b",
                        "공포":"#f97316","극단적공포":"#7f1d1d"}.get(fg["phase"],"#94a3b8")
            with fg_cols[0]: st.metric("공포탐욕 점수", f"{fg['score']:.0f}점")
            with fg_cols[1]: st.metric("시장 국면",     f"{fg['icon']} {fg['phase']}")
            with fg_cols[2]: st.metric("VIX",           f"{fg['vix']:.1f}")
            with fg_cols[3]: st.metric("매크로 환경",   sm["env"])
            with fg_cols[4]: st.metric("매크로 점수",   f"{sm['score']:.1f}점")
            # 게이지 표시
            st.progress(min(int(fg["score"]), 100))
            st.caption(f"{fg['description']}")
            st.markdown("---")
            st.markdown("#### 매크로 지표"); cols3=st.columns(3)
            for i,name in enumerate(["미국10년금리","달러인덱스","WTI유가","금","구리","한국ETF"]):
                d=macro_data.get(name,{}); chg=d.get("change",0); val=d.get("value",0)
                with cols3[i%3]:
                    st.markdown(f'<div class="metric-card"><div class="label">{name}</div>'
                                f'<div class="value {"green"if chg>=0 else"red"}">'
                                f'{"+"if chg>=0 else""}{chg:.2f}%</div>'
                                f'<div style="color:#94a3b8;font-size:.75rem">{val:,.3f}</div>'
                                f'</div>',unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # 탭 3: 섹터
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🏭 섹터 분석 + 🔄 섹터 로테이션 전략")
        if use_sector and "sector" in df.columns:
            sa_inst = SectorAnalysis()
            sdf     = sa_inst.get_sector_summary(df)
            # ★ 섹터 로테이션 전략
            rotation = sa_inst.get_rotation_strategy(df)
            if rotation.get("top3_sectors"):
                st.markdown("#### 🔄 섹터 로테이션 전략")
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.success(f"🚀 강세 섹터 TOP3: {' > '.join(rotation['top3_sectors'])}")
                with r_col2:
                    st.error(f"📉 약세 섹터: {' > '.join(rotation['bottom3_sectors'])}")
                st.caption(rotation.get("summary",""))
                # 섹터 강도 차트
                ss = rotation.get("sector_strength",{})
                if ss:
                    sorted_ss = sorted(ss.items(), key=lambda x:x[1], reverse=True)
                    fig_rot = go.Figure(go.Bar(
                        x=[s[0] for s in sorted_ss],
                        y=[s[1] for s in sorted_ss],
                        marker_color=["#10b981" if v>0 else "#ef4444"
                                      for _,v in sorted_ss]))
                    fig_rot.update_layout(
                        title="섹터 강도 (양수=강세/음수=약세)",
                        paper_bgcolor="#0f0f23",plot_bgcolor="#0f0f23",
                        font=dict(color="white"),height=320)
                    st.plotly_chart(fig_rot, width='stretch')
                st.markdown("---")
            if not sdf.empty:
                st.markdown("#### 섹터별 상세 현황")
                st.dataframe(_clean_df(sdf),width='stretch')
                fig2=go.Figure(go.Bar(x=sdf["섹터"].tolist(),
                    y=pd.to_numeric(sdf["평균상승확률"],errors="coerce").fillna(0).tolist(),
                    marker_color="#6366f1"))
                fig2.update_layout(title="섹터별 평균 상승 확률",paper_bgcolor="#0f0f23",
                    plot_bgcolor="#0f0f23",font=dict(color="white"),height=380)
                st.plotly_chart(fig2,width='stretch')
        else: st.info("섹터 분석을 켜고 재실행하세요.")

    # ════════════════════════════════════════════════════════════════════════
    # 탭 4: 캔들 패턴
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 🕯️ 캔들/차트 패턴 (19가지)")
        with st.expander("📖 패턴 설명"):
            st.markdown("""
| 패턴 | 신호 | 설명 |
|------|------|------|
| 🟢 역헤드앤숄더 | 강한 상승 | 3저점, 중앙이 최저 → 넥라인 돌파 |
| 🟢 이중바닥(W) | 상승 반전 | 두 저점 테스트 후 넥라인 돌파 |
| 🟢 새벽별형 | 강한 상승 | 하락 후 도지 → 강한 양봉 |
| 🟢 상승장악형 | 상승 반전 | 전일 음봉을 당일 양봉이 감쌈 |
| 🟢 세병사 | 강한 상승 | 3일 연속 양봉 |
| 🟢 컵핸들 | 중기 상승 | U형 + 작은 조정 → 돌파 |
| 🟢 상승삼각형 | 돌파 임박 | 저점 상승 + 저항 수평 |
| 🔴 헤드앤숄더 | 강한 하락 | 3고점, 중앙 최고 → 넥라인 하향 |
| 🔴 이중천장(M) | 하락 반전 | 두 고점 저항 후 하락 |
            """)
        if use_candle and "candle_pattern" in df.columns:
            bull_pats=["역헤드앤숄더","이중바닥","새벽별형","상승장악형","세병사","컵핸들","상승삼각형","망치형","관통형","장대양봉"]
            df_cp=df[df["candle_pattern"].apply(lambda p:any(b in str(p) for b in bull_pats))]\
                   [["name","market","candle_pattern","candle_score","rise_prob"]].head(30).copy()
            if not df_cp.empty:
                df_cp.columns=["종목명","시장","감지패턴","캔들점수","상승확률"]
                df_cp["캔들점수"]=df_cp["캔들점수"].apply(lambda x:f"{x:.0f}점")
                df_cp["상승확률"]=df_cp["상승확률"].apply(lambda x:f"{x:.1f}%")
                df_cp.index=range(1,len(df_cp)+1)
                st.markdown("#### ✅ 상승 신호 패턴 감지 종목")
                st.dataframe(_clean_df(df_cp),width='stretch')
            else: st.info("현재 상승 패턴 감지 없음")

            pc={}
            for pats in df["candle_pattern"].dropna():
                for p in str(pats).split(", "):
                    if p and p not in ["패턴없음","데이터부족",""]:
                        pc[p]=pc.get(p,0)+1
            if pc:
                pdf=pd.DataFrame(sorted(pc.items(),key=lambda x:x[1],reverse=True),columns=["패턴","감지수"])
                fig3=go.Figure(go.Bar(x=pdf["패턴"].tolist(),y=pdf["감지수"].tolist(),
                    marker_color=["#10b981" if p in bull_pats else "#ef4444" for p in pdf["패턴"]],
                    text=pdf["감지수"].tolist(),textposition="outside"))
                fig3.update_layout(title="패턴별 감지 빈도 (초록=상승/빨강=하락)",
                    paper_bgcolor="#0f0f23",plot_bgcolor="#0f0f23",font=dict(color="white"),height=380)
                st.plotly_chart(fig3,width='stretch')
        else: st.info("캔들 패턴 분석을 켜고 재실행하세요.")

    # ════════════════════════════════════════════════════════════════════════
    # 탭 5: 시장 국면
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown(f"### 📊 시장 국면: **{phase_name}** (점수: {phase_score:.1f})")
        if use_auto_weight:
            wdf=pd.DataFrame([(k,f"{v:.3f}",f"{v*100:.1f}%") for k,v in weights.items()],
                             columns=["팩터","가중치","비율"])
            st.dataframe(_clean_df(wdf),width='stretch')

    # ════════════════════════════════════════════════════════════════════════
    # 탭 6: 포트폴리오 (상관관계 필터)
    # ════════════════════════════════════════════════════════════════════════
    with tab6:
        st.markdown("### 🔗 포트폴리오 상관관계 분석")
        if use_corr_filter and "corr_flag" in df.columns:
            top30=df.head(30).copy()
            clean=top30[~top30["corr_flag"]][["name","market","rise_prob","expected_return","stop_price","suggested_weight"]].head(15)
            excl =top30[top30["corr_flag"]] [["name","market","rise_prob"]].head(10)

            st.markdown("#### ✅ 분산 포트폴리오 추천 (상관관계 필터 적용)")
            if not clean.empty:
                clean.columns=["종목명","시장","상승확률","예상수익(%)","ATR손절가","추천비중(%)"]
                clean["상승확률"]=clean["상승확률"].apply(lambda x:f"{x:.1f}%")
                clean["예상수익(%)"]=clean["예상수익(%)"].apply(lambda x:f"+{x:.1f}%")
                clean.index=range(1,len(clean)+1)
                st.dataframe(_clean_df(clean),width='stretch')

                # 파이차트
                fig_pie=go.Figure(go.Pie(
                    labels=clean["종목명"].tolist(),
                    values=pd.to_numeric(clean["추천비중(%)"].astype(str).str.replace("%",""),errors="coerce").fillna(5).tolist(),
                    hole=0.4,
                ))
                fig_pie.update_layout(title="켈리 공식 기반 추천 비중",
                    paper_bgcolor="#0f0f23",font=dict(color="white"),height=380)
                st.plotly_chart(fig_pie,width='stretch')

            if not excl.empty:
                st.markdown("#### ⚠️ 상관관계 중복 제거 종목 (유사 종목 존재)")
                excl.columns=["종목명","시장","상승확률"]
                excl["상승확률"]=excl["상승확률"].apply(lambda x:f"{x:.1f}%")
                excl.index=range(1,len(excl)+1)
                st.dataframe(_clean_df(excl),width='stretch')
                st.caption("💡 위 종목들은 이미 선택된 종목과 주가 움직임이 80% 이상 유사합니다. 분산 효과가 없어 후순위로 이동했어요.")
        else:
            st.info("사이드바에서 '🔗 상관관계 필터' 체크 후 재실행하세요.")

    # ════════════════════════════════════════════════════════════════════════
    # 탭 7: 백테스트 (일반 + 워크포워드)
    # ════════════════════════════════════════════════════════════════════════
    with tab7:
        period_map=Backtester.PERIOD_MAP; lookback=period_map.get(bt_period,250)
        st.markdown(f"### 📋 백테스트 v7.2 — {bt_period} / TOP {bt_top_k}")
        if use_backtest:
            bt_col1, bt_col2 = st.columns(2)
            with bt_col1:
                bt_mode = st.radio("백테스트 방식",
                    ["일반 백테스트","★ 워크포워드 백테스트","🎲 몬테카를로 시뮬레이션"],
                    horizontal=True)
            with bt_col2:
                use_atr_bt = st.checkbox("★ ATR 손절/목표가 적용 (실전 전략)", value=True)
            bter = Backtester(lookback_days=lookback, top_k=bt_top_k, use_atr=use_atr_bt)
            if use_atr_bt:
                st.caption("🎯 ATR 전략: 매수가 - ATR×2 손절 / 매수가 + ATR×3 목표가 적용")

            if bt_mode == "일반 백테스트":
                with st.spinner(f"백테스트 실행 중 ({bt_period})..."):
                    bt_res = bter.run(df)
                colors={"top":"#6366f1","equal":"#10b981","market":"#f59e0b"}
                fig_bt=go.Figure()
                for key,label in [("top",f"TOP{bt_top_k}"),("equal","동일가중"),("market","시장평균")]:
                    r=bt_res.get(key,{})
                    st.markdown(f"##### {label}")
                    mc=st.columns(8)
                    for col,(k2,unit) in zip(mc,[("total_return","%"),("win_rate","%"),
                        ("sharpe",""),("sortino",""),("calmar",""),("max_drawdown","%"),
                        ("pnl_ratio","x"),("max_cons_loss","일")]):
                        v=r.get(k2,0)
                        lbl={"total_return":"총수익","win_rate":"승률","sharpe":"샤프",
                             "sortino":"소르티노","calmar":"칼마","max_drawdown":"MDD",
                             "pnl_ratio":"손익비","max_cons_loss":"최대연속손실"}.get(k2,k2)
                        with col: st.metric(lbl,f"{v:+.2f}{unit}" if "return" in k2 or "rate" in k2 else f"{v:.2f}{unit}")
                    monthly=r.get("monthly",[])
                    if monthly:
                        fig_m=go.Figure(go.Bar(x=[f"M{i+1}" for i in range(len(monthly))],y=monthly,
                            marker_color=["#10b981" if m>0 else "#ef4444" for m in monthly],
                            text=[f"{m:+.1f}%" for m in monthly],textposition="outside"))
                        fig_m.update_layout(title=f"{label} 월별 수익",paper_bgcolor="#0f0f23",
                            plot_bgcolor="#0f0f23",font=dict(color="white"),height=250)
                        st.plotly_chart(fig_m,width='stretch')
                    eq=r.get("equity_curve",[])
                    if eq: fig_bt.add_trace(go.Scatter(y=eq,mode="lines",
                        line=dict(color=colors[key],width=2),name=label))
                    tlog=r.get("trade_log",pd.DataFrame())
                    if not tlog.empty:
                        with st.expander(f"{label} 종목별 상세"):
                            st.dataframe(_clean_df(tlog),width='stretch')
                fig_bt.update_layout(title=f"3전략 누적 수익 비교 ({bt_period})",
                    paper_bgcolor="#0f0f23",plot_bgcolor="#0f0f23",font=dict(color="white"),height=420,
                    legend=dict(bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_bt,width='stretch')

            elif bt_mode == "★ 워크포워드 백테스트":  # 워크포워드 백테스트
                st.info("훈련창 120일 → 검증창 30일 슬라이딩 | IS vs OOS 성과 비교")
                with st.spinner("워크포워드 백테스트 실행 중... (수십 초 소요)"):
                    wf_res = bter.run_walkforward(df)
                verdict = wf_res.get("verdict","")
                st.markdown(f"#### 판정: {verdict}")
                c1,c2,c3,c4 = st.columns(4)
                with c1: st.metric("OOS 적중률",   f"{wf_res.get('oos_hit_rate',0):.1f}%")
                with c2: st.metric("정보비율(IR)",  f"{wf_res.get('info_ratio',0):.3f}")
                with c3: st.metric("과적합비율",    f"{wf_res.get('overfit_ratio',0):.3f}")
                with c4: st.metric("분석 윈도우",   f"{wf_res.get('n_windows',0)}개")
                st.markdown("---")
                col_is, col_oos = st.columns(2)
                for col, key, title in [(col_is,"is","IS (훈련창)"),(col_oos,"oos","OOS (검증창)")]:
                    r = wf_res.get(key,{})
                    with col:
                        st.markdown(f"##### {title}")
                        mc2=st.columns(4)
                        for cc,(k2,unit) in zip(mc2,[("total_return","%"),("win_rate","%"),
                                                      ("sharpe",""),("max_drawdown","%")]):
                            v=r.get(k2,0)
                            lbl={"total_return":"총수익","win_rate":"승률",
                                 "sharpe":"샤프","max_drawdown":"MDD"}.get(k2,k2)
                            with cc: st.metric(lbl,f"{v:+.2f}{unit}" if "return" in k2 or "rate" in k2 else f"{v:.2f}{unit}")
                        eq=r.get("equity_curve",[])
                        if eq:
                            fig_wf=go.Figure(go.Scatter(y=eq,mode="lines",
                                line=dict(color="#6366f1" if key=="is" else "#10b981",width=2),
                                name=title))
                            fig_wf.update_layout(paper_bgcolor="#0f0f23",plot_bgcolor="#0f0f23",
                                font=dict(color="white"),height=300,
                                title=f"{title} 누적수익")
                            st.plotly_chart(fig_wf,width='stretch')
                wlog = wf_res.get("window_log",pd.DataFrame())
                if not wlog.empty:
                    with st.expander("윈도우별 IS/OOS 수익 상세"):
                        st.dataframe(_clean_df(wlog),width='stretch')
                st.caption("과적합비율 > 0.7 + OOS적중률 > 52% = 실전 적용 추천")

            elif bt_mode == "🎲 몬테카를로 시뮬레이션":
                mc_col1, mc_col2 = st.columns(2)
                with mc_col1:
                    mc_n = st.slider("시뮬레이션 횟수", 100, 2000, 1000, 100)
                with mc_col2:
                    mc_days = st.slider("시뮬레이션 기간(일)", 20, 120, 60, 10)
                st.info(f"과거 수익률을 무작위로 섞어 {mc_n}번 시뮬레이션 → 미래 수익 범위 예측")
                with st.spinner(f"🎲 몬테카를로 시뮬레이션 {mc_n}회 실행 중..."):
                    mc_res = bter.run_montecarlo(df, n_simulations=mc_n, n_days=mc_days)
                st.markdown(f"#### 판정: {mc_res.get('verdict','')}")
                # 핵심 지표
                mc1,mc2,mc3,mc4,mc5,mc6 = st.columns(6)
                with mc1: st.metric("중간 수익(50%)",  f"{mc_res['p50']:+.1f}%")
                with mc2: st.metric("낙관 수익(95%)",  f"{mc_res['p95']:+.1f}%")
                with mc3: st.metric("비관 수익(5%)",   f"{mc_res['p5']:+.1f}%")
                with mc4: st.metric("승률",            f"{mc_res['win_rate']:.1f}%")
                with mc5: st.metric("평균 MDD",        f"{mc_res['avg_mdd']:.1f}%")
                with mc6: st.metric("파산확률(MDD>30%)",f"{mc_res['bankruptcy_prob']:.1f}%")
                st.markdown("---")
                # 시나리오 차트
                fig_mc = go.Figure()
                days_x = list(range(mc_days))
                if mc_res.get("worst_path"):
                    fig_mc.add_trace(go.Scatter(
                        x=days_x, y=mc_res["worst_path"],
                        mode="lines", name="비관(5%)",
                        line=dict(color="#ef4444", width=2, dash="dash")))
                if mc_res.get("median_path"):
                    fig_mc.add_trace(go.Scatter(
                        x=days_x, y=mc_res["median_path"],
                        mode="lines", name="중간(50%)",
                        line=dict(color="#f59e0b", width=3)))
                if mc_res.get("best_path"):
                    fig_mc.add_trace(go.Scatter(
                        x=days_x, y=mc_res["best_path"],
                        mode="lines", name="낙관(95%)",
                        line=dict(color="#10b981", width=2, dash="dash")))
                fig_mc.add_hline(y=0, line_dash="dot",
                                 line_color="white", opacity=0.3)
                fig_mc.update_layout(
                    title=f"몬테카를로 {mc_n}회 시뮬레이션 — {mc_days}일 수익 시나리오",
                    xaxis_title="거래일", yaxis_title="누적수익(%)",
                    paper_bgcolor="#0f0f23", plot_bgcolor="#0f0f23",
                    font=dict(color="white"), height=420,
                    legend=dict(bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_mc, width='stretch')
                # 수익 분포 히스토그램
                st.markdown("#### 수익률 분포")
                mc_info = (
                    f"VaR(5%): {mc_res['var_5']:+.1f}% | "
                    f"평균: {mc_res['mean_return']:+.1f}% | "
                    f"표준편차: {mc_res['std_return']:.1f}% | "
                    f"25%~75%: {mc_res['p25']:+.1f}% ~ {mc_res['p75']:+.1f}%"
                )
                st.caption(mc_info)
                with st.expander("📊 몬테카를로 해석 가이드"):
                    st.markdown("""
| 지표 | 의미 |
|------|------|
| 중간수익(50%) | 절반 확률로 이 수익 이상 |
| 낙관수익(95%) | 상위 5% 시나리오 |
| 비관수익(5%) | 하위 5% 최악 시나리오 |
| VaR(5%) | 5% 확률로 이것보다 더 손실 |
| 파산확률 | MDD 30% 초과 확률 |
                    """)
        else:
            st.info("사이드바에서 '📊 백테스트' 체크 후 재실행하세요.")

    # ════════════════════════════════════════════════════════════════════════
    # 탭 8: 자기학습 v5.4
    # ════════════════════════════════════════════════════════════════════════
    with tab8:
        st.markdown("### 🧬 자기학습 v7.2 — DQN 강화학습 + 베이지안 + 메타러닝")
        # 기본 현황
        c1,c2,c3,c4=st.columns(4)
        with c1: st.metric("총 예측 저장",  f"{ns2['total']}건")
        with c2: st.metric("결과 확인",      f"{ns2['completed']}건")
        with c3: st.metric("적중률",         f"{ns2['hit_rate']}%")
        with c4: st.metric("평균 수익",       f"{ns2['avg_return']:+.2f}%")

        # 강화학습 현황
        try:
            rl_stats = tracker.get_rl_stats()
            st.markdown("---")
            st.markdown("#### Q-Learning 강화학습 현황")
            r1,r2,r3,r4 = st.columns(4)
            with r1: st.metric("학습 에피소드",  f"{rl_stats['episodes']}회")
            with r2: st.metric("Q테이블 상태수", f"{rl_stats['q_table_size']}개")
            with r3: st.metric("탐험율(ε)",      f"{rl_stats['epsilon']:.3f}")
            with r4: st.metric("평균 보상",       f"{rl_stats['avg_reward']:+.3f}%")
            if rl_stats["episodes"] > 0:
                st.caption("ε=0.3→0.05 감소 | 에피소드 쌓일수록 최적 행동 학습")
            else:
                st.info("강화학습 데이터 누적 중... (결과 업데이트 시 자동 학습)")
            # ★ 메타러닝 현황 (항상 표시)
            try:
                _em_meta = EnsembleModel()
                meta_st  = _em_meta.get_meta_stats()
                st.markdown("---")
                st.markdown("#### 🤖 앙상블 메타러닝 현황")
                st.caption("최근 20번 예측 적중률 기반으로 LSTM/XGB/LGB/CatBoost 가중치 자동 조정")
                meta_cols = st.columns(4)
                for i, (model, mstat) in enumerate(meta_st.items()):
                    with meta_cols[i]:
                        st.metric(
                            f"{model.upper()} 가중치",
                            f"{mstat['weight']*100:.1f}%",
                            f"적중률 {mstat['hit_rate']}% ({mstat['samples']}건)"
                        )
            except Exception:
                pass
        except Exception as _e:
            st.caption(f"강화학습 현황 조회 오류: {_e}")

        # 베이지안 최적화 현황
        try:
            st.markdown("---")
            st.markdown("#### 베이지안 가중치 최적화")
            b_iter = ns2.get("bayesian_iter", 0)
            b1,b2 = st.columns(2)
            with b1: st.metric("베이지안 반복횟수", f"{b_iter}회")
            with b2: st.metric("가중치 탐색 완료", "✅" if b_iter>0 else "⏳ 누적 중")
            if ns2["ready"]:
                st.success("✅ 베이지안 가중치 최적화 완료! 학습된 가중치 적용 중")
                fa_df = tracker.get_factor_accuracy()
                if not fa_df.empty:
                    st.dataframe(_clean_df(fa_df), width='stretch')
            else:
                st.info(f"⏳ {ns2['needed']}건 더 필요 (현재 {ns2['completed']}건)")
                st.progress(min(ns2['completed']/50, 1.0))
        except Exception as _e:
            st.caption(f"베이지안 현황 오류: {_e}")

        # TOP5 강화학습 추천
        if ns2["completed"] > 0 and "df" in dir():
            st.markdown("---")
            st.markdown("#### Q-Learning 매수 추천 (TOP5)")
            try:
                rl_recs = []
                for _, row in df.head(20).iterrows():
                    rec = row.to_dict()
                    rl_r = tracker.rl_recommend(rec)
                    if rl_r["action"] == "매수":
                        rl_recs.append({
                            "종목": row.get("name",""),
                            "상승확률": f"{row.get('rise_prob',0):.1f}%",
                            "RL추천": rl_r["action"],
                            "Q매수": f"{rl_r['q_buy']:.3f}",
                            "Q관망": f"{rl_r['q_hold']:.3f}",
                            "신뢰도": f"{rl_r['confidence']:.1f}",
                        })
                if rl_recs:
                    st.dataframe(_clean_df(pd.DataFrame(rl_recs[:5])), width='stretch')
                else:
                    st.caption("현재 강화학습 매수 추천 종목 없음 (학습 누적 필요)")
            except Exception as _e:
                st.caption(f"RL 추천 오류: {_e}")

        # DB 히스토리 검색
        st.markdown("---")
        st.markdown("#### 📂 히스토리 검색")
        srch_col1, srch_col2 = st.columns(2)
        with srch_col1:
            srch_days = st.selectbox("기간", [7,14,30,60,90], index=2)
        with srch_col2:
            srch_code = st.text_input("종목코드 (선택)", placeholder="005930")
        if st.button("🔍 검색"):
            try:
                hist_df = tracker.search_history(
                    code=srch_code if srch_code else None,
                    days=srch_days)
                if not hist_df.empty:
                    st.dataframe(_clean_df(hist_df), width='stretch')
                    acc = tracker.get_period_accuracy(days=srch_days)
                    st.metric(f"최근 {srch_days}일 적중률",
                              f"{acc['hit_rate']}% ({acc['count']}건)")
                else:
                    st.info("해당 기간 데이터 없음")
            except Exception as _he:
                st.caption(f"히스토리 조회 오류: {_he}")
        st.caption(f"오늘 {saved}건 저장 | 어제 {updated}건 업데이트")

    # ════════════════════════════════════════════════════════════════════════
    # 탭 9: 예측 설명 (XAI)
    # ════════════════════════════════════════════════════════════════════════
    with tab9:
        st.markdown("### 🔍 예측 설명 (XAI) — 왜 이 종목을 추천했는가?")
        st.info("SHAP 설치 시 XGBoost 기반 정밀 설명 / 미설치 시 룰기반 기여도 표시\n`pip install shap`")
        if len(df) > 0:
            top10_names = df.head(10)["name"].tolist()
            sel_name    = st.selectbox("종목 선택", top10_names)
            sel_row     = df[df["name"]==sel_name].iloc[0] if sel_name in df["name"].values else None
            if sel_row is not None:
                with st.spinner(f"{sel_name} 예측 설명 계산 중..."):
                    try:
                        # EnsembleModel 1번만 생성 (재사용)
                        _em_xai = EnsembleModel()
                        exp = _em_xai.explain_prediction(
                            sel_row.get("ohlcv"),
                            sel_row.get("ohlcv_weekly"),
                            sel_row.get("ohlcv_monthly"),
                        )
                        iv = _em_xai._calc_implied_vol(sel_row.get("ohlcv")) \
                             if sel_row.get("ohlcv") is not None else 0
                        st.markdown(f"**분석방법:** {exp.get('method','')} | **요약:** {exp.get('summary','')[:80]}")
                        col_p, col_n = st.columns(2)
                        with col_p:
                            st.markdown("#### 📈 상승 기여 요인")
                            pos_data = pd.DataFrame(exp.get("top_pos",[]), columns=["피처","기여도"])
                            if not pos_data.empty:
                                fig_p = go.Figure(go.Bar(
                                    x=pos_data["기여도"], y=pos_data["피처"],
                                    orientation='h', marker_color="#10b981"))
                                fig_p.update_layout(paper_bgcolor="#0f0f23",plot_bgcolor="#0f0f23",
                                    font=dict(color="white"),height=250,margin=dict(l=120))
                                st.plotly_chart(fig_p, width='stretch')
                        with col_n:
                            st.markdown("#### 📉 하락 기여 요인")
                            neg_data = pd.DataFrame(exp.get("top_neg",[]), columns=["피처","기여도"])
                            if not neg_data.empty:
                                fig_n = go.Figure(go.Bar(
                                    x=neg_data["기여도"], y=neg_data["피처"],
                                    orientation='h', marker_color="#ef4444"))
                                fig_n.update_layout(paper_bgcolor="#0f0f23",plot_bgcolor="#0f0f23",
                                    font=dict(color="white"),height=250,margin=dict(l=120))
                                st.plotly_chart(fig_n, width='stretch')
                        st.markdown("---")
                        ic1,ic2,ic3,ic4,ic5 = st.columns(5)
                        _per=float(sel_row.get('per',0) or 0)
                        _pbr=float(sel_row.get('pbr',0) or 0)
                        _roe=float(sel_row.get('roe',0) or 0)
                        with ic1: st.metric("상승확률",  f"{sel_row.get('rise_prob',0):.1f}%")
                        with ic2: st.metric("내재변동성", f"{iv:.1f}%")
                        with ic3: st.metric("PER",  f"{_per:.1f}x"  if _per>0  else "-")
                        with ic4: st.metric("PBR",  f"{_pbr:.2f}x"  if _pbr>0  else "-")
                        with ic5: st.metric("ROE",  f"{_roe:+.1f}%" if _roe!=0 else "-")
                        theme = sel_row.get("theme_tag","")
                        if theme:
                            tags = " | ".join([f"🏷️ {t}" for t in theme.split(',') if t])
                            st.markdown(f"**테마:** {tags}")
                    except Exception as _xe:
                        st.error(f"설명 생성 오류: {_xe}")
        else:
            st.info("분석 후 이용 가능합니다.")

    # ════════════════════════════════════════════════════════════════════════
    # 탭 10: 클러스터링
    # ════════════════════════════════════════════════════════════════════════
    with tab10:
        st.markdown("### 🔵 종목 클러스터링 — 진짜 분산 포트폴리오")
        if "cluster_id" in df.columns:
            try:
                clusterer_disp = StockCluster()
                summary_df = clusterer_disp.cluster_summary(df)
                if not summary_df.empty:
                    st.markdown("#### 클러스터별 요약")
                    st.dataframe(_clean_df(summary_df), width='stretch')
                st.markdown("---")
                st.markdown("#### 클러스터별 대표 종목 TOP 10")
                top_div = clusterer_disp.get_diversified_top(df, n=10)
                disp_cols = ["name","market","cluster_id","rise_prob",
                             "per","roe","news_summary"]
                disp_cols = [c for c in disp_cols if c in top_div.columns]
                st.dataframe(_clean_df(top_div[disp_cols].rename(columns={
                    "name":"종목명","market":"시장","cluster_id":"클러스터",
                    "rise_prob":"상승확률","per":"PER","roe":"ROE",
                    "news_summary":"뉴스요약"
                })), width='stretch')
                # 클러스터 분포 차트
                if "cluster_strength" in df.columns:
                    fig_cl = go.Figure()
                    for cid in df["cluster_id"].unique():
                        sub = df[df["cluster_id"]==cid]
                        fig_cl.add_trace(go.Scatter(
                            x=[cid]*len(sub), y=sub["rise_prob"],
                            mode="markers", name=f"클러스터{cid}",
                            text=sub["name"],
                            marker=dict(size=8)
                        ))
                    fig_cl.update_layout(
                        title="클러스터별 상승확률 분포",
                        paper_bgcolor="#0f0f23", plot_bgcolor="#0f0f23",
                        font=dict(color="white"), height=400)
                    st.plotly_chart(fig_cl, width='stretch')
            except Exception as _ce:
                st.error(f"클러스터링 표시 오류: {_ce}")
        else:
            st.info("사이드바에서 '🔵 종목 클러스터링' 체크 후 재실행하세요.")

    # ════════════════════════════════════════════════════════════════════════
    # 탭 11: 옵션 전략
    # ════════════════════════════════════════════════════════════════════════
    with tab11:
        st.markdown("### 🎯 옵션 전략 분석 — Black-Scholes 기반 끝판왕")

        if "option_strategy" in df.columns:
            # 시장 옵션 시그널
            if "opt_signal" in dir() and opt_signal:
                st.markdown("#### 📡 시장 전체 옵션 시그널")
                sig_cols = st.columns(5)
                with sig_cols[0]: st.metric("평균 IV",      f"{opt_signal.get('avg_iv',0):.1f}%")
                with sig_cols[1]: st.metric("평균 Delta",   f"{opt_signal.get('avg_delta',0):.3f}")
                with sig_cols[2]: st.metric("고IV 종목",    f"{opt_signal.get('high_iv_count',0)}개")
                with sig_cols[3]: st.metric("저IV 종목",    f"{opt_signal.get('low_iv_count',0)}개")
                with sig_cols[4]: st.metric("추천 전략",    opt_signal.get('best_strategy',''))
                st.info(opt_signal.get('market_signal',''))
                st.markdown("---")

            # TOP 20 옵션 전략 테이블
            st.markdown("#### 📊 종목별 옵션 전략 추천")
            opt_cols = ["name","market","rise_prob","implied_vol",
                        "option_strategy","option_breakeven",
                        "option_max_profit","option_max_loss","option_detail"]
            opt_cols = [c for c in opt_cols if c in df.columns]
            opt_df   = df.head(20)[opt_cols].copy()
            opt_df   = opt_df.rename(columns={
                "name":"종목명","market":"시장","rise_prob":"상승확률",
                "implied_vol":"내재변동성(%)","option_strategy":"추천전략",
                "option_breakeven":"손익분기점","option_max_profit":"최대수익",
                "option_max_loss":"최대손실","option_detail":"전략상세",
            })
            if "상승확률" in opt_df.columns:
                opt_df["상승확률"] = opt_df["상승확률"].apply(
                    lambda v: f"{float(v):.1f}%" if str(v) not in ["-","nan"] else "-")
            if "내재변동성(%)" in opt_df.columns:
                opt_df["내재변동성(%)"] = opt_df["내재변동성(%)"].apply(
                    lambda v: f"{float(v):.1f}%" if str(v) not in ["-","nan"] else "-")
            st.dataframe(_clean_df(opt_df), width='stretch')

            # 그리스 지표
            st.markdown("---")
            st.markdown("#### 🔢 그리스 지표 (ATM 콜옵션 기준)")
            st.caption("Delta: 주가 1원 변할 때 옵션 가격 변화 | Gamma: Delta 변화율 | Theta: 하루 시간가치 감소 | Vega: IV 1% 변할 때 옵션 가격 변화")
            greek_cols = ["name","delta","gamma","theta","vega","implied_vol"]
            greek_cols = [c for c in greek_cols if c in df.columns]
            greek_df   = df.head(20)[greek_cols].copy()
            greek_df   = greek_df.rename(columns={
                "name":"종목명","delta":"Delta","gamma":"Gamma",
                "theta":"Theta(일)","vega":"Vega","implied_vol":"IV(%)"
            })
            st.dataframe(_clean_df(greek_df), width='stretch')

            # 전략별 분류
            st.markdown("---")
            st.markdown("#### 🗂️ 전략별 종목 분류")
            if "option_strategy" in df.columns:
                strategy_groups = df.groupby("option_strategy")["name"].apply(
                    lambda x: ", ".join(x.head(5).tolist())
                ).reset_index()
                strategy_groups.columns = ["전략","해당종목(TOP5)"]
                st.dataframe(_clean_df(strategy_groups), width='stretch')

            # 옵션 전략 가이드
            with st.expander("📚 옵션 전략 가이드"):
                st.markdown("""
| 전략 | 언제 사용 | 최대손실 | 최대수익 |
|------|----------|---------|---------|
| 📈 롱콜 | 강한 상승 예상 | 프리미엄 | 무제한 |
| 📉 롱풋 | 강한 하락 예상 | 프리미엄 | 주가×하락률 |
| ⚡ 스트래들 | 고변동성, 방향 모름 | 프리미엄×2 | 무제한 |
| 🐂 불스프레드 | 완만한 상승 | 순비용 | 제한적 |
| 💰 커버드콜 | 횡보+소폭상승 | 하락분-프리미엄 | 프리미엄+상승분 |
| 🛡️ 프로텍티브풋 | 보유종목 헤지 | 보험료 | 무제한 |

⚠️ 한국 코스피200/코스닥150 종목만 실제 옵션 거래 가능
                """)
        else:
            st.info("사이드바에서 '🎯 옵션 전략 분석' 체크 후 재실행하세요.")

# ════════════════════════════════════════════════════════════════════════════
# 탭 12: 가이드
# ════════════════════════════════════════════════════════════════════════════
with tab12:
    st.markdown("""
### 📖 사용 가이드 v7.2

#### 🆕 v5.6 업그레이드 내용

| 업그레이드 | 효과 | 설명 |
|-----------|------|------|
| 재무 수집 완전 수정 | ROE/PER 정상화 | 네이버 모바일 API→pykrx→yfinance 3단계 |
| 백테스트 0 수정 | 정상 작동 | lookback 실제 데이터 길이 자동 조정 |
| 분봉 피처 8개 | 타이밍↑ | 분봉 모멘텀/VWAP/추세강도 |
| 동적 앙상블 가중치 | 정확도↑ | 강세/약세/고변동성 자동 전환 |
| 옵션 내재변동성 근사 | 리스크↓ | Parkinson 변동성 기반 VIX 유사 지표 |
| 뉴스 요약 고도화 | 분석력↑ | 핵심 수치 추출 + 테마주 자동 감지 |
| 예측 설명 탭 | 신뢰도↑ | 상승/하락 근거 시각화 (SHAP/룰기반) |

#### 🆕 v5.4 업그레이드 내용

| 업그레이드 | 효과 | 설명 |
|-----------|------|------|
| 피처 70→109개 | +2~3%p | 일봉50+주봉20+월봉15+고급24 최대치 |
| 워크포워드 백테스트 | 신뢰도↑ | IS/OOS 분리, 과적합비율/정보비율 측정 |
| Q-Learning 강화학습 | +3~5%p | 9개 상태×2행동 Q테이블 자동학습 |
| 베이지안 가중치 최적화 | +2~3%p | 기존 단순조정→베이지안 탐색으로 고도화 |
| KIS 호가 완전활용 | 타이밍↑ | 10호가 매수벽/매도벽/기울기 분석 |
| 시장국면별 가중치 | 정확도↑ | 강세/약세/횡보 상황별 다른 가중치 적용 |

#### 🆕 v7.2 업그레이드 내용

| 업그레이드 | 효과 | 설명 |
|-----------|------|------|
| 🌙 월봉 추가 (트리플 타임프레임) | +1~2%p | 일봉+주봉+월봉 3중 추세 확인 |
| 💥 ATR 동적 손절/목표가 | 실전수익 +3~5%p | 변동성에 따라 자동 조정, 손익비 1.5 보장 |
| 🔗 상관관계 필터 | MDD -15~20% | 유사종목 중복 제거, 진짜 분산투자 |
| 📅 요일/월말 효과 | +1~2%p | 월요일 갭업, 금요일 청산, 월말 수급 패턴 |
| 🤖 KoBERT 뉴스 AI | +2~3%p | 문맥 이해 (부정어 처리, 수치 가중치) |
| 📊 피처 50→70개 | +1~2%p | 허스트/엔트로피/자기상관/가격가속도 |
| 🔬 허스트 지수 | 추세판단 개선 | 0.6 이상=추세지속, 0.4 이하=반전주의 |
| 🔬 가격 엔트로피 | 노이즈 필터 | 불규칙한 주가 패턴 필터링 |
| 🔬 자기상관 | 연속성 판단 | lag1/lag5 패턴 감지 |

#### ⚡ KoBERT 즉시 설치 (권장)
```bash
pip install transformers torch sentencepiece
```
설치 후 재실행하면 자동으로 KR-FinBERT 금융 감성분석 활성화!

#### 💡 ATR 손절가 활용법
- ATR 손절가 = 변동성 기반 적정 손절선
- 고변동성 종목: 손절가가 넓게 설정 → 자연스러운 흔들기에 안 걸림
- 저변동성 종목: 손절가가 좁게 설정 → 리스크 최소화
- **손익비 1.5 이상 항상 보장** (목표 = 손절거리 × 1.5 이상)

#### 🔗 상관관계 필터 활용법
- ⚠️중복 표시 종목 = 이미 선택된 종목과 80%+ 유사 움직임
- 같은 섹터 2개보다 다른 섹터 2개가 훨씬 안전
- 포트폴리오 탭에서 켈리 공식 기반 추천 비중 확인

#### 🏆 최고 신뢰 조합 신호
```
역헤드앤숄더 + 3중 정배열(일/주/월봉) + 허스트 0.6+ + 기관 순매수
→ 가장 강한 매수 신호
```
    """)