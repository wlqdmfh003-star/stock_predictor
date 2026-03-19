import pandas as pd
import numpy as np
from utils.indicators import TechnicalIndicators

dates  = pd.date_range(end='2026-03-09', periods=120, freq='B')
prices = 50000 * (1 + np.random.randn(120).cumsum() * 0.01)
ohlcv  = pd.DataFrame({
    'close':  prices,
    'open':   prices * 0.999,
    'high':   prices * 1.01,
    'low':    prices * 0.99,
    'volume': np.random.randint(100000, 500000, 120).astype(float),
}, index=dates)

test_df = pd.DataFrame({'code': ['005930'], 'name': ['삼성전자'], 'current_price': [float(prices[-1])]})
test_df['ohlcv'] = pd.Series([ohlcv], dtype=object)

result = TechnicalIndicators().calculate_all(test_df)
print("컬럼 수:", len(result.columns))
print("sharpe_60:", result['sharpe_60'].iloc[0] if 'sharpe_60' in result.columns else '❌')
print("candle_score:", result['candle_score'].iloc[0] if 'candle_score' in result.columns else '❌')
print("vp_poc:", result['vp_poc'].iloc[0] if 'vp_poc' in result.columns else '❌')
print("risk_score:", result['risk_score'].iloc[0] if 'risk_score' in result.columns else '❌')