#
# ASR 프로젝트의 flask-service/services/prediction_service.py (신규 파일)
# FF 프로젝트(final_model.py, model_analyzer.py, backend_logic.py)의 핵심 로직을 이식합니다.
#

import pandas as pd
import numpy as np
import re
import FinanceDataReader as fdr
import yfinance as yf
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas_ta as ta # ASR flask-service/requirements.txt 에 pandas_ta 추가가 필요할 수 있습니다.

# --- FF 프로젝트의 technical_analyzer.py (backend_logic.py 내) 로직 ---
def calculate_technical_indicators(df):
    """주어진 데이터프레임에 기술적 지표를 계산하여 추가합니다."""
    if df is None or df.empty:
        return None
    if 'Close' not in df.columns:
        print("오류: calculate_technical_indicators - 'Close' 컬럼 없음")
        return None

    try:
        if 'Date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
             print("오류: calculate_technical_indicators - 유효한 날짜 인덱스가 없습니다.")
             return None

        df.ta.rsi(close='Close', length=14, append=True)
        df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        df.ta.cci(close='Close', length=20, append=True)
        df.ta.stoch(close='Close', k=14, d=3, smooth_k=3, append=True)
        df.ta.bbands(close='Close', length=20, append=True)
        df.ta.willr(close='Close', length=14, append=True)
        df.ta.ema(close='Close', length=20, append=True)
        df.ta.ema(close='Close', length=50, append=True)
        df['Change'] = df['Close'].pct_change() * 100

        cols_to_drop = [col for col in df.columns if col.startswith(('BBL_', 'BBM_', 'BBU_', 'BBB_', 'STOCHk_', 'STOCHd_'))]
        if 'STOCHk_14_3_3' in df.columns: cols_to_drop.remove('STOCHk_14_3_3')
        if 'STOCHd_14_3_3' in df.columns: cols_to_drop.remove('STOCHd_14_3_3')
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        df.rename(columns={
            'RSI_14': 'RSI',
            'MACD_12_26_9': 'MACD',
            'MACDh_12_26_9': 'MACD_Hist',
            'MACDs_12_26_9': 'MACD_Signal',
            'CCI_20_0.015': 'CCI',
            'STOCHk_14_3_3': 'STOCHk',
            'STOCHd_14_3_3': 'STOCHd',
            'BBP_20_2.0': 'BB_Percent',
            'WILLR_14': 'WilliamsR',
            'EMA_20': 'EMA20',
            'EMA_50': 'EMA50'
        }, inplace=True)

        return df
    except Exception as e:
        print(f"오류: calculate_technical_indicators 내부 오류: {e}")
        return None

# --- FF 프로젝트의 model_analyzer.py 로직 ---
def get_stock_data_for_analysis(ticker, years=3):
    """분석을 위한 데이터를 준비합니다."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    df_raw = fdr.DataReader(ticker, start_date, end_date)
    
    if df_raw.empty:
        return None, None

    df = calculate_technical_indicators(df_raw.copy())
    if df is None: # 지표 계산 실패 시
        return None, None
        
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    features = [col for col in df.columns if col != 'Target']
    X = df[features]
    y = df['Target']
    
    return X, y

def analyze_feature_importance(ticker, top_n=3):
    """가장 중요한 상위 N개의 피처 이름을 반환합니다."""
    X, y = get_stock_data_for_analysis(ticker)
    
    if X is None or X.empty:
        return None

    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        max_depth=10,
        min_samples_leaf=5
    )
    model.fit(X, y)
    
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)
    return sorted_importances.head(top_n).index.tolist()

# --- FF 프로젝트의 final_model.py (데이터 로더) ---
def get_yfinance_data_safely(ticker: str, years: int = 1) -> pd.DataFrame:
    """yfinance 데이터를 안전하게 로드합니다 (FF의 final_model.py에서 가져옴)."""
    print(f"[{ticker}] 최신 {years}년 데이터 로드 중... (yfinance)")
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=years)
    
    data = None
    MIN_DATA_ROWS = 50
    
    if re.fullmatch(r"\d{6}", ticker):
        try:
            data_ks = yf.download(f"{ticker}.KS", start=start_date, end=end_date)
            if not data_ks.empty and len(data_ks) >= MIN_DATA_ROWS: data = data_ks
        except Exception: pass

        if data is None:
            try:
                data_kq = yf.download(f"{ticker}.KQ", start=start_date, end=end_date)
                if not data_kq.empty and len(data_kq) >= MIN_DATA_ROWS: data = data_kq
            except Exception: pass
    else:
        try:
            data_normal = yf.download(ticker, start=start_date, end=end_date)
            if not data_normal.empty and len(data_normal) >= MIN_DATA_ROWS: data = data_normal
        except Exception: pass

    if data is None or data.empty or len(data) < MIN_DATA_ROWS:
        print(f"[{ticker}] KST 주가 데이터 로드 실패 또는 부족.")
        return pd.DataFrame()
    
    data.reset_index(inplace=True) 
    return data

# --- [ASR용 수정] FF의 _merge_sentiment_data (단순 통과) ---
def _merge_sentiment_data(df_raw, feature_ticker, start_date, end_date):
    """
    ASR 이식용 스텁(Stub) 함수.
    FF 프로젝트의 DB 의존성을 제거하기 위해 감성분석 없이 원본 DF를 반환합니다.
    (추후 ASR의 news_analysis_service와 연동 필요)
    """
    print(f"[{feature_ticker}] 감성 분석 스텁 호출됨 (현재 기능 비활성화).")
    # fillna(0)을 시뮬레이션하기 위해 컬럼 추가
    df_raw['sentiment_score'] = 0.0
    return df_raw

# --- FF 프로젝트의 final_model.py (핵심 예측 함수) ---
def get_stock_prediction(ticker: str, years: int = 1, data_df: pd.DataFrame = None) -> dict:
    """
    FF의 predict_stock 함수를 ASR 프로젝트용으로 수정한 메인 함수.
    """
    empty_return = {"direction": 0, "price": 0, "current_price": 0, "features": [], "status": "Error"}
    
    fdr_ticker = ticker
    print(f"[{fdr_ticker}] 예측 시작 (데이터: {'제공됨' if data_df is not None else f'최신 {years}년'})...")

    try:
        feature_ticker = ticker.split('.')[0] if isinstance(ticker, str) else ticker
        top_3_features = analyze_feature_importance(feature_ticker, top_n=3)
        if not top_3_features:
            print(f"[{fdr_ticker}] Top 3 피처 선택 실패.")
            return {**empty_return, "status": "Top 3 피처 선택 실패"}

        if data_df is not None and not data_df.empty:
            df_raw = data_df.copy()
        else:
            df_raw = get_yfinance_data_safely(fdr_ticker, years)
            if df_raw.empty or 'Close' not in df_raw.columns or len(df_raw) < 50:
                return {**empty_return, "status": "주가 데이터 로드 실패"}

        if 'Date' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Date'])
            df_raw.set_index('Date', inplace=True)
            
        start_date = df_raw.index.min(); end_date = df_raw.index.max()
        
        # [수정] ASR용 스텁 함수 호출
        df_merged = _merge_sentiment_data(df_raw, feature_ticker, start_date, end_date) 
        
        if df_merged.index.has_duplicates:
            df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
            
        if isinstance(df_merged.columns, pd.MultiIndex):
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            new_cols = {}
            for col in df_merged.columns:
                col_name = col[0] if isinstance(col, tuple) and len(col) > 0 else str(col)
                if col_name in required_cols and col_name not in new_cols:
                    new_cols[col_name] = df_merged[col]
                elif 'sentiment_score' in str(col) and 'sentiment_score' not in new_cols:
                    new_cols['sentiment_score'] = df_merged[col]
            df_merged = pd.DataFrame(new_cols)
            if 'Close' not in df_merged.columns and 'Adj Close' in df_merged.columns:
                 df_merged['Close'] = df_merged['Adj Close']
            if 'High' not in df_merged.columns or 'Low' not in df_merged.columns or 'Close' not in df_merged.columns:
                 return {**empty_return, "status": "데이터 컬럼 구성 실패"}

        if 'Date' not in df_merged.columns:
            df_merged.reset_index(inplace=True)

        df_full = calculate_technical_indicators(df_merged.copy())

        if df_full is None:
            return {**empty_return, "status": "기술 지표 계산 실패"}
        
        features_to_use = [f for f in top_3_features if f in df_full.columns]
        sentiment_available = False
        
        # (ASR에서는 _merge_sentiment_data가 항상 0을 반환하므로 sentiment_available은 False가 됨)
        if 'sentiment_score' in df_full.columns and df_full['sentiment_score'].sum() != 0:
            features_to_use.append('sentiment_score')
            sentiment_available = True

        features_to_use = [f for f in features_to_use if f in df_full.columns]

        if not features_to_use or len(df_full.dropna(subset=features_to_use)) < 2:
            return {**empty_return, "status": "최종 피처 데이터 부족"}

        df_model_ready = df_full.dropna(subset=features_to_use).copy()

        # 6. 방향 예측 (Classification)
        df_cls = df_model_ready.copy()
        df_cls['Target'] = (df_cls['Close'].shift(-1) > df_cls['Close']).astype(int)
        df_cls = df_cls.dropna(subset=['Target'])

        if len(df_cls) < 2:
             return {**empty_return, "status": "분류 모델 학습 데이터 부족"}

        X_cls = df_cls[features_to_use]
        y_cls = df_cls['Target']
        cls_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
        cls_model.fit(X_cls, y_cls)
        latest_data_features = df_model_ready[features_to_use].iloc[-1].values.reshape(1, -1)
        direction_prediction = cls_model.predict(latest_data_features)[0]

        # 7. 가격 예측 (Regression)
        df_reg = df_model_ready.copy()
        df_reg['Target_Price'] = df_reg['Close'].shift(-1)
        df_reg = df_reg.dropna(subset=['Target_Price'])

        if len(df_reg) < 2:
            return {**empty_return, "status": "회귀 모델 학습 데이터 부족"}

        X_reg = df_reg[features_to_use]
        y_reg = df_reg['Target_Price']
        reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
        reg_model.fit(X_reg, y_reg)
        predicted_price = reg_model.predict(latest_data_features)[0]

        # 8. 비현실적 가격 예측 방지
        current_price = df_model_ready['Close'].iloc[-1]
        price_upper_bound = current_price * 1.05
        price_lower_bound = current_price * 0.95
        realistic_price = min(max(predicted_price, price_lower_bound), price_upper_bound)

        model_type = "+Sent" if sentiment_available else "" # ASR에선 "+Sent"가 표시되지 않음
        status_message = f"예측(RF Top 3{model_type}) 성공"
        
        print(f"[{ticker}] 예측 성공: {'상승' if direction_prediction == 1 else '하락'} / {realistic_price:.2f} (현재가: {current_price:.2f})")

        return {
            "direction": int(direction_prediction),
            "price": realistic_price,
            "current_price": current_price,
            "features": features_to_use,
            "status": status_message
        }

    except Exception as e:
        print(f"[{ticker}] predict_stock 실행 중 심각한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {**empty_return, "status": f"서버 오류: {e}"}