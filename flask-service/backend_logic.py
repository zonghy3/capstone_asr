# =============================================================================
# 1. 통합 Import (모든 파일에서 사용되는 모듈)
# =============================================================================
import os
import json
import time
import re
import hashlib
import math
import copy
import concurrent.futures
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- 서드파티 라이브러리 ---
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import mysql.connector
import openai
from openai import OpenAI
from weasyprint import HTML

# 금융/모델링 관련
import yfinance as yf
import FinanceDataReader as fdr
from pykrx import stock
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler

# ML / DL
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from db_utils import ( # db_utils에서 필요한 함수들을 가져옵니다.
    get_db_connection, init_db,
    hash_password, add_user, check_user,
    save_history, get_history, get_history_detail, delete_history_record,
    save_chat_message, get_chat_history, delete_chat_history,
    save_news_articles, get_existing_links_by_source, get_articles_from_db,
    news_sources_exist, get_articles_since, get_daily_stock_sentiment_scores
)
from crawlers import search_domestic_news, search_overseas_news

# 기술적 분석
try:
    import pandas_ta as ta
    from pandas import DataFrame # pandas_ta 내부 오류 방지용
except ImportError:
    print("경고: 'pandas_ta' 라이브러리가 설치되지 않았습니다. 일부 모델(MA+MACD+RSI)이 작동하지 않을 수 있습니다.")
    ta = None

try:
    from app_helpers import get_stock_mapping
    stock_name_to_ticker_map_for_logic = get_stock_mapping()
    ticker_to_name_map_for_logic = {v: k for k, v in stock_name_to_ticker_map_for_logic.items()}
except ImportError:
    print("경고: app_helpers 모듈 로드 실패. 티커<->이름 변환 불가.")
    stock_name_to_ticker_map_for_logic = {}
    ticker_to_name_map_for_logic = {}
    
# =============================================================================
# 2. 외부 의존성 처리 (crawlers 스텁)

try:
    # crawlers.py가 같은 경로에 있다고 가정하고 import 시도
    from crawlers import search_domestic_news, search_overseas_news
    print(" -> 'crawlers' 모듈 로드 성공.")
except ImportError:
    print("경고: 'crawlers' 모듈을 찾을 수 없습니다. 'portfolio_optimizer'가 작동하지 않습니다.")
    print(" -> 임시 'search_domestic_news', 'search_overseas_news' 함수를 정의합니다.")
    
    def search_domestic_news(query, limit=50):
        """[임시 스텁] crawlers.py의 함수"""
        print(f"경고: 'search_domestic_news' (임시 스텁) 호출됨 (Query: {query})")
        return []

    def search_overseas_news(ticker, limit=50):
        """[임시 스텁] crawlers.py의 함수"""
        print(f"경고: 'search_overseas_news' (임시 스텁) 호출됨 (Ticker: {ticker})")
        return []

# =============================================================================
# 3. 파일 내용 (의존성 순서로 정렬)
# =============================================================================


print("모듈 로드: technical_analyzer.py")
def calculate_technical_indicators(df):
    """주어진 데이터프레임에 기술적 지표를 계산하여 추가합니다."""
    if df is None or df.empty:
        return None
    if 'Close' not in df.columns: # 필수 컬럼 확인
        print("오류: calculate_technical_indicators - 'Close' 컬럼 없음")
        return None

    try:
        # --- [신규] pandas_ta 사용 전 인덱스 설정 ---
        if 'Date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            print(" -> technical_analyzer: 'Date' 컬럼을 인덱스로 설정합니다.")
            df['Date'] = pd.to_datetime(df['Date']) # datetime 타입 확인
            df.set_index('Date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
             print("오류: calculate_technical_indicators - 유효한 날짜 인덱스가 없습니다.")
             return None # 날짜 인덱스 없으면 계산 불가
        # --- [신규] 끝 ---

        # pandas_ta 지표 계산 (이제 날짜 인덱스가 설정된 상태)
        df.ta.rsi(close='Close', length=14, append=True)
        df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        # CCI 추가
        df.ta.cci(close='Close', length=20, append=True)
        # Stochastic Oscillator (%K, %D) 추가
        df.ta.stoch(close='Close', k=14, d=3, smooth_k=3, append=True)
        # Bollinger Bands (%B) 추가
        df.ta.bbands(close='Close', length=20, append=True) # bbp()가 %B 컬럼(BBP_20_2.0) 추가함
        # Williams %R 추가
        df.ta.willr(close='Close', length=14, append=True)
        # 이동평균선 추가 (EMA)
        df.ta.ema(close='Close', length=20, append=True)
        df.ta.ema(close='Close', length=50, append=True)
        # 가격 변동률 추가
        df['Change'] = df['Close'].pct_change() * 100

        # 불필요한 보조 컬럼 제거 (예: bbands의 BBL_20_2.0, BBM_20_2.0 등)
        cols_to_drop = [col for col in df.columns if col.startswith(('BBL_', 'BBM_', 'BBU_', 'BBB_', 'STOCHk_', 'STOCHd_'))]
        if 'STOCHk_14_3_3' in df.columns: cols_to_drop.remove('STOCHk_14_3_3') # %K 유지
        if 'STOCHd_14_3_3' in df.columns: cols_to_drop.remove('STOCHd_14_3_3') # %D 유지
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # 컬럼 이름 간단하게 변경 (선택적)
        df.rename(columns={
            'RSI_14': 'RSI',
            'MACD_12_26_9': 'MACD',
            'MACDh_12_26_9': 'MACD_Hist',
            'MACDs_12_26_9': 'MACD_Signal',
            'CCI_20_0.015': 'CCI',
            'STOCHk_14_3_3': 'STOCHk',
            'STOCHd_14_3_3': 'STOCHd',
            'BBP_20_2.0': 'BB_Percent', # Bollinger %B 컬럼
            'WILLR_14': 'WilliamsR',
            'EMA_20': 'EMA20',
            'EMA_50': 'EMA50'
        }, inplace=True)

        return df
    except Exception as e:
        print(f"오류: calculate_technical_indicators 내부 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- [파일 3] model_analyzer.py ---
print("모듈 로드: model_analyzer.py")
def get_stock_data_for_analysis(ticker, years=3):
    """분석을 위한 데이터를 준비합니다."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    df_raw = fdr.DataReader(ticker, start_date, end_date)
    
    if df_raw.empty:
        return None, None

    # (의존성) calculate_technical_indicators 함수 호출
    df = calculate_technical_indicators(df_raw.copy())
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    features = [col for col in df.columns if col != 'Target']
    X = df[features]
    y = df['Target']
    
    return X, y

def analyze_feature_importance(ticker, top_n=3):
    """
    강화된 랜덤 포레스트 모델을 학습시키고,
    가장 중요한 상위 N개의 피처(정보) 이름을 반환합니다.
    """
    # (의존성) get_stock_data_for_analysis 함수 호출
    X, y = get_stock_data_for_analysis(ticker)
    
    if X is None:
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


# --- [파일 4] model_trainer_final.py ---
print("모듈 로드: model_trainer_final.py")
def get_stock_data(ticker, years=3):
    """[수정] KST 시간대를 현지화(localize)하고 대문자 컬럼으로 반환합니다."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    try:
        df = fdr.DataReader(ticker, start_date, end_date)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        present_cols = [col for col in required_cols if col in df.columns]
        df = df[present_cols]

        # [수정] fdr에서 받은 Naive KST 인덱스에 시간대 정보('Asia/Seoul') 부여
        df.index = pd.to_datetime(df.index).tz_localize('Asia/Seoul')

        df.index.name = 'Date'
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        return df
    except Exception as e:
        print(f"fdr 데이터 로드 오류 ({ticker}): {e}")
        return pd.DataFrame()

# --- LSTM 모델 관련 함수 ---
def create_sequences(data, target, time_steps=60):
    """LSTM 입력용 시퀀스 데이터를 생성합니다."""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

# 1. LSTM (기본)
def train_and_evaluate_lstm(ticker, years):
    """LSTM 모델 평가 함수 (기본 피처)."""
    df = get_stock_data(ticker, years);
    features_used = ['Open', 'High', 'Low', 'Close', 'Volume']
    if df.empty or 'Close' not in df.columns: return None, None, None

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    valid_features = [f for f in features_used if f in df.columns]
    if len(valid_features) != len(features_used):
         print(f"LSTM 경고: 필요한 피처 부족. 사용 가능: {valid_features}")
         return None, None, None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[valid_features])

    X, y = create_sequences(scaled_data, df['Target'].values)
    if X.shape[0] == 0:
        print(f"LSTM 오류: 시퀀스 데이터 생성 불가 (데이터 수: {len(df)})")
        return None, None, None

    tscv = TimeSeriesSplit(n_splits=5); scores, total_cm = [], np.zeros((2, 2))
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]; y_train, y_test = y[train_index], y[test_index]
        model = keras.models.Sequential([
            keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50), keras.layers.Dropout(0.2),
            keras.layers.Dense(25), keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
        scores.append({
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred, labels=[0, 1], zero_division=0),
            "Precision": precision_score(y_test, y_pred, labels=[0, 1], zero_division=0),
            "Recall": recall_score(y_test, y_pred, labels=[0, 1], zero_division=0)
        })
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.shape == (2, 2): total_cm += cm
    avg_metrics = pd.DataFrame(scores).mean().to_dict(); return avg_metrics, total_cm.astype(int), valid_features

# --- 트리 기반 모델 공통 학습/평가 함수 ---
def _train_tree_model(X, y, model_type='rf', n_estimators=100, max_depth=10, min_samples_leaf=5):
    """RF 모델 학습 및 평가 (하이퍼파라미터 인자 사용)."""
    tscv = TimeSeriesSplit(n_splits=5)
    scores, total_cm = [], np.zeros((2, 2))
    model = None
    if model_type != 'rf': raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1,
                                       max_depth=max_depth, min_samples_leaf=min_samples_leaf)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append({
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred, labels=[0, 1], zero_division=0),
            "Precision": precision_score(y_test, y_pred, labels=[0, 1], zero_division=0),
            "Recall": recall_score(y_test, y_pred, labels=[0, 1], zero_division=0)
        })
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.shape == (2, 2): total_cm += cm
    return pd.DataFrame(scores).mean().to_dict(), total_cm.astype(int), model

# --- 1-5번: 기존 모델 평가 함수들 ---

# 2. Baseline RF
def train_and_evaluate_rf_baseline(ticker, years):
    df = get_stock_data(ticker, years)
    if df.empty or 'Close' not in df.columns: return None, None, None
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    valid_features = [f for f in features if f in df.columns];
    if len(valid_features) < len(features): return None, None, None
    metrics, cm, _ = _train_tree_model(df[valid_features], df['Target'], model_type='rf')
    return metrics, cm, valid_features

# 3. Enhanced RF
def train_and_evaluate_rf_enhanced(ticker, years):
    df_raw = get_stock_data(ticker, years);
    if df_raw.empty or 'Close' not in df.columns: return None, None, None
    # (의존성) calculate_technical_indicators 함수 호출
    df = calculate_technical_indicators(df_raw.copy());
    if df is None or df.empty: return None, None, None
    if 'Close' not in df.columns: return None, None, None
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    features = [col for col in df.columns if col != 'Target'];
    if not features: return None, None, None
    metrics, cm, _ = _train_tree_model(df[features], df['Target'], model_type='rf')
    return metrics, cm, features

# 4. Top 3 RF
def train_and_evaluate_rf_top3(ticker, years):
    # (의존성) analyze_feature_importance 함수 호출
    top_3_features = analyze_feature_importance(ticker, top_n=3);
    if not top_3_features: return None, None, None
    df_raw = get_stock_data(ticker, years);
    if df_raw.empty or 'Close' not in df_raw.columns: return None, None, None
    # (의존성) calculate_technical_indicators 함수 호출
    df = calculate_technical_indicators(df_raw.copy());
    if df is None or df.empty: return None, None, None
    if 'Close' not in df.columns: return None, None, None
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    valid_features = [f for f in top_3_features if f in df.columns];
    if len(valid_features) != 3 or not valid_features: return None, None, None
    metrics, cm, _ = _train_tree_model(df[valid_features], df['Target'], model_type='rf')
    return metrics, cm, valid_features

# 5. RF (MA+MACD+RSI)
def train_and_evaluate_rf_MA_MACD_RSI(ticker, years):
    if ta is None: return None, None, None
    df = get_stock_data(ticker, years);
    if df.empty or 'Close' not in df.columns: return None, None, None
    try:
        df.ta.rsi(close='Close', length=14, append=True, col_names=('RSI_14'))
        df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'))
        df.ta.ema(close='Close', length=20, append=True, col_names=('EMA_20'))
        df.ta.ema(close='Close', length=50, append=True, col_names=('EMA_50'))
    except Exception as e: print(f"RF MA_MACD_RSI 지표 계산 실패: {e}"); return None, None, None
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    features = ['Close', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'EMA_20', 'EMA_50']
    if 'Volume' in df.columns: features.append('Volume')
    valid_features = [f for f in features if f in df.columns];
    if len(valid_features) < 4: return None, None, None
    metrics, cm, _ = _train_tree_model(df[valid_features], df['Target'], model_type='rf')
    return metrics, cm, valid_features

# --- [신규] 6-10번: 감성 점수 추가 모델 함수 ---

# 공통: 감성 점수 데이터 병합 함수
def _merge_sentiment_data(df_ohlcv, ticker, start_date, end_date):
    """[수정] '종목별' 일별 평균 감성 '점수 계산' 함수를 호출하여 병합 (시간대/인덱스/컬럼 레벨 최종 수정)"""
    if df_ohlcv.empty: return df_ohlcv

    try:
        if start_date.tzinfo is None: start_date_aware = start_date.tz_localize(timezone.utc)
        else: start_date_aware = start_date
        if end_date.tzinfo is None: end_date_aware = end_date.tz_localize(timezone.utc)
        else: end_date_aware = end_date
        start_dt_utc = start_date_aware.astimezone(timezone.utc)
        end_dt_utc = end_date_aware.astimezone(timezone.utc)
    except Exception as e:
        print(f"오류: _merge_sentiment_data 시간대 변환 실패: {e}")
        return pd.DataFrame()

    df_ohlcv_processed = df_ohlcv.copy()

    try:
        if isinstance(df_ohlcv_processed.index, pd.MultiIndex):
            df_ohlcv_processed.index = pd.to_datetime(df_ohlcv_processed.index.get_level_values(0))
        df_ohlcv_processed.index = pd.to_datetime(df_ohlcv_processed.index)
        if df_ohlcv_processed.index.tz is not None:
            df_ohlcv_processed.index = df_ohlcv_processed.index.tz_convert('UTC').tz_localize(None).normalize()
        else:
            df_ohlcv_processed.index = df_ohlcv_processed.index.tz_localize(None).normalize()
        df_ohlcv_processed.index.name = 'Date'
    except Exception as e:
        print(f"오류: df_ohlcv 인덱스 처리 실패: {e}")
        return pd.DataFrame()

    if isinstance(df_ohlcv_processed.columns, pd.MultiIndex):
        df_ohlcv_processed.columns = df_ohlcv_processed.columns.get_level_values(0)
        df_ohlcv_processed = df_ohlcv_processed.loc[:,~df_ohlcv_processed.columns.duplicated()]

    # (의존성) get_daily_stock_sentiment_scores 함수 호출
    df_sentiment = get_daily_stock_sentiment_scores(ticker, start_dt_utc, end_dt_utc)

    df_sentiment_processed = pd.DataFrame(columns=['sentiment_score'])
    df_sentiment_processed.index.name = 'Date'
    if not df_sentiment.empty:
        try:
            df_sentiment.index = pd.to_datetime(df_sentiment.index).tz_localize(None).normalize()
            df_sentiment.index.name = 'Date'
            if 'sentiment_score' in df_sentiment.columns:
                 df_sentiment_processed = df_sentiment[['sentiment_score']]
        except Exception as e:
             print(f"경고: df_sentiment 인덱스/컬럼 처리 실패: {e}.")
    else:
        df_sentiment_processed = pd.DataFrame(columns=['sentiment_score']).set_index(pd.to_datetime([]))
        df_sentiment_processed.index.name = 'Date'

    try:
        df_merged = df_ohlcv_processed.join(df_sentiment_processed, how='left')
        df_merged['sentiment_score'] = df_merged['sentiment_score'].ffill()
        df_merged['sentiment_score'] = df_merged['sentiment_score'].fillna(0)
        return df_merged
    except Exception as e:
        print(f"오류: 최종 데이터 병합(join) 실패: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    
# 6. LSTM + Sentiment
def train_and_evaluate_lstm_with_sentiment(ticker, years):
    """[신규] LSTM 모델 + 종목별 감성 점수 피처."""
    df_ohlcv = get_stock_data(ticker, years)
    if df_ohlcv.empty or 'Close' not in df_ohlcv.columns: return None, None, None
    start_date = df_ohlcv.index.min(); end_date = df_ohlcv.index.max()
    # (의존성) _merge_sentiment_data 함수 호출
    df = _merge_sentiment_data(df_ohlcv, ticker, start_date, end_date)
    if df.empty: return None, None, None

    features_used = ['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_score']
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    valid_features = [f for f in features_used if f in df.columns]
    if len(valid_features) < len(features_used): return None, None, None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[valid_features])

    X, y = create_sequences(scaled_data, df['Target'].values);
    if X.shape[0] == 0: return None, None, None

    tscv = TimeSeriesSplit(n_splits=5); scores, total_cm = [], np.zeros((2, 2))
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]; y_train, y_test = y[train_index], y[test_index]
        model = keras.models.Sequential([
            keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            keras.layers.Dropout(0.2), keras.layers.LSTM(50), keras.layers.Dropout(0.2),
            keras.layers.Dense(25), keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
        scores.append({
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred, labels=[0, 1], zero_division=0),
            "Precision": precision_score(y_test, y_pred, labels=[0, 1], zero_division=0),
            "Recall": recall_score(y_test, y_pred, labels=[0, 1], zero_division=0)
        })
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.shape == (2, 2): total_cm += cm
    avg_metrics = pd.DataFrame(scores).mean().to_dict(); return avg_metrics, total_cm.astype(int), valid_features

# 7. Baseline RF + Sentiment
def train_and_evaluate_rf_baseline_with_sentiment(ticker, years):
    """[신규] Baseline RF 모델 + 종목별 감성 점수 피처."""
    df_ohlcv = get_stock_data(ticker, years)
    if df_ohlcv.empty or 'Close' not in df_ohlcv.columns: return None, None, None
    start_date = df_ohlcv.index.min(); end_date = df_ohlcv.index.max()
    # (의존성) _merge_sentiment_data 함수 호출
    df = _merge_sentiment_data(df_ohlcv, ticker, start_date, end_date)
    if df.empty: return None, None, None

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_score']
    valid_features = [f for f in features if f in df.columns];
    if len(valid_features) < len(features): return None, None, None
    metrics, cm, _ = _train_tree_model(df[valid_features], df['Target'], model_type='rf')
    return metrics, cm, valid_features

# 8. Enhanced RF + Sentiment
def train_and_evaluate_rf_enhanced_with_sentiment(ticker, years):
    """[신규] Enhanced RF 모델 + 종목별 감성 점수 피처."""
    df_raw = get_stock_data(ticker, years);
    if df_raw.empty or 'Close' not in df_raw.columns: return None, None, None
    start_date = df_raw.index.min(); end_date = df_raw.index.max()
    # (의존성) _merge_sentiment_data 함수 호출
    df_merged = _merge_sentiment_data(df_raw, ticker, start_date, end_date)
    if df_merged.empty: return None, None, None

    # (의존성) calculate_technical_indicators 함수 호출
    df = calculate_technical_indicators(df_merged.copy());
    if df is None or df.empty: return None, None, None
    if 'Close' not in df.columns: return None, None, None
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    features = [col for col in df.columns if col != 'Target'];
    if not features: return None, None, None
    metrics, cm, _ = _train_tree_model(df[features], df['Target'], model_type='rf')
    return metrics, cm, features

# 9. Top 3 RF + Sentiment
def train_and_evaluate_rf_top3_with_sentiment(ticker, years):
    """[신규] Top 3 RF 모델 + 종목별 감성 점수 피처."""
    # (의존성) analyze_feature_importance 함수 호출
    top_3_features = analyze_feature_importance(ticker, top_n=3);
    if not top_3_features: return None, None, None
    df_raw = get_stock_data(ticker, years);
    if df_raw.empty or 'Close' not in df_raw.columns: return None, None, None
    start_date = df_raw.index.min(); end_date = df_raw.index.max()
    # (의존성) _merge_sentiment_data 함수 호출
    df_merged = _merge_sentiment_data(df_raw, ticker, start_date, end_date)
    if df_merged.empty: return None, None, None

    # (의존성) calculate_technical_indicators 함수 호출
    df = calculate_technical_indicators(df_merged.copy());
    if df is None or df.empty: return None, None, None
    if 'Close' not in df.columns: return None, None, None
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    features_to_use = [f for f in top_3_features if f in df.columns]
    if 'sentiment_score' in df.columns and df['sentiment_score'].sum() != 0:
        if 'sentiment_score' not in features_to_use:
             features_to_use.append('sentiment_score')

    valid_features = [f for f in features_to_use if f in df.columns];
    if len(valid_features) < 1: return None, None, None
    metrics, cm, _ = _train_tree_model(df[valid_features], df['Target'], model_type='rf')
    return metrics, cm, valid_features

# 10. RF (MA+MACD+RSI) + Sentiment
def train_and_evaluate_rf_MA_MACD_RSI_with_sentiment(ticker, years):
    """[신규] MA+MACD+RSI 모델 + 종목별 감성 점수 피처."""
    if ta is None: return None, None, None
    df_ohlcv = get_stock_data(ticker, years);
    if df_ohlcv.empty or 'Close' not in df_ohlcv.columns: return None, None, None
    start_date = df_ohlcv.index.min(); end_date = df_ohlcv.index.max()
    # (의존성) _merge_sentiment_data 함수 호출
    df = _merge_sentiment_data(df_ohlcv, ticker, start_date, end_date)
    if df.empty: return None, None, None

    try:
        df.ta.rsi(close='Close', length=14, append=True, col_names=('RSI_14'))
        df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'))
        df.ta.ema(close='Close', length=20, append=True, col_names=('EMA_20'))
        df.ta.ema(close='Close', length=50, append=True, col_names=('EMA_50'))
    except Exception as e: return None, None, None
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    features = ['Close', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'EMA_20', 'EMA_50']
    if 'sentiment_score' in df.columns and df['sentiment_score'].sum() != 0:
         features.append('sentiment_score')
    if 'Volume' in df.columns: features.append('Volume')

    valid_features = [f for f in features if f in df.columns];
    if len(valid_features) < 5: return None, None, None
    metrics, cm, _ = _train_tree_model(df[valid_features], df['Target'], model_type='rf')
    return metrics, cm, valid_features


# --- [파일 5] final_model.py ---
print("모듈 로드: final_model.py")
def predict_stock(ticker: str, years: int = 1, data_df: pd.DataFrame = None) -> dict:
    """
    [수정됨] data_df가 주어지면 해당 데이터로 예측, 없으면 최신 데이터 로드.
    Plan A/B 자동 전환 로직 포함.
    """

    empty_return = {"direction": 0, "price": 0, "current_price": 0, "features": []}
    fdr_ticker = ticker.split('.')[0] if isinstance(ticker, str) else ticker
    print(f"[{fdr_ticker}] 예측 시작 (데이터: {'제공됨' if data_df is not None else f'최신 {years}년'})...")

    try:
        # 1. Top 3 피처 선택 (fdr_ticker 사용)
        # (의존성) analyze_feature_importance 함수 호출 (model_analyzer.py)
        top_3_features = analyze_feature_importance(fdr_ticker, top_n=3)
        if not top_3_features:
            print(f"[{fdr_ticker}] Top 3 피처 선택 실패.")
            return empty_return

        # 2. 데이터 준비 (fdr_ticker 사용)
        if data_df is not None and not data_df.empty:
            df_raw = data_df.copy()
            print(f"[{fdr_ticker}] 제공된 데이터 사용 ({len(df_raw)} 행).")
            if len(df_raw) < 50:
                 print(f"[{fdr_ticker}] 제공된 데이터 기간 부족 ({len(df_raw)} 행). 예측 불가.")
                 return empty_return
        else:
            print(f"[{fdr_ticker}] 최신 {years}년 데이터 로드 중...")
            # (의존성) get_stock_data 함수 호출 (model_trainer_final.py)
            df_raw = get_stock_data(fdr_ticker, years) # fdr_ticker 사용
            if df_raw.empty or 'Close' not in df_raw.columns or len(df_raw) < 50:
                print(f"[{fdr_ticker}] KST 주가 데이터 로드 실패 또는 부족.")
                return empty_return

        # 3. 감성 점수 데이터 병합 시도 (fdr_ticker 사용)
        start_date = df_raw.index.min(); end_date = df_raw.index.max()
        # (의존성) _merge_sentiment_data 함수 호출 (model_trainer_final.py)
        df_merged = _merge_sentiment_data(df_raw, fdr_ticker, start_date, end_date)
        
        if df_merged.index.has_duplicates:
            print(f"경고: [{fdr_ticker}] 데이터 인덱스에 중복된 값이 있습니다. 마지막 값만 유지합니다.")
            df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
            

        # 4. 기술적 지표 계산
        if 'Date' not in df_merged.columns:
            df_merged.reset_index(inplace=True)

        # (의존성) calculate_technical_indicators 함수 호출 (technical_analyzer.py)
        df_full = calculate_technical_indicators(df_merged.copy())

        if df_full is None:
            print(f"[{fdr_ticker}] 기술 지표 계산 실패.")
            return empty_return
        
        # 5. 폴백 로직 및 최종 피처 확정
        features_to_use = [f for f in top_3_features if f in df_full.columns]
        sentiment_available = False
        if 'sentiment_score' in df_full.columns and df_full['sentiment_score'].sum() != 0:
            features_to_use.append('sentiment_score')
            sentiment_available = True

        features_to_use = [f for f in features_to_use if f in df_full.columns]

        if not features_to_use or len(df_full.dropna(subset=features_to_use)) < 2:
            print(f"[{ticker}] 최종 피처 구성 실패 또는 유효 데이터 부족.")
            return empty_return

        df_model_ready = df_full.dropna(subset=features_to_use).copy()

        # 6. 방향 예측 (Classification) 모델 학습 및 예측
        df_cls = df_model_ready.copy()
        df_cls['Target'] = (df_cls['Close'].shift(-1) > df_cls['Close']).astype(int)
        df_cls = df_cls.dropna(subset=['Target'])

        if len(df_cls) < 2:
             print(f"[{ticker}] 분류 모델 학습 데이터 부족.")
             return empty_return

        X_cls = df_cls[features_to_use]
        y_cls = df_cls['Target']

        cls_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
        cls_model.fit(X_cls, y_cls)

        latest_data_features = df_model_ready[features_to_use].iloc[-1].values.reshape(1, -1)
        direction_prediction = cls_model.predict(latest_data_features)[0]

        # 7. 가격 예측 (Regression) 모델 학습 및 예측
        df_reg = df_model_ready.copy()
        df_reg['Target_Price'] = df_reg['Close'].shift(-1)
        df_reg = df_reg.dropna(subset=['Target_Price'])

        if len(df_reg) < 2:
            print(f"[{ticker}] 회귀 모델 학습 데이터 부족.")
            return empty_return

        X_reg = df_reg[features_to_use]
        y_reg = df_reg['Target_Price']

        reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
        reg_model.fit(X_reg, y_reg)

        predicted_price = reg_model.predict(latest_data_features)[0]

        # 8. 비현실적 가격 예측 방지 (안전장치) - [수정] +/- 5% 적용
        current_price = df_model_ready['Close'].iloc[-1]
        price_upper_bound = current_price * 1.05
        price_lower_bound = current_price * 0.95
        realistic_price = min(max(predicted_price, price_lower_bound), price_upper_bound)

        model_type = "+Sent" if sentiment_available else ""
        print(f"[{ticker} {df_model_ready.index[-1].date()}] 예측(RF Top 3{model_type}): {'상승' if direction_prediction == 1 else '하락'} / {realistic_price:.2f} (현재가: {current_price:.2f})")

        return {
            "direction": int(direction_prediction),
            "price": realistic_price,
            "current_price": current_price,
            "features": features_to_use
        }

    except Exception as e:
        print(f"[{ticker}] predict_stock 실행 중 심각한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return empty_return


# --- [파일 6] news_analyzer.py ---
print("모듈 로드: news_analyzer.py")
# --- OpenAI 클라이언트 초기화 ---
load_dotenv()
try:
    if not os.environ.get("OPENAI_API_KEY"): raise ValueError("...")
    news_analyzer_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    print("OpenAI 클라이언트 (번역용) 설정 완료.")
except Exception as e:
    print(f"오류: OpenAI 클라이언트 초기화 실패: {e}"); news_analyzer_client = None

# --- EN-FinBERT 모델 로드 ---
def load_en_finbert_model():
    try:
        model_name = "ProsusAI/finbert"; tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("EN-FinBERT 모델 로드 완료."); return tokenizer, model
    except Exception as e: print(f"오류: EN-FinBERT 모델 로딩 실패: {e}"); return None, None

# --- OpenAI 배치 번역 함수 (변경 없음) ---
def translate_headlines_batch_openai(headlines_ko: list[str]) -> list[str | None]:
    if not news_analyzer_client: print("OpenAI 클라이언트 없음"); return [None] * len(headlines_ko)
    numbered_headlines = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines_ko)])
    system_prompt = "You are a helpful assistant who translates Korean news headlines into English."
    user_prompt = f"""Translate the following numbered Korean headlines into English.
Return the results ONLY as a JSON object containing a single key "translations" which holds a list of strings. Each string in the list must be the English translation corresponding to the numbered headline.
Example JSON structure: {{"translations": ["translation 1", "translation 2", ...]}}

Korean Headlines:
{numbered_headlines}

JSON Result:"""
    try:
        response = news_analyzer_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=len(headlines_ko) * 70, temperature=0.0, response_format={"type": "json_object"})
        result_text = response.choices[0].message.content.strip(); result_json = json.loads(result_text)
        translations = result_json.get("translations")
        if isinstance(translations, list) and len(translations) == len(headlines_ko):
            cleaned = [t.strip().strip('"').strip("'") if isinstance(t, str) else None for t in translations]
            print(f"배치 번역 성공 ({len(headlines_ko)}개)"); return cleaned
        else: print(f"배치 번역 결과 형식/개수 오류..."); return [None] * len(headlines_ko)
    except Exception as e: print(f"OpenAI 배치 번역 오류: {e}"); return [None] * len(headlines_ko)

# --- 개별 언어 감지 함수 (변경 없음) ---
def get_headline_language(headline: str) -> str:
    if not headline: return 'en'
    if re.search("[ㄱ-ㅎㅏ-ㅣ가-힣]", headline): return 'ko'
    return 'en'

# --- EN-FinBERT 예측 함수 (변경 없음) ---
def predict_sentiment_en_finbert(headlines: list):
    tokenizer, model = load_en_finbert_model()
    if not tokenizer or not model: print("경고: EN-FinBERT 모델 없음"); return [{'sentiment': 'neutral', 'confidence': 0.5}] * len(headlines)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    results = []
    try:
        inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad(): outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1); preds = torch.argmax(probs, dim=1)
        for i in range(len(headlines)): pred_id = preds[i].item(); results.append({'sentiment': id2label[pred_id], 'confidence': probs[i][pred_id].item()})
        print(f"EN-FinBERT 예측 완료 ({len(headlines)}개)")
    except Exception as e: print(f"오류: EN-FinBERT 예측 오류: {e}"); results = [{'sentiment': 'neutral', 'confidence': 0.5}] * len(headlines)
    return results

# --- (내부용) 영어 뉴스 종합 분석 함수 ---
def _analyze_sentiment_english(news_items: list, name: str):
    """[수정] 분석 결과에 sentiment 레이블만 포함"""
    if not news_items: return {"stock_name": name, "sentiment_score": 0, "market_outlook": "분석할 영어 뉴스 없음", "investment_strategy": "정보 부족", "summary": {"total": 0}, "analyzed_articles": []}
    headlines = [item['headline'] for item in news_items]
    analysis_results = predict_sentiment_en_finbert(headlines)
    analyzed_articles = []
    gamma_scores = []
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    for i in range(len(news_items)):
        original_item = news_items[i]
        res = analysis_results[i]
        senti = res.get('sentiment', 'neutral').lower()
        conf = res.get('confidence', 0.5)

        analysis_result_item = {
            **original_item,
            'headline': original_item.get('headline'),
            'link': original_item.get('link', '#'),
            'sentiment': senti,
            'confidence': conf
        }
        analyzed_articles.append(analysis_result_item)

        if senti == 'positive': gamma_scores.append(1); counts['positive'] += 1
        elif senti == 'negative': gamma_scores.append(-1); counts['negative'] += 1
        else: gamma_scores.append(0); counts['neutral'] += 1

    score = sum(gamma_scores) / len(gamma_scores) if gamma_scores else 0
    outlook, strategy = "", "중립"
    if score > 0.1: outlook = f"{name} 관련 (영어/번역) 뉴스 감성 긍정적."; strategy = "적극 투자"
    elif score < -0.1: outlook = f"{name} 관련 (영어/번역) 뉴스 감성 부정적."; strategy = "보수적/방어적 투자"
    else: outlook = f"{name} 관련 (영어/번역) 뉴스 감성 중립적."; strategy = "중립"

    return {"stock_name": name, "sentiment_score": score, "market_outlook": outlook, "investment_strategy": strategy, "summary": {"total": len(news_items), **counts},
            "analyzed_articles": analyzed_articles}

# --- [수정] 자동 감지 및 분석 라우터 (_analyze_sentiments_auto_detect) ---
def _analyze_sentiments_auto_detect(name: str, news_items: list):
    """[수정] 뉴스 언어 감지, 번역, 분석 후 **sentiment 레이블**만 결과에 포함."""
    if not news_items: return {"stock_name": name, "sentiment_score": 0, "market_outlook": "분석할 뉴스 없음", "investment_strategy": "정보 부족", "summary": {"total": 0}, "analyzed_articles": []}

    final_analyzed_articles = [None] * len(news_items)
    all_gamma_scores = []
    translation_failed_count = 0
    analysis_failed_count = 0

    korean_indices = []
    english_indices = []
    items_for_analysis = []
    original_indices_map = {}

    print(f"총 {len(news_items)}개 뉴스 언어 감지 및 분리 시작 ({name})...")
    for idx, item in enumerate(news_items):
        lang = get_headline_language(item['headline'])
        if lang == 'ko': korean_indices.append(idx)
        else:
            english_indices.append(idx)
            items_for_analysis.append(item)
            original_indices_map[len(items_for_analysis) - 1] = idx
    print(f"  -> 한국어 {len(korean_indices)}개 / 영어 {len(english_indices)}개 분리 완료.")

    if korean_indices:
        print(f"한국어 뉴스 {len(korean_indices)}개 배치 번역 시작...")
        BATCH_SIZE = 20
        num_ko_batches = math.ceil(len(korean_indices) / BATCH_SIZE)
        for i in range(num_ko_batches):
            batch_start_ko_idx = i * BATCH_SIZE
            batch_end_ko_idx = batch_start_ko_idx + BATCH_SIZE
            current_batch_original_indices = korean_indices[batch_start_ko_idx:batch_end_ko_idx]
            current_batch_headlines_ko = [news_items[idx]['headline'] for idx in current_batch_original_indices]

            print(f"  - 배치 {i+1}/{num_ko_batches} 번역 요청 ({len(current_batch_headlines_ko)}개)...")
            batch_translations = translate_headlines_batch_openai(current_batch_headlines_ko)

            for j, translation in enumerate(batch_translations):
                original_idx = current_batch_original_indices[j]
                if translation:
                    analysis_item = copy.deepcopy(news_items[original_idx])
                    analysis_item['headline'] = translation
                    items_for_analysis.append(analysis_item)
                    original_indices_map[len(items_for_analysis) - 1] = original_idx
                else:
                    translation_failed_count += 1
                    failed_item = copy.deepcopy(news_items[original_idx])
                    failed_item['sentiment'] = 'neutral'
                    failed_item['confidence'] = 0.0
                    final_analyzed_articles[original_idx] = failed_item
                    all_gamma_scores.append(0)

    analysis_results_list = []
    if items_for_analysis:
        print(f"총 {len(items_for_analysis)}개 뉴스(영어+번역) 영어 모델 분석 시작...")
        analysis_output = _analyze_sentiment_english(items_for_analysis, name)
        analysis_results_list = analysis_output.get("analyzed_articles", [])
        print("영어 모델 분석 완료.")
        if len(analysis_results_list) != len(items_for_analysis):
            analysis_failed_count = abs(len(items_for_analysis) - len(analysis_results_list))
            print(f"경고: 분석 결과 개수 불일치 ({len(analysis_results_list)} != {len(items_for_analysis)})!")

    final_summary = {"total": len(news_items), "positive": 0, "negative": 0, "neutral": 0}
    processed_indices = set()

    for analysis_idx, analyzed_item in enumerate(analysis_results_list):
        if analysis_idx in original_indices_map:
            original_idx = original_indices_map[analysis_idx]
            final_analyzed_articles[original_idx] = analyzed_item
            processed_indices.add(original_idx)

            senti = analyzed_item.get('sentiment', 'neutral')
            if senti == 'positive': all_gamma_scores.append(1); final_summary['positive'] += 1
            elif senti == 'negative': all_gamma_scores.append(-1); final_summary['negative'] += 1
            else: all_gamma_scores.append(0); final_summary['neutral'] += 1
        else:
             print(f"경고: 분석 인덱스 {analysis_idx} 매핑 정보 없음!")
             analysis_failed_count += 1

    final_summary['neutral'] += translation_failed_count

    missing_items = 0
    for idx in range(len(news_items)):
        if final_analyzed_articles[idx] is None:
            print(f"경고: 뉴스 인덱스 {idx} 최종 결과 누락! 중립 처리.")
            missing_item = copy.deepcopy(news_items[idx])
            missing_item['sentiment'] = 'neutral'
            missing_item['confidence'] = 0.0
            final_analyzed_articles[idx] = missing_item
            all_gamma_scores.append(0)
            final_summary['neutral'] += 1
            missing_items += 1
    analysis_failed_count += missing_items

    final_score = sum(all_gamma_scores) / len(all_gamma_scores) if all_gamma_scores else 0.0
    final_outlook, final_strategy = "", "중립"
    if final_score > 0.1: final_outlook = f"{name} 관련 뉴스(영어/번역)의 전반적인 감성이 긍정적입니다."; final_strategy = "적극 투자"
    elif final_score < -0.1: final_outlook = f"{name} 관련 뉴스(영어/번역)의 전반적인 감성이 부정적입니다."; final_strategy = "보수적/방어적 투자"
    else: final_outlook = f"{name} 관련 뉴스(영어/번역)의 전반적인 감성은 중립적입니다."; final_strategy = "중립"

    status_messages = []
    if translation_failed_count > 0: status_messages.append(f"{translation_failed_count}개 번역 실패")
    if analysis_failed_count > 0: status_messages.append(f"{analysis_failed_count}개 분석 오류")
    if status_messages: final_outlook += f" ({', '.join(status_messages)})"

    return {
        "stock_name": name, "sentiment_score": final_score,
        "market_outlook": final_outlook, "investment_strategy": final_strategy,
        "summary": final_summary,
        "analyzed_articles": final_analyzed_articles
    }

# --- 외부 호출용 함수들 (변경 없음) ---
def analyze_market_sentiment_from_news(news_items: list, stock_name: str) -> dict: return _analyze_sentiments_auto_detect(stock_name, news_items)
def analyze_news_sentiments_and_recommend(name, news_items): return _analyze_sentiments_auto_detect(name, news_items)
def analyze_news_sentiments_english(name, news_items): return _analyze_sentiments_auto_detect(name, news_items)


# --- [파일 7] chatbot_service.py ---
print("모듈 로드: chatbot_service.py")
load_dotenv()

# API 키 설정
chatbot_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 시스템 프롬프트 (AI의 역할 정의)
SYSTEM_PROMPT = """
당신은 금융 및 주식 시장 전문 AI 어시스턴트입니다.
당신의 임무는 사용자에게 경제와 주식 관련 주제에 대해서만 답변하는 것입니다.
모든 답변은 가능한 최신 정보에 기반해야 하며, 정보의 출처를 명확히 밝혀야 합니다.
답변은 항상 한국어로 제공하고, 한 번의 답변은 최대 1000 토큰을 넘지 않도록 간결하게 요약해주세요.
경제나 주식과 관련 없는 질문에는 "저는 금융 및 주식 전문 AI이므로, 해당 주제에 대해서만 답변할 수 있습니다."라고 정중히 거절해주세요.
"""

def get_chatbot_response(messages):
    """사용자의 메시지를 받아 OpenAI API를 통해 답변을 생성합니다."""
    try:
        # 시스템 프롬프트를 대화 시작에 추가
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        response = chatbot_client.chat.completions.create(
            model="gpt-4.1-nano",  # 또는 사용 가능한 최신 모델
            messages=conversation,
            max_tokens=1000,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API 호출 중 오류가 발생했습니다: {e}"


# --- [파일 8] report_generator.py ---
print("모듈 로드: report_generator.py")
def create_report_html(analysis_result: dict, news_items: list) -> str:
    """
    상세 감성 분석 결과와 시장 전망을 바탕으로 PDF로 변환될 HTML 리포트 문자열을 생성합니다.
    """
    stock_name = analysis_result.get('stock_name', 'N/A')
    summary = analysis_result.get('summary', {})
    sentiment_score = analysis_result.get('sentiment_score', 0.0)
    market_outlook = analysis_result.get('market_outlook', '분석 오류가 발생했습니다.')
    investment_strategy = analysis_result.get('investment_strategy', '중립')
    analyzed_articles = analysis_result.get('analyzed_articles', [])

    strategy_class = {
        "적극 투자": "positive",
        "보수적/방어적 투자": "negative"
    }.get(investment_strategy, "neutral")

    news_list_html = """
    <table class="news-table">
        <thead>
            <tr>
                <th>뉴스 헤드라인</th>
                <th>감성</th>
                <th>AI 분석 근거</th>
            </tr>
        </thead>
        <tbody>
    """
    if not analyzed_articles:
        news_list_html += '<tr><td colspan="3" style="text-align:center;">분석할 뉴스가 없습니다.</td></tr>'
    else:
        for article in analyzed_articles:
            headline = article.get('headline', 'No Title')
            link = article.get('link', '#')
            sentiment = article.get('sentiment', 'neutral')
            confidence = article.get('confidence', 0.0)
            
            sentiment_display = sentiment.capitalize()
            sentiment_class = sentiment
            confidence_percent = f"{confidence * 100:.1f}%"
            analysis_basis = f"AI가 이 뉴스를 '{sentiment_display}'(으)로 판단했습니다. (신뢰도: {confidence_percent})"

            news_list_html += f"""
            <tr>
                <td><a href="{link}" target="_blank">{headline}</a></td>
                <td class="sentiment-{sentiment_class}"><strong>{sentiment_display}</strong></td>
                <td>{analysis_basis}</td>
            </tr>
            """
    news_list_html += "</tbody></table>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
            body {{ font-family: 'Noto Sans KR', sans-serif; margin: 40px; background-color: #f9f9f9; color: #333; }}
            .report-container {{ background-color: white; border: 1px solid #e0e0e0; padding: 40px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
            h1 {{ color: #1a237e; font-size: 32px; border-bottom: 3px solid #3949ab; padding-bottom: 15px; margin-bottom: 10px; text-align: center; }}
            h2 {{ color: #283593; font-size: 24px; border-bottom: 1px solid #9fa8da; padding-bottom: 10px; margin-top: 40px; }}
            h3 {{ color: #3f51b5; font-size: 18px; margin-top: 30px; }}
            p {{ line-height: 1.8; font-size: 16px; }}
            .info-box {{ background-color: #e8eaf6; border-left: 5px solid #3f51b5; padding: 20px; margin: 20px 0; border-radius: 8px; font-size: 15px; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; text-align: center; margin-top: 20px; }}
            .grid-item {{ background-color: #f1f3f5; padding: 20px; border-radius: 8px; }}
            .grid-item .label {{ font-weight: bold; font-size: 16px; color: #555; }}
            .grid-item .value {{ font-size: 28px; font-weight: bold; margin-top: 10px; }}
            .score .value {{ color: #3f51b5; }}
            .positive .value, .positive span, .sentiment-positive {{ color: #2e7d32 !important; }}
            .negative .value, .negative span, .sentiment-negative {{ color: #c62828 !important; }}
            .neutral .value, .neutral span, .sentiment-neutral {{ color: #546e7a !important; }}
            .opinion-box {{ border: 1px solid #ddd; padding: 25px; margin-top: 15px; border-radius: 8px; background-color: #fafafa; }}
            
            .news-table-box {{ border: 1px solid #ddd; padding: 25px; margin-top: 15px; border-radius: 8px; background-color: #fafafa; }}
            .news-table {{ width: 100%; border-collapse: collapse; }}
            .news-table th, .news-table td {{ border-bottom: 1px solid #e0e0e0; padding: 12px 10px; text-align: left; vertical-align: middle; }}
            .news-table thead {{ background-color: #f8f9fa; }}
            .news-table th {{ font-weight: 700; color: #333; }}
            .news-table tr:last-child td {{ border-bottom: none; }}
            .news-table td a {{ color: #303f9f; text-decoration: none; font-weight: 500; }}
            .news-table td a:hover {{ text-decoration: underline; }}

            strong {{ font-weight: 700; }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <h1>AI 주식 분석 리포트: {stock_name}</h1>
            <div class="info-box">
                <strong>분석 모델 정보:</strong> 본 리포트는 KR-FinBERT를 활용한 뉴스 감성 분석 기반으로 생성되었습니다. 특정 기간의 KOSPI 관련 뉴스 헤드라인을 종합 분석하여 시장의 전반적인 감성 지수(Sentiment Score)를 도출하고, 이에 따라 위험 자산의 투자 비중 조정을 제안합니다.
            </div>

            <h2>I. 시장 감성 분석 종합 (KOSPI 뉴스 기반)</h2>
            <div class="summary-box">
                <p>총 <strong>{summary.get('total', 0)}개</strong>의 최신 뉴스를 분석한 결과입니다.</p>
                <div class="summary-grid">
                    <div class="grid-item score"><div class="label">시장 감성 지수 (Sₜ)</div><div class="value">{sentiment_score:.4f}</div></div>
                    <div class="grid-item positive"><div class="label">긍정 뉴스</div><div class="value">{summary.get('positive', 0)}개</div></div>
                    <div class="grid-item negative"><div class="label">부정 뉴스</div><div class="value">{summary.get('negative', 0)}개</div></div>
                    <div class="grid-item neutral"><div class="label">중립 뉴스</div><div class="value">{summary.get('neutral', 0)}개</div></div>
                </div>
            </div>

            <h2>II. AI 투자 의견 및 전략</h2>
            <div class="opinion-box">
                <h3>시장 전망 및 투자 전략: <span class="{strategy_class}">{investment_strategy}</span></h3>
                <p>{market_outlook}</p>
            </div>

            <h2>III. 분석에 사용된 뉴스 목록</h2>
            <div class="news-table-box">
                {news_list_html}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def create_report_pdf(html_content: str, ticker: str, name: str) -> Path:
    """
    HTML 문자열을 PDF 파일로 변환하여 저장합니다.
    """
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace(' ', '_').replace('/', '_')
    pdf_filename = f"Report_{ticker}_{safe_name}_{timestamp}.pdf"
    pdf_path = reports_dir / pdf_filename

    HTML(string=html_content).write_pdf(pdf_path)
    
    return pdf_path


# --- [파일 9] portfolio_optimizer.py ---
print("모듈 로드: portfolio_optimizer.py")
def get_exchange_rate():
    """실시간 USD/KRW 환율 정보를 가져옵니다. (수정된 파싱 로직)"""
    url = "https://m.search.naver.com/p/csearch/content/qapirender.nhn?key=calculator&pkid=141&q=%ED%99%98%EC%9C%A8&where=m&u1=keb&u6=standardUnit&u7=0&u3=USD&u4=KRW&u8=down&u2=1"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if (data and isinstance(data, dict) and 
                'country' in data and isinstance(data['country'], list) and 
                len(data['country']) > 1 and 'value' in data['country'][1]):
                
                rate_str = data['country'][1]['value'] # 예: "1,439.80"
                rate_str = rate_str.replace(',', '')   # 쉼표 제거
                return float(rate_str)
            
            else:
                print("환율 정보 파싱 실패: 예상한 JSON 구조가 아닙니다.")
                print("받은 데이터:", data) # 디버깅을 위해 실제 받은 데이터 출력

        else:
            print(f"환율 정보 조회 실패: HTTP 상태 코드 {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"환율 정보 조회 실패 (네트워크 오류): {e}")
    except json.JSONDecodeError as e:
        print(f"환율 정보 파싱 실패 (JSON 오류): {e}")
    except Exception as e:
        print(f"환율 정보 처리 중 알 수 없는 오류: {e}")
    
    # 실패 시 기본값 반환
    print("기본값 1400.0을 반환합니다.")
    return 1400.0

def get_stock_data_for_portfolio(tickers, period="3y"): # 함수 이름 변경 (model_trainer_final과 충돌 방지)
    """주가 데이터를 가져오고, 각 종목의 통화 정보를 포함하여 반환합니다."""
    korean_tickers = [t for t in tickers if t.isdigit() and len(t) == 6]
    overseas_tickers = [t for t in tickers if not (t.isdigit() and len(t) == 6)]
    
    all_data_frames = []
    errors = []
    currency_info = {}

    if korean_tickers:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1095)
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        for ticker in korean_tickers:
            try:
                df = stock.get_market_ohlcv(start_date_str, end_date_str, ticker)
                if df.empty:
                    errors.append(f"pykrx: 종목 코드 '{ticker}'에 대한 데이터를 찾을 수 없습니다.")
                    continue
                series = df['종가'].rename(ticker)
                all_data_frames.append(series)
                currency_info[ticker] = 'KRW'
            except Exception as e:
                errors.append(f"pykrx: 종목 코드 '{ticker}' 데이터 조회 실패: {e}")

    if overseas_tickers:
        for ticker in overseas_tickers:
            try:
                stock_ticker = yf.Ticker(ticker)
                info = stock_ticker.info
                data = stock_ticker.history(period=period, auto_adjust=False)
                
                if data.empty or 'Close' not in data.columns or data['Close'].isnull().all():
                    errors.append(f"yfinance: 티커 '{ticker}'에 대한 유효한 시계열 데이터를 찾을 수 없습니다.")
                    continue

                price_col = 'Adj Close' if 'Adj Close' in data.columns and not data['Adj Close'].isnull().all() else 'Close'
                price_series = data[price_col].rename(ticker)
                price_series.index = price_series.index.tz_convert('UTC').normalize()
                all_data_frames.append(price_series)
                currency_info[ticker] = info.get('currency', 'USD')
            except Exception as e:
                errors.append(f"yfinance: 티커 '{ticker}' 조회 중 예외 발생: {e}")

    if not all_data_frames:
        return pd.DataFrame(), errors, {}
    
    # 인덱스 시간대 통일 (UTC)
    for i in range(len(all_data_frames)):
        if all_data_frames[i].index.tz is None:
            all_data_frames[i].index = pd.to_datetime(all_data_frames[i].index).tz_localize('UTC')
        else:
            all_data_frames[i].index = all_data_frames[i].index.tz_convert('UTC')

    final_df = pd.concat(all_data_frames, axis=1, join='inner')

    if final_df.empty and not errors:
        errors.append("선택한 종목들의 공통 거래일이 없어 분석할 데이터가 없습니다.")

    return final_df.dropna(axis=0, how='any'), errors, currency_info

def predict_stock_prices_rf(stock_data, currency_info):
    """Random Forest 모델로 주가를 예측하고 통화 정보를 포함합니다."""
    predictions = {}
    for ticker in stock_data.columns:
        df = pd.DataFrame(stock_data[ticker])
        if df.empty or len(df) < 2: continue

        df['prediction'] = df[ticker].shift(-1)
        df.dropna(inplace=True)
        if len(df) < 2: continue
        
        X = np.array(df.drop(['prediction'], axis=1))
        y = np.array(df['prediction'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if len(X_train) == 0: continue

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        last_price = X[-1].reshape(1, -1)
        predicted_price = rf.predict(last_price)[0]
        current_price = last_price[0][0]
        
        direction = "상승" if predicted_price > current_price else "하락"
        
        predictions[ticker] = {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "direction": direction,
            "currency": currency_info.get(ticker, 'N/A')
        }
    return predictions

def get_portfolio_sentiment(tickers_info):
    """
    포트폴리오 전체의 뉴스 감성 지수를 분석합니다.
    국내/해외 종목을 구분하여 각각에 맞는 뉴스 소스를 사용합니다.
    """
    try:
        from crawlers import search_domestic_news, search_overseas_news
    except ImportError:
        print("!!! 치명적 오류: 'crawlers.py' 파일을 찾을 수 없어 포트폴리오 감성 분석이 불가능합니다.")
        return {"status": "데이터 없음", "sentiment_score": 0, "summary": {}, "analyzed_articles": []}
    
    all_news = []
    news_limit = 30

    # [수정] 입력받은 'tickers_info' 변수를 사용하도록 변경
    domestic_stocks = [s for s in tickers_info if s.get('ticker') and re.fullmatch(r"\d{6}", s['ticker'].split('.')[0])]
    overseas_stocks = [s for s in tickers_info if s.get('ticker') and not re.fullmatch(r"\d{6}", s['ticker'].split('.')[0])]

    domestic_names = [s.get('company_name', '알수없음') for s in domestic_stocks]
    # 해외 티커 (이름 변환은 crawlers.py에서 수행)
    overseas_tickers = [s.get('ticker') for s in overseas_stocks]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        if domestic_names:
            domestic_query = " OR ".join(domestic_names)
            print(f"포트폴리오 감성 분석: 국내 뉴스 검색 실행 (Query: {domestic_query[:100]}...)")
            # search_domestic_news는 이제 이름을 받아서 처리
            futures.append(executor.submit(search_domestic_news, domestic_query, limit=news_limit * len(domestic_names)))

        for ticker in overseas_tickers:
            # search_overseas_news는 티커를 받아서 내부에서 이름으로 변환 후 검색
            print(f"포트폴리오 감성 분석: 해외 뉴스 검색 실행 (Ticker: {ticker})")
            futures.append(executor.submit(search_overseas_news, ticker, limit=news_limit))

        for future in concurrent.futures.as_completed(futures):
            try:
                all_news.extend(future.result())
            except Exception as e:
                print(f"포트폴리오 뉴스 수집 중 오류: {e}")

    if not all_news:
        # ... (기존 데이터 없음 처리) ...
        return {"status": "데이터 없음", "sentiment_score": 0, "summary": {}, "analyzed_articles": []}

    unique_news = list({item['link']: item for item in all_news}.values())
    print(f"총 {len(unique_news)}개의 고유 뉴스 기사 분석 시작...")

    portfolio_name = "포트폴리오" if domestic_names else "해외 포트폴리오"
    if domestic_names:
         # 국내 이름이 하나라도 있으면 analyze_market_sentiment_from_news 호출 (내부에서 번역 처리)
         sentiment_result = analyze_market_sentiment_from_news(unique_news, portfolio_name)
    else:
         # 해외 주식만 있으면 analyze_news_sentiments_english 호출
         sentiment_result = analyze_news_sentiments_english(portfolio_name, unique_news)

    return {
        "status": sentiment_result.get('investment_strategy', 'N/A'),
        "sentiment_score": sentiment_result.get('sentiment_score', 0),
        "summary": sentiment_result.get('summary', {}),
        "analyzed_articles": sentiment_result.get('analyzed_articles', [])
    }
def optimize_portfolio(stock_data):
    """마코위츠 모델로 포트폴리오를 최적화합니다."""
    num_assets = stock_data.shape[1]
    if num_assets < 1: return {}
    if num_assets == 1: return {stock_data.columns[0]: 1.0}

    try:
        mu = expected_returns.mean_historical_return(stock_data)
        S = risk_models.sample_cov(stock_data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = dict(ef.clean_weights())
        return cleaned_weights
    except Exception as e:
        print(f"마코위츠 최적화 오류: {e}. 균등 분배로 대체합니다.")
        return {ticker: 1.0 / num_assets for ticker in stock_data.columns}

def run_full_portfolio_analysis(tickers_info, expert_rules, use_news_sentiment=True):
    """모든 분석 단계를 실행하고 결과를 통합합니다."""
    tickers = list(tickers_info.values())
    exchange_rate = get_exchange_rate()
    # (의존성) get_stock_data_for_portfolio 함수 호출
    stock_data, errors, currency_info = get_stock_data_for_portfolio(tickers)

    if stock_data.empty:
        raise ValueError(f"분석할 주가 데이터가 없습니다. 상세 오류: {errors}")

    if errors:
        st.warning(f"일부 종목 조회 실패: {', '.join(errors)}. 성공한 종목으로 분석을 진행합니다.")
        valid_tickers = [t for t in tickers if t in stock_data.columns]
        # tickers_info 딕셔너리도 유효한 티커 기준으로 필터링
        tickers_info = {name: ticker for name, ticker in tickers_info.items() if ticker in valid_tickers}
        if len(stock_data.columns) < 2:
             raise ValueError("데이터 조회 성공한 종목이 2개 미만이라 포트폴리오 최적화를 진행할 수 없습니다.")

    valid_tickers_info = {name: ticker for name, ticker in tickers_info.items() if ticker in stock_data.columns}
    valid_stock_data = stock_data[list(valid_tickers_info.values())]

    # (의존성) predict_stock_prices_rf, get_portfolio_sentiment, optimize_portfolio 호출
    model_predictions = predict_stock_prices_rf(valid_stock_data, currency_info) # 변경 없음

    sentiment_analysis = None # 초기화
    if use_news_sentiment:
        related_stocks_for_sentiment = [
            {'company_name': name, 'ticker': ticker}
            for name, ticker in valid_tickers_info.items()
        ]
        sentiment_analysis = get_portfolio_sentiment(related_stocks_for_sentiment)


    markowitz_weights = optimize_portfolio(valid_stock_data) # 변경 없음

    final_weights = markowitz_weights.copy()
    adjustment_reason = ""
    if use_news_sentiment and sentiment_analysis: # sentiment_analysis가 None이 아닌지 확인
        base_weight = expert_rules.get("base_weight", 1.0)
        final_weights = {t: w * base_weight for t, w in final_weights.items()}

        senti_status = sentiment_analysis.get('status', '중립적')
        senti_adjustment = expert_rules.get("sentiment_map", {}).get(senti_status, 0)

        # 티커 기준으로 가중치 조정
        for ticker in final_weights:
            # senti_adjustment 적용 시 기존 가중치 비율 유지하며 조정 (논리 수정 필요 가능성 있음)
            # 여기서는 단순히 더하는 방식으로 구현됨 (이전 코드 유지)
            final_weights[ticker] += senti_adjustment * final_weights[ticker] # 이 부분 논리 재검토 필요 시 가능

        if senti_adjustment != 0:
            adjustment_reason += f"뉴스 감성({senti_status}, {senti_adjustment:+.1%})을 반영해 전체 비중을 조정했습니다. "

        # 모델 예측 반영 (티커 기준)
        for ticker, pred in model_predictions.items():
            if ticker in final_weights and pred: # pred가 None이 아닌지 확인
                 pred_direction = pred.get('direction') # 예측 방향 가져오기
                 # direction이 0 또는 1 형태이므로, 문자열 키('상승', '하락')로 변환 필요
                 pred_direction_str = "상승" if pred_direction == 1 else "하락" if pred_direction == 0 else "중립"

                 pred_weight = expert_rules.get("prediction_weights", {}).get(pred_direction_str, 0)
                 final_weights[ticker] += pred_weight # 가중치 직접 더하기
                 if pred_weight != 0:
                     # valid_tickers_info 에서 이름 찾기
                     name = [k for k, v in valid_tickers_info.items() if v == ticker]
                     if name: # 이름 찾았을 경우
                         adjustment_reason += f"{name[0]}의 '{pred_direction_str}' 예측({pred_weight:+.1%})을 반영했습니다. "

        # 음수 가중치 0으로 만들고 정규화
        final_weights = {t: max(0, w) for t, w in final_weights.items()}
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {t: w / total_weight for t, w in final_weights.items()}
        else: # 모든 가중치가 0 이하가 된 경우
             num_tickers = len(final_weights)
             if num_tickers > 0: final_weights = {t: 1.0 / num_tickers for t in final_weights} # 균등 분배
             adjustment_reason += "AI 조정 결과 모든 비중이 0 이하가 되어 균등 분배로 대체합니다. "

    # 최종 결과를 이름 기준으로 변환
    markowitz_weights_by_name = {name: markowitz_weights.get(ticker, 0.0) for name, ticker in valid_tickers_info.items()}
    final_weights_by_name = {name: final_weights.get(ticker, 0.0) for name, ticker in valid_tickers_info.items()}
    predictions_by_name = {name: model_predictions.get(ticker) for name, ticker in valid_tickers_info.items()}

    return {
        "exchange_rate": exchange_rate,
        "model_prediction": {"individual": predictions_by_name},
        "sentiment_analysis": sentiment_analysis, # 최종 sentiment_analysis 결과 포함
        "markowitz_portfolio": {"weights": markowitz_weights_by_name},
        "final_portfolio": {"final_weights": final_weights_by_name, "reason": adjustment_reason.strip()}
    }
# =============================================================================
# 스크립트 실행 예시 (DB 초기화)
# =============================================================================
if __name__ == "__main__":
    print("DB 스키마 업데이트 (init_db)를 시작합니다...")
    # (의존성) init_db 함수 호출 (database.py)
    init_db()
    print("DB 스키마 업데이트 완료.")

    print("\n--- (참고) 외부 의존성 확인 ---")
    print(f"crawlers.search_domestic_news 함수: {search_domestic_news}")
    print(f"crawlers.search_overseas_news 함수: {search_overseas_news}")