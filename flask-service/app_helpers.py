import pandas as pd
import json
import os
from pykrx import stock 

try:
    from db_utils import ( # db_utils 에서 직접 가져오도록 수정
        get_history,
        get_history_detail,
        check_user,
        add_user,
        delete_history_record
    )
    print(" -> [app_helpers.py] 'db_utils.py'에서 DB 함수 로드 성공.")
except ImportError:
    print("경고: 'db_utils.py' 파일을 찾을 수 없습니다. sidebar 기능이 작동하지 않습니다.")
    # 임시 함수 정의는 그대로 둡니다.
    def get_history(username): return []
    def get_history_detail(history_id): return None
    def check_user(username, password): return None
    def add_user(username, password, name): return False
    def delete_history_record(history_id): return False
# -------------------------------------------------


# =============================================================================
# 2. config.py 내용
# =============================================================================

REALTIME_MONITORING_CONFIG = {
    "monitoring_interval_minutes": 1,
    "news_sources": ["get_major_news", "get_bloomberg_news", "get_yahoo_finance_news"],
}
MODEL_PARAMETERS = {
    "training_years": 3,
    "top_n_features": 3,
}
PORTFOLIO_PARAMETERS = {
    "markowitz_period_days": 365,
    "base_stock_weight": 0.95,
    "sentiment_weight_map": {
        "매우 긍정적": 0.05, "긍정적": 0.02, "중립적": 0.0,
        "부정적": -0.10, "매우 부정적": -0.20,
    },
}
ANALYSIS_PARAMETERS = {
    "domestic_news_limit": 100,
    "overseas_news_limit": 100,
    "portfolio_news_limit": 20,
}
ALERT_TRIGGERS = {
    "min_new_articles_to_trigger_event": 1, 
    "required_sentiment_opinion": {
        "buy": "긍정적 전망",
        "sell": "부정적 전망"
    },
    "required_model_prediction": {
        "buy": 1,
        "sell": 0
    },
}
TELEGRAM_BOT_CONFIG = {
    "send_status_updates": False, 
}


# =============================================================================
# 3. utils.py 내용 (+ 누락된 stock_list_manager.py 기능)
# =============================================================================

# [신규] stock_map.json 파일 경로 정의
STOCK_MAP_FILE = "stock_map.json"

# [신규] 누락되었던 주식 목록 생성 함수
def update_stock_mapping_file():
    """
    pykrx를 사용해 KOSPI, KOSDAQ의 모든 종목 코드를 가져와
    stock_map.json 파일로 저장하고, 주요 해외 주식을 추가합니다.
    """
    print("주식 목록(stock_map.json) 생성을 시작합니다...")
    stock_map = {}
    
    try:
        # KOSPI + KOSDAQ 종목 가져오기
        for market in ["KOSPI", "KOSDAQ"]:
            tickers = stock.get_market_ticker_list(market=market)
            for ticker in tickers:
                name = stock.get_market_ticker_name(ticker)
                if name: # 스팩(SPAC) 등 제외
                    stock_map[name] = ticker
        
        print(f"  -> 국내 주식 {len(stock_map)}개 로드 완료.")
        
        # 주요 해외 주식 수동 추가
        overseas_stocks = {
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "Google": "GOOGL",
            "Amazon": "AMZN",
            "NVIDIA": "NVDA",
            "Tesla": "TSLA",
            "Meta": "META",
            "NAVER": "035420", # [중요] 네이버는 국내 주식이므로 여기에 추가 (또는 pykrx가 잡음)
            "카카오": "035720"  # 카카오 추가
        }
        
        # 중복 방지하며 추가
        for name, ticker in overseas_stocks.items():
            if name not in stock_map:
                stock_map[name] = ticker
                
        # [수정] 네이버(035420)가 pykrx 목록에 없을 경우를 대비해 한번 더 확인
        if "NAVER" not in stock_map and "네이버" not in stock_map:
            stock_map["NAVER"] = "035420"
            
        with open(STOCK_MAP_FILE, 'w', encoding='utf-8') as f:
            json.dump(stock_map, f, ensure_ascii=False, indent=4)
            
        print(f"주식 목록 저장 완료. (총 {len(stock_map)}개, 파일: {STOCK_MAP_FILE})")
        return True

    except Exception as e:
        print(f"주식 목록 생성 중 오류 발생: {e}")
        # 오류 발생 시 최소한의 목록이라도 저장
        if not os.path.exists(STOCK_MAP_FILE):
             fallback_map = {"삼성전자": "005930", "Apple": "AAPL", "NAVER": "035420"}
             with open(STOCK_MAP_FILE, 'w', encoding='utf-8') as f:
                json.dump(fallback_map, f, ensure_ascii=False, indent=4)
        return False

# --- 기존 utils.py 함수들 ---

def get_stock_list_from_map():
    """저장된 파일에서 국내 주식 목록만 필터링하여 반환합니다."""
    try:
        with open(STOCK_MAP_FILE, 'r', encoding='utf-8') as f:
            stock_map = json.load(f)
        # 티커가 숫자로만 이루어진 것이 국내 주식
        return [f"{name} ({ticker})" for name, ticker in stock_map.items() if ticker.isdigit()]
    except Exception:
        # [수정] 예외 발생 시 기본값 반환
        print(f"경고: get_stock_list_from_map 실패. '{STOCK_MAP_FILE}' 파일이 없습니다.")
        return ["삼성전자 (005930)"]
   


def get_overseas_stock_list_from_map():
    """저장된 파일에서 해외 주식 목록만 필터링하여 반환합니다."""
    try:
        with open(STOCK_MAP_FILE, 'r', encoding='utf-8') as f:
            stock_map = json.load(f)
        # 티커에 문자가 포함된 것이 해외 주식
        return [f"{name} ({ticker})" for name, ticker in stock_map.items() if not ticker.isdigit()]
    except Exception:
        print(f"경고: get_overseas_stock_list_from_map 실패. '{STOCK_MAP_FILE}' 파일이 없습니다.")
        return ["Apple (AAPL)"]
   

def get_all_stock_list_from_map():
    """저장된 파일에서 전체 주식 목록을 반환합니다."""
    try:
        with open(STOCK_MAP_FILE, 'r', encoding='utf-8') as f:
            stock_map = json.load(f)
        return [f"{name} ({ticker})" for name, ticker in stock_map.items()]
    except Exception:
         print(f"경고: get_all_stock_list_from_map 실패. '{STOCK_MAP_FILE}' 파일이 없습니다.")
         # [수정] 예외 발생 시 기본값 반환 (NAVER 포함)
         return ["삼성전자 (005930)", "Apple (AAPL)", "NAVER (035420)"]
   

def get_stock_mapping():
    """
    저장된 주식 목록 파일을 읽어오고, 파일이 없으면 새로 생성합니다.
    """
    if not os.path.exists(STOCK_MAP_FILE):
        print(f"'{STOCK_MAP_FILE}' 파일이 없어 새로 생성합니다.")
        # [수정] 외부 import 대신 내부 함수 호출
        update_stock_mapping_file()
        
    try:
        with open(STOCK_MAP_FILE, 'r', encoding='utf-8') as f:
            stock_map = json.load(f)
        print(f"'{STOCK_MAP_FILE}'에서 {len(stock_map)}개의 주식 정보를 불러왔습니다.")
        return stock_map
    except Exception as e:
        print(f"주식 맵 파일 로딩 중 오류: {e}")
        return {"삼성전자": "005930", "Apple": "AAPL", "NAVER": "035420"}

# [신규] Home.py에서 사용하던 pykrx 함수들
def get_top_trading_volume(top_n=10):
    """거래대금 상위 N개 종목 반환"""
    try:
        today_str = pd.Timestamp.now().strftime('%Y%m%d')
        df = stock.get_market_cap(today_str)
        df_sorted = df.sort_values(by="거래대금", ascending=False)
        top_stocks = df_sorted.head(top_n)
        return [(row['종목명'], ticker) for ticker, row in top_stocks.iterrows()]
    except Exception as e:
        print(f"거래대금 상위 조회 오류: {e}")
        return [("삼성전자", "005930")]


def get_top_gainers(top_n=10):
    """등락률 상위 N개 종목 반환"""
    try:
        today_str = pd.Timestamp.now().strftime('%Y%m%d')
        df = stock.get_market_cap(today_str)
        df_sorted = df.sort_values(by="등락률", ascending=False)
        top_stocks = df_sorted.head(top_n)
        return [(row['종목명'], ticker) for ticker, row in top_stocks.iterrows()]
    except Exception as e:
        print(f"등락률 상위 조회 오류: {e}")
        return [("삼성전자", "005930")]


def get_top_losers(top_n=10):
    """등락률 하위 N개 종목 반환"""
    try:
        today_str = pd.Timestamp.now().strftime('%Y%m%d')
        df = stock.get_market_cap(today_str)
        df_sorted = df.sort_values(by="등락률", ascending=True)
        top_stocks = df_sorted.head(top_n)
        return [(row['종목명'], ticker) for ticker, row in top_stocks.iterrows()]
    except Exception as e:
        print(f"등락률 하위 조회 오류: {e}")
        return [("삼성전자", "005930")]




if __name__ == "__main__":
    """
    이 스크립트를 직접 실행하면(python app_helpers.py)
    stock_map.json 파일을 생성하거나 업데이트합니다.
    """
    print("app_helpers.py 스크립트 실행됨...")
    update_stock_mapping_file()