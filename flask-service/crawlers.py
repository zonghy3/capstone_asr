import os
import json
import time
import re
import requests
import yfinance as yf
import urllib.parse
import concurrent.futures
import tempfile
from app_helpers import get_stock_mapping
from urllib.parse import quote, urljoin, quote_plus
from datetime import datetime, timezone, timedelta
from dateutil import parser as date_parser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Selenium 관련 import
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

CHROME_DRIVER_PATH = r"C:\Users\chslo\.wdm\drivers\chromedriver\win64\141.0.7390.122\chromedriver-win32/chromedriver.exe" 

stock_name_to_ticker_map = get_stock_mapping()
ticker_to_name_map = {v: k for k, v in stock_name_to_ticker_map.items()}

def get_driver():
    """매번 새로운 Selenium WebDriver를 초기화하고 반환합니다."""
    
    options = Options() 
    
    options.page_load_strategy = 'eager' 
    
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    user_data_dir = tempfile.mkdtemp()
    options.add_argument(f"--user-data-dir={user_data_dir}")
    
    # [수정] chromedriver-manager를 service 객체로 사용
    try:
        service = Service(executable_path=CHROME_DRIVER_PATH)
    except Exception as e:
        print(f"!!! 치명적 오류: Chrome 드라이버 경로({CHROME_DRIVER_PATH})를 찾을 수 없습니다.")
        print("!!! CHROME_DRIVER_PATH 변수를 올바르게 설정했는지 확인하세요.")
        print(f"!!! 상세 오류: {e}")
        return None # 드라이버 생성 실패
        
    driver = webdriver.Chrome(service=service, options=options)
    
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
              get: () => undefined
            })
        '''
    })
    return driver

    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
              get: () => undefined
            })
        '''
    })
    return driver
# 3. google

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def _parse_google_item(item):
    """Google RSS item을 파싱하여 딕셔너리로 반환 (pub_date 포함)"""
    title = item.title.get_text(strip=True) if item.title else "제목 없음"
    link = item.link.get_text(strip=True) if item.link else ""
    source = item.source.get_text(strip=True) if item.source else "Google News"

    pub_date_utc = None # 초기화
    date_str = item.pubDate.get_text(strip=True) if item.pubDate else ""
    if date_str:
        try:
            # Google RSS는 GMT/UTC 시간을 제공하므로 dateutil이 aware UTC 객체로 파싱
            aware_dt_utc = date_parser.parse(date_str)
            # 이미 UTC이므로 astimezone 필요 없음
            pub_date_utc = aware_dt_utc
        except ValueError:
            print(f"구글 시간 파싱 오류: {date_str}")
            pass # 실패 시 아래 기본값 사용

    if pub_date_utc is None:
        pub_date_utc = datetime.now(timezone.utc)

    return {
        "headline": title, "link": link, "source": source, "image_url": "",
        "pub_date": pub_date_utc # UTC 시간 저장
    }

def search_google_news(query, limit=100):
    news_items = []
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        for item in items[:limit]:
            news_items.append(_parse_google_item(item))
    except Exception as e:
        print(f"Google 뉴스 검색 중 오류 발생: {e}")
    return news_items

def get_latest_google_news(limit=100):
    news_items = []
    url = "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=ko&gl=KR&ceid=KR:ko"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        for item in items[:limit]:
            news_items.append(_parse_google_item(item))
    except Exception as e:
        print(f"Google 최신 뉴스 수집 중 오류 발생: {e}")
    return news_items

# 4. yahoo

# 상대 시간 파싱 함수 (변경 없음)
def parse_relative_time(time_str):
    now = datetime.now(timezone.utc)
    time_str = time_str.lower()
    if "just now" in time_str or "moments ago" in time_str: return now
    match = re.search(r'(\d+)\s*(m|h|d)', time_str)
    if match:
        value = int(match.group(1)); unit = match.group(2)
        if unit == 'm': return now - timedelta(minutes=value)
        elif unit == 'h': return now - timedelta(hours=value)
        elif unit == 'd': return now - timedelta(days=value)
    return None

def _crawl_yahoo_news(limit=100):
    """실제 야후 파이낸스 뉴스 크롤링을 수행하는 내부 함수"""
    url = "https://finance.yahoo.com/topic/latest-news/"
    news_items = []
    driver = None
    print(f"야후 실시간 속보 크롤링 시작")
    try:
        driver = get_driver() # common.py에 정의된 함수 사용
        driver.set_page_load_timeout(30) # 페이지 로드 최대 30초 대기
        driver.set_script_timeout(20)
        driver.implicitly_wait(10) # 요소를 찾을 때 최대 10초 대기
        driver.maximize_window(); driver.get(url)
        wait = WebDriverWait(driver, 2)

        

        news_container_selector = "li[class*='stream-item']"
        try: # 뉴스 컨테이너 로딩 대기
            WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, news_container_selector)))
        # [수정] SyntaxError 해결: except 블록 내용 분리 및 들여쓰기
        except Exception as e:
            print(f"야후: 뉴스 컨테이너 로딩 실패: {e}")
            return [] 

        # 스크롤 로직 (변경 없음)
        scroll_attempts, max_scroll_attempts = 0, 15
        while len(news_items) < limit and scroll_attempts < max_scroll_attempts:
            current_article_count = len(driver.find_elements(By.CSS_SELECTOR, news_container_selector))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);"); time.sleep(2)
            new_article_count = len(driver.find_elements(By.CSS_SELECTOR, news_container_selector))
            if new_article_count == current_article_count:
                scroll_attempts += 1
                if scroll_attempts >= 3: break
            else: scroll_attempts = 0
            scroll_attempts += 1

        # 파싱 로직 (변경 없음)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        articles = soup.select(news_container_selector)
        for item in articles:
            if len(news_items) >= limit: break
            headline_tag = item.select_one('h3')
            link_tag = headline_tag.find_parent('a') if headline_tag else None
            relative_time_tag = item.select_one('div[class*="publishing"]')
            if not relative_time_tag: relative_time_tag = item.select_one('div.yf-m1e6lz')
            absolute_time_tag = item.select_one('time')
            if not headline_tag or not link_tag: continue
            headline = headline_tag.text.strip()
            link = link_tag.get('href', '')
            if not link.startswith('http'): link = urljoin(url, link)
            img_tag = item.select_one('img')
            pub_date_utc = None
            if relative_time_tag: # 상대 시간
                time_str = relative_time_tag.get_text(strip=True)
                match = re.search(r'•\s*(.*)', time_str)
                if match: relative_str = match.group(1).strip(); pub_date_utc = parse_relative_time(relative_str)
            if pub_date_utc is None and absolute_time_tag and absolute_time_tag.get('datetime'): # 절대 시간
                date_str = absolute_time_tag['datetime']
                try: aware_dt = date_parser.parse(date_str); pub_date_utc = aware_dt.astimezone(timezone.utc)
                except ValueError: pass
            if pub_date_utc is None: pub_date_utc = datetime.now(timezone.utc) # 현재 시간
            news_items.append({"headline": headline, "link": link, "source": "Yahoo Finance", "image_url": img_tag.get('src') if img_tag and img_tag.get('src') else "", "pub_date": pub_date_utc})
    finally:
        if driver: driver.quit()
    return news_items[:limit]

# search_yahoo_news_by_ticker 함수 (이전 수정본 유지 - 'content' 파싱)
def search_yahoo_news_by_ticker(query, limit=50):
    """yfinance 라이브러리를 사용해 특정 티커의 뉴스를 가져오고, 실제 구조에 맞게 파싱합니다."""
    news_items = []
    try:
        ticker = yf.Ticker(query)
        news_list = ticker.news
        for news_item in news_list[:limit]:
            if isinstance(news_item, dict) and 'content' in news_item and isinstance(news_item['content'], dict):
                content = news_item['content']
                title = content.get('title')
                pub_date_str = content.get('pubDate')
                link_url = None
                if isinstance(content.get('canonicalUrl'), dict): link_url = content['canonicalUrl'].get('url')
                if not link_url and isinstance(content.get('clickThroughUrl'), dict): link_url = content['clickThroughUrl'].get('url')
                if not link_url and 'link' in news_item: link_url = news_item.get('link')

                if title and link_url and pub_date_str:
                    pub_date_utc = None
                    try: aware_dt = date_parser.parse(pub_date_str); pub_date_utc = aware_dt.astimezone(timezone.utc)
                    except (ValueError, TypeError): pass
                    if pub_date_utc is None: pub_date_utc = datetime.now(timezone.utc)
                    news_items.append({"headline": title, "link": link_url, "source": "Yahoo Finance", "image_url": "", "pub_date": pub_date_utc})
                # else: print(f"DEBUG yfinance: 필수 키(title, url, pubDate) 누락 또는 추출 실패: {content}") # 디버깅 필요 시 주석 해제
            # else: print(f"DEBUG yfinance: 예상치 못한 뉴스 항목 구조: {news_item}") # 디버깅 필요 시 주석 해제
    except Exception as e:
        print(f"yfinance로 '{query}' 뉴스 검색 중 오류 발생: {e}")
    return news_items


# 5. naver.py

# .env 파일 로드
load_dotenv()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# 공통 헤더
COMMON_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# KST 시간대
KST = timezone(timedelta(hours=9))

# [수정] 내부 로직만 변경 (URL, 선택자)
def _crawl_major_news(limit=100):
    """[수정] 네이버 금융 '실시간 속보'를 'limit' 개수만큼 여러 페이지에 걸쳐 크롤링합니다."""
    news_items = []
    base_url = "https://finance.naver.com"
    
    # [수정] '실시간 속보(LSS2D)' URL 사용
    list_base_url = "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258"

    # [수정] 최대 15페이지까지 순회 (페이지당 약 20개 기사)
    max_page_to_crawl = 15 
    
    print(f"네이버 실시간 속보 크롤링 시작...") # 로그 수정

    try:
        for page in range(1, max_page_to_crawl + 1):
            url = f"{list_base_url}&page={page}"

            response = requests.get(url, headers=COMMON_HEADERS, timeout=10)
            response.raise_for_status()
            response.encoding = 'EUC-KR'
            html_content = response.text
            soup = BeautifulSoup(html_content, 'lxml')

            news_list = soup.select('#contentarea_left > ul.realtimeNewsList > li')
            
            if not news_list:
                print(f"  -> {page} 페이지에서 뉴스 목록을 찾지 못함. 크롤링 중단.")
                break

            for item in news_list:
                headline_link_tag = item.select_one('dl > dd:nth-child(2) > a')
                
             
                date_tag = item.select_one('dl > dd:nth-child(3) > span.wdate')

                if headline_link_tag and date_tag:
                    headline = headline_link_tag.get('title', headline_link_tag.get_text(strip=True))
                    link = headline_link_tag.get('href', '')
                    if not link.startswith('http'):
                        link = urljoin(base_url, link)

                    pub_date_utc = None
                    date_str = date_tag.get_text(strip=True) # 예: "2025-10-22 17:09"
                    try:
                        # [수정] '실시간 속보' 시간 형식 '%Y-%m-%d %H:%M' 파싱
                        naive_dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                        aware_dt_kst = naive_dt.replace(tzinfo=KST)
                        pub_date_utc = aware_dt_kst.astimezone(timezone.utc)
                    except ValueError:
                        print(f"네이버 시간 파싱 오류 (실시간 속보): {date_str}")
                        pub_date_utc = datetime.now(timezone.utc)

                    news_items.append({
                        "headline": headline,
                        "link": link,
                        "source": "Naver Major",
                        "image_url": "", # 이미지 없음
                        "pub_date": pub_date_utc
                    })

                    # [추가] limit 개수에 도달하면 즉시 모든 루프 종료
                    if len(news_items) >= limit:
                        break
            
            # [추가] limit 개수에 도달하면 외부 루프도 종료
            if len(news_items) >= limit:
                break

            # [추가] 네이버 서버에 대한 과도한 요청 방지
            time.sleep(0.5) 

    except Exception as e:
        print(f"네이버 실시간 속보 크롤링 중 오류 발생: {e}") # 로그 수정

    print(f"네이버 실시간 속보 크롤링 완료. 총 {len(news_items)}개 수집.") # 로그 수정
    return news_items[:limit] # 정확히 limit 개수만큼 (또는 그 이하) 반환
# search_naver_news 함수 (변경 없음)
def search_naver_news(query, limit=100):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("네이버 API 키가 .env 파일에 설정되지 않았습니다.")
        return []
    encText = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={encText}&start=1&display={min(limit, 100)}&sort=sim"
    api_headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        "User-Agent": COMMON_HEADERS['User-Agent']
    }
    news_items = []
    try:
        response = requests.get(url, headers=api_headers, timeout=10)
        response.raise_for_status()
        result = response.json()
        for item in result.get('items', []):
            pub_date_utc = None
            date_str = item.get('pubDate', '')
            if date_str:
                try:
                    aware_dt = date_parser.parse(date_str)
                    pub_date_utc = aware_dt.astimezone(timezone.utc)
                except ValueError:
                    print(f"네이버 시간 파싱 오류 (API): {date_str}")
                    pass
            if pub_date_utc is None:
                pub_date_utc = datetime.now(timezone.utc)
            news_items.append({
                "headline": item['title'].replace('<b>', '').replace('</b>', ''),
                "link": item['link'], "source": "Naver News", "image_url": "",
                "pub_date": pub_date_utc
            })
    except Exception as e:
        print(f"네이버 뉴스 API 검색 중 오류: {e}")
    return news_items

# 6.init

__all__ = [
    # news_collertor.py가 직접 호출하는 함수들
    '_crawl_major_news',
    # '_crawl_bloomberg_news', # 삭제됨
    '_crawl_yahoo_news',
    'get_latest_google_news',
    # 다른 파일에서 사용하는 함수들
    'search_unified_news',      # News.py 검색용
    'search_domestic_news',     # AI 분석용 (1_AI_Report 등)
    'search_overseas_news'      # AI 분석용 (1_AI_Report 등)
]

def search_unified_news(query, limit=100):
    """(News.py 검색용) 네이버와 구글을 동시에 검색합니다."""
    all_results = []
    limit_per_source = (limit + 1) // 2

    search_functions = [
        lambda l: search_naver_news(query, limit=l),
        lambda l: search_google_news(query, limit=l)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(func, limit_per_source) for func in search_functions}
        for future in concurrent.futures.as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception as exc:
                print(f'통합 검색 중 오류 발생: {exc}')

    # pub_date 기준으로 최신순 정렬
    min_datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
    all_results.sort(key=lambda x: x.get('pub_date', min_datetime), reverse=True)
    return all_results[:limit]

def search_domestic_news(query, limit=50):
    """(AI 분석용) 국내 주식 분석을 위해 네이버와 구글을 '주식 이름'으로 검색합니다."""
    # query가 이미 주식 이름 또는 이름 목록이라고 가정하고 그대로 사용
    print(f"국내 뉴스 검색 시작 (Query: {query}, 최대 {limit}개)")
    return search_unified_news(query, limit) # 기존 통합 검색 함수 활용

def search_overseas_news(ticker, limit=50):
    """(AI 분석용) 해외 주식 분석을 위해 야후와 구글을 '영어 이름'으로 검색합니다."""
    stock_name = ticker_to_name_map.get(ticker, ticker) # stock_map에서 이름 찾기, 없으면 티커 사용
    print(f"해외 뉴스 검색 시작 (Name: {stock_name}, Ticker: {ticker}, 최대 {limit}개)")

    all_results = []
    limit_per_source = (limit + 1) // 2

    # 검색어로 stock_name (영어 이름) 사용
    search_tasks = [
        ("Yahoo Name Search", lambda l: search_yahoo_news_by_ticker(stock_name, limit=l)), # 야후는 티커 대신 이름 검색 시도 (yf.Ticker가 이름도 일부 처리 가능)
        ("Google Name Search", lambda l: search_google_news(stock_name, limit=l))
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_source = {executor.submit(func, limit_per_source): source_name for source_name, func in search_tasks}
        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                result = future.result()
                print(f"  -> [{source_name}]에서 {len(result)}개 뉴스 수집 완료.")
                all_results.extend(result)
            except Exception as exc:
                print(f'해외 통합 검색 중 [{source_name}]에서 오류 발생: {exc}')

    # pub_date 기준으로 최신순 정렬
    min_datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
    all_results.sort(key=lambda x: x.get('pub_date', min_datetime), reverse=True)
    return all_results[:limit]

# 7. manager


#import는 'database.py' 파일이 이 스크립트와 동일한 위치에
# 존재해야 정상적으로 작동
try:
    from db_utils import get_existing_links_by_source, save_news_articles, get_articles_from_db 
    print(" -> [crawlers.py] 'db_utils.py'에서 DB 함수 로드 성공.")
except ImportError:
    print("="*50)
    print("경고: 'db_utils.py' 파일을 찾을 수 없습니다.")
    print("'manage_news_crawling' 함수는 DB 관련 기능 없이 실행되거나 오류를 발생시킬 수 있습니다.")
    print("="*50)
    
    # 임시 함수 정의 (ImportError 방지)
    def get_existing_links_by_source(source=None):
        print(f"임시 함수 호출 (backend_logic.py 없음): get_existing_links_by_source(source={source})")
        return []

    def save_news_articles(articles):
        print(f"임시 함수 호출 (backend_logic.py 없음): save_news_articles(총 {len(articles)}개)")
        pass

    def get_articles_from_db(source=None, limit=100):
        print(f"임시 함수 호출 (backend_logic.py 없음): get_articles_from_db(source={source}, limit={limit})")
        return []

# --------------------------

def manage_news_crawling(source_name, crawl_func, limit):
    """
    뉴스 크롤링, 신규 기사 저장, DB 조회까지의 전체 과정을 관리합니다.
    (원래의 캐싱 로직으로 복원)
    """
    
    try:
        from backend_logic import get_existing_links_by_source, save_news_articles, get_articles_from_db
    except ImportError:
        print(f"!!! 치명적 오류: [{source_name}] 'backend_logic.py' 파일을 찾을 수 없어 DB 작업이 불가능합니다.")
        return [] # 빈 리스트 반환
    
    try:
        # 1. 웹사이트에서 최신 기사 목록을 크롤링합니다.
        crawled_articles = crawl_func()
        
        if crawled_articles:
            # 2. DB에 이미 저장된 기사 링크들을 가져옵니다.
            existing_links = get_existing_links_by_source(source=source_name)

            # 3. 크롤링된 기사 중 DB에 없는 '새로운' 기사만 필터링합니다.
            new_articles = [
                article for article in crawled_articles if article['link'] not in existing_links
            ]

            # 4. 새로운 기사가 있으면 DB에 저장합니다.
            if new_articles:
                # DB 저장을 위해 source_name을 명시적으로 추가
                for article in new_articles:
                    article['source'] = source_name
                print(f"[{source_name}] {len(new_articles)}개의 새 기사를 DB에 저장합니다.")
                save_news_articles(new_articles)
    
    except Exception as e:
        print(f"[{source_name}] 크롤링 중 오류 발생: {e}")

    return get_articles_from_db(source=source_name, limit=limit)[0]

