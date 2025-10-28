import hashlib
import mysql.connector # 다른 import들과 함께
from datetime import datetime, timezone # 필요한 다른 import들
import pandas as pd

DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '1253', # MySQL 설치 시 설정한 비밀번호
    'database': 'FF_db'      # 생성한 데이터베이스 이름
}


def get_history_detail(history_id):
    """특정 분석 기록의 상세 내용을 가져옵니다."""
    conn = get_db_connection()
    if not conn: return None
    c = conn.cursor(dictionary=True)
    try:
        c.execute("SELECT report_content, pdf_path FROM history WHERE id = %s", (history_id,))
        return c.fetchone()
    finally:
        if c: c.close()
        if conn: conn.close()

def delete_chat_history(username: str) -> bool:
    """특정 사용자의 모든 채팅 기록을 삭제합니다."""
    conn = get_db_connection()
    if not conn: return False
    c = conn.cursor()
    try:
        c.execute("DELETE FROM chat_history WHERE username = %s", (username,))
        print(f"사용자 '{username}'의 채팅 기록 {c.rowcount}개 삭제 성공.")
        return True
    except mysql.connector.Error as err:
        print(f"채팅 기록 삭제 오류 (User: {username}): {err}")
        return False
    finally:
        if c: c.close()
        if conn: conn.close()

def news_sources_exist():
    """DB에 뉴스 데이터가 하나라도 있는지 확인합니다."""
    conn = get_db_connection()
    if not conn: return False
    c = conn.cursor()
    try:
        c.execute("SELECT 1 FROM news_articles LIMIT 1")
        return c.fetchone() is not None
    finally:
        if c: c.close()
        if conn: conn.close()


def get_daily_stock_sentiment_scores(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """[수정] 특정 종목/기간의 'sentiment' 레이블을 조회하고 '점수'를 계산하여 반환"""
    conn = get_db_connection()
    if not conn: return pd.DataFrame(columns=['sentiment_score']).set_index(pd.to_datetime([])) # 빈 DF 구조 명확화
    c = conn.cursor(dictionary=True)

    # 시작일과 종료일을 Naive UTC Date 문자열로 변환 (DB 비교용)
    start_date_str = start_date.astimezone(timezone.utc).strftime('%Y-%m-%d')
    end_date_str = end_date.astimezone(timezone.utc).strftime('%Y-%m-%d')

    # [수정] sentiment 레이블을 날짜별로 조회
    query = """
    SELECT DATE(pub_date) as date, sentiment, COUNT(*) as count
    FROM news_articles
    WHERE related_ticker = %s AND DATE(pub_date) BETWEEN %s AND %s AND sentiment IS NOT NULL
    GROUP BY DATE(pub_date), sentiment
    ORDER BY date ASC
    """
    try:
        # DB에 005930만 저장될 것을 대비하여 .KS 등 접미사 제거
        clean_ticker = ticker.split('.')[0]

        c.execute(query, (clean_ticker, start_date_str, end_date_str))
        data = c.fetchall()

        if not data: # 해당 기간/종목 데이터 없으면 빈 DF 반환
             return pd.DataFrame(columns=['sentiment_score']).set_index(pd.to_datetime([]))

        df_raw = pd.DataFrame(data)
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # 날짜별로 긍정/부정/중립 개수 집계 (pivot_table 사용)
        df_pivot = df_raw.pivot_table(index='date', columns='sentiment', values='count', fill_value=0)

        # 집계 결과에 누락된 sentiment 컬럼 추가 (예: 'positive'가 없는 날)
        for col in ['positive', 'negative', 'neutral']:
            if col not in df_pivot.columns:
                df_pivot[col] = 0

        # 점수 계산: (긍정 수 - 부정 수) / (긍정 수 + 부정 수 + 중립 수)
        total = df_pivot['positive'] + df_pivot['negative'] + df_pivot['neutral']

        # 0으로 나누는 경우 방지 (total이 0이면 점수도 0)
        df_score = ((df_pivot['positive'] - df_pivot['negative']) / total).where(total > 0, 0.0)

        # 최종 결과 DataFrame 생성 (컬럼 이름은 'sentiment_score' 유지)
        df_final = pd.DataFrame(df_score, columns=['sentiment_score'])
        df_final.index.name = 'date'

        return df_final

    except mysql.connector.Error as err:
        print(f"[{ticker}] 일별 종목 감성 점수 계산 오류: {err}")
        return pd.DataFrame(columns=['sentiment_score']) # 오류 시 빈 DF
    finally:
        if c: c.close()
        if conn: conn.close()


def delete_history_record(history_id: int) -> bool:
    """ID를 기준으로 특정 분석 기록을 삭제합니다."""
    conn = get_db_connection()
    if not conn: return False
    c = conn.cursor()
    try:
        c.execute("DELETE FROM history WHERE id = %s", (history_id,))
        if c.rowcount > 0: # 실제로 삭제된 행이 있는지 확인
            print(f"분석 기록 ID {history_id} 삭제 성공.")
            return True
        else:
            print(f"분석 기록 ID {history_id}를 찾을 수 없어 삭제하지 못했습니다.")
            return False
    except mysql.connector.Error as err:
        print(f"분석 기록 삭제 오류 (ID: {history_id}): {err}")
        return False
    finally:
        if c: c.close()
        if conn: conn.close()


def get_articles_since(last_check_time_utc, limit=100):
    """[수정] (텔레그램 봇용, sentiment 레이블 포함)"""
    conn = get_db_connection()
    if not conn: return []
    c = conn.cursor(dictionary=True)
    try:
        # 비교할 시간을 Naive UTC로 변환
        last_check_naive_utc = last_check_time_utc.astimezone(timezone.utc).replace(tzinfo=None)

        # --- [수정] SELECT 문에서 sentiment_score -> sentiment ---
        c.execute(
            """SELECT headline, link, source, image_url, pub_date, crawl_date, sentiment, related_ticker
               FROM news_articles
               WHERE pub_date > %s /* Naive UTC 끼리 비교 */
               ORDER BY pub_date ASC
               LIMIT %s""",
            (last_check_naive_utc, limit)
        )
        # --- [수정] 끝 ---
        articles = c.fetchall()

        # 조회된 naive 시간을 aware UTC로 변환
        for article in articles:
            if article.get('pub_date') and isinstance(article['pub_date'], datetime):
                article['pub_date'] = article['pub_date'].replace(tzinfo=timezone.utc)
            if article.get('crawl_date') and isinstance(article['crawl_date'], datetime):
                article['crawl_date'] = article['crawl_date'].replace(tzinfo=timezone.utc)

        return articles
    finally:
        if c: c.close()
        if conn: conn.close()


def get_db_connection():
    """데이터베이스 연결 객체를 반환합니다."""
    try:
        # autocommit=True 추가
        return mysql.connector.connect(**DB_CONFIG, autocommit=True)
    except mysql.connector.Error as err:
        print(f"데이터베이스 연결 오류: {err}")
        return None

def init_db():
    """[수정] sentiment 컬럼 추가, sentiment_score 컬럼 삭제"""
    conn = get_db_connection()
    if not conn: return
    c = conn.cursor()

    # --- 테이블 생성 ---
    # users 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            name VARCHAR(255) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    ''')
    # history 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            stock_name VARCHAR(255),
            ticker_code VARCHAR(20),
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            report_content LONGTEXT,
            pdf_path VARCHAR(255),
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    ''')
    # chat_history 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    ''')

    # --- [수정] news_articles 테이블 ---
    c.execute('''
        CREATE TABLE IF NOT EXISTS news_articles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            headline VARCHAR(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
            link VARCHAR(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL UNIQUE,
            source VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
            image_url VARCHAR(512),
            pub_date DATETIME(6),
            crawl_date DATETIME(6) NOT NULL,
            sentiment VARCHAR(10),       /* [신규] 감성 레이블 (positive, negative, neutral) */
            related_ticker VARCHAR(20),   /* 관련 종목 티커 */
            INDEX(source), INDEX(pub_date), INDEX(crawl_date),
            INDEX(sentiment),            /* [신규] 인덱스 추가 */
            INDEX(related_ticker)        /* 인덱스 추가 */
            /* sentiment_score 컬럼은 제거됨 */
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    ''')

    # --- [수정] 컬럼 추가/삭제 시도 ---
    # (기존 컬럼 추가 시도 유지 - 이미 존재하면 무시됨)
    try:
        c.execute("ALTER TABLE news_articles ADD COLUMN crawl_date DATETIME(6) NOT NULL AFTER image_url")
        c.execute("ALTER TABLE news_articles ADD INDEX (crawl_date)")
    except mysql.connector.Error: pass # 이미 있으면 무시
    try:
        c.execute("ALTER TABLE news_articles ADD COLUMN pub_date DATETIME(6) AFTER image_url")
        c.execute("ALTER TABLE news_articles ADD INDEX (pub_date)")
    except mysql.connector.Error: pass # 이미 있으면 무시
    try: # related_ticker 위치 조정 (sentiment 앞으로)
        c.execute("ALTER TABLE news_articles ADD COLUMN related_ticker VARCHAR(20) AFTER crawl_date")
        c.execute("ALTER TABLE news_articles ADD INDEX (related_ticker)")
    except mysql.connector.Error: pass # 이미 있으면 무시

    # [신규] sentiment 컬럼 추가 시도
    try:
        c.execute("ALTER TABLE news_articles ADD COLUMN sentiment VARCHAR(10) AFTER related_ticker") # related_ticker 뒤로 위치 조정
        c.execute("ALTER TABLE news_articles ADD INDEX (sentiment)")
    except mysql.connector.Error as err:
        if 'Duplicate column name' not in err.msg and 'Duplicate key name' not in err.msg:
            print(f"테이블 수정 중 예상치 못한 오류(sentiment): {err}")

    # [신규] sentiment_score 컬럼 삭제 시도 (주의: 데이터 손실 발생 가능)
    try:
        c.execute("ALTER TABLE news_articles DROP COLUMN sentiment_score")
        print("기존 sentiment_score 컬럼 삭제 완료.")
    except mysql.connector.Error as err:
        # 컬럼이 이미 없으면 오류 발생 (무시)
        if 'check that column/key exists' not in err.msg and 'Unknown column' not in err.msg:
             print(f"테이블 수정 중 예상치 못한 오류(sentiment_score 삭제): {err}")

    c.close()
    conn.close()

def save_news_articles(articles):
    """[수정] DB 저장 시 sentiment 레이블 저장"""
    conn = get_db_connection()
    if not conn or not articles: return
    c = conn.cursor()
    # [수정] 쿼리 변경 (sentiment 추가, sentiment_score 제거)
    query = "INSERT IGNORE INTO news_articles (headline, link, source, image_url, pub_date, crawl_date, sentiment, related_ticker) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None) # 현재 시간 (Naive UTC)

    data_to_insert = []
    for a in articles:
        if a.get('link'):
            pub_date, pub_date_naive_utc = a.get('pub_date'), None
            if isinstance(pub_date, datetime):
                # Aware UTC -> Naive UTC로 변환
                if pub_date.tzinfo:
                    pub_date_naive_utc = pub_date.astimezone(timezone.utc).replace(tzinfo=None)
                else: # 이미 Naive라면 UTC라고 가정
                    pub_date_naive_utc = pub_date
            # datetime 객체가 아니거나 없으면 NULL (None)

            # [수정] sentiment 레이블 가져오기 (없으면 'neutral')
            sentiment_label = a.get('sentiment', 'neutral')
            if sentiment_label not in ['positive', 'negative', 'neutral']:
                sentiment_label = 'neutral' # 유효하지 않으면 neutral

            data_to_insert.append((
                a.get('headline',''), a.get('link',''), a.get('source',''),
                a.get('image_url',''),
                pub_date_naive_utc, # Naive UTC 또는 None 저장
                now_utc_naive,      # Naive UTC 저장
                sentiment_label,         # [수정] 레이블 저장
                a.get('related_ticker')  # 관련 티커 또는 None 저장
            ))

    if not data_to_insert: return
    try:
        c.executemany(query, data_to_insert)
    finally:
        if c: c.close()
        if conn: conn.close()

def get_existing_links_by_source(source):
    """특정 출처의 기존 뉴스 링크들을 가져옵니다."""
    conn = get_db_connection()
    if not conn: return set()
    c = conn.cursor()
    try:
        c.execute("SELECT link FROM news_articles WHERE source = %s", (source,))
        return {row[0] for row in c.fetchall()}
    finally:
        if c: c.close()
        if conn: conn.close()

def get_articles_from_db(source=None, limit=20, page=1, search_query=None):
    """[수정] DB 조회 시 sentiment 레이블 포함"""
    conn = get_db_connection()
    if not conn: return [], 0
    c = conn.cursor(dictionary=True); offset = (page - 1) * limit
    conditions, params = [], [];
    if source: conditions.append("source = %s"); params.append(source)
    if search_query: conditions.append("headline LIKE %s"); params.append(f"%{search_query}%")
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    try:
        c.execute(f"SELECT COUNT(*) as total FROM news_articles {where_clause}", params)
        total_count = c.fetchone().get('total', 0)
        query_params = params + [limit, offset]
        # [수정] sentiment 조회 추가, sentiment_score 제거
        query = f"SELECT headline, link, source, image_url, pub_date, crawl_date, sentiment, related_ticker FROM news_articles {where_clause} ORDER BY IFNULL(pub_date, crawl_date) DESC LIMIT %s OFFSET %s"
        c.execute(query, query_params); articles = c.fetchall()

        # 조회된 naive 시간을 aware UTC로 변환
        for article in articles:
            if article.get('pub_date') and isinstance(article['pub_date'], datetime):
                article['pub_date'] = article['pub_date'].replace(tzinfo=timezone.utc)
            if article.get('crawl_date') and isinstance(article['crawl_date'], datetime):
                article['crawl_date'] = article['crawl_date'].replace(tzinfo=timezone.utc)

        return articles, total_count
    except mysql.connector.Error as err:
        print(f"DB 기사 조회 오류: {err}")
        return [], 0
    finally:
        if c: c.close()
        if conn: conn.close()

def add_user(username, password, name):
    """새 사용자를 추가합니다."""
    conn = get_db_connection()
    if conn is None: return False
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, name) VALUES (%s, %s, %s)",
                  (username, hash_password(password), name))
        return True
    except mysql.connector.Error: # Unique 제약 조건 위반 등
        return False
    finally:
        if c: c.close()
        if conn: conn.close()

def check_user(username, password):
    """사용자 로그인 정보를 확인합니다."""
    conn = get_db_connection()
    if conn is None: return None
    c = conn.cursor(dictionary=True) # 결과를 딕셔너리로 받기
    try:
        c.execute("SELECT username, password, name FROM users WHERE username = %s", (username,))
        user = c.fetchone()
        if user and user['password'] == hash_password(password):
            return user # 사용자 정보 반환
        return None # 로그인 실패
    finally:
        if c: c.close()
        if conn: conn.close()

def save_history(username, stock_name, ticker, report_content, pdf_path):
    """분석 기록을 저장합니다."""
    conn = get_db_connection()
    if not conn: return False
    c = conn.cursor()
    try:
        pdf_path_str = str(pdf_path) # Path 객체일 수 있으므로 문자열로 변환
        c.execute("INSERT INTO history (username, stock_name, ticker_code, report_content, pdf_path) VALUES (%s, %s, %s, %s, %s)",
                  (username, stock_name, ticker, report_content, pdf_path_str))
        return True
    except mysql.connector.Error as err:
        print(f"기록 저장 오류: {err}")
        return False
    finally:
        if c: c.close()
        if conn: conn.close()

def get_history(username):
    """특정 사용자의 분석 기록 목록을 가져옵니다."""
    conn = get_db_connection()
    if not conn: return []
    c = conn.cursor(dictionary=True)
    try:
        c.execute("SELECT id, stock_name, analysis_date FROM history WHERE username = %s ORDER BY analysis_date DESC", (username,))
        return c.fetchall()
    except mysql.connector.Error as err:
        print(f"기록 불러오기 오류: {err}")
        return []
    finally:
        if c: c.close()
        if conn: conn.close()

def save_chat_message(username, role, content):
    """채팅 메시지를 저장합니다."""
    conn = get_db_connection()
    if not conn: return
    c = conn.cursor()
    try:
        c.execute("INSERT INTO chat_history (username, role, content) VALUES (%s, %s, %s)", (username, role, content))
    except mysql.connector.Error as err:
        print(f"채팅 기록 저장 오류: {err}")
    finally:
        if c: c.close()
        if conn: conn.close()

def get_chat_history(username):
    """특정 사용자의 채팅 기록을 가져옵니다."""
    conn = get_db_connection()
    if not conn: return []
    c = conn.cursor() # 튜플로 받아 처리
    try:
        c.execute("SELECT role, content FROM chat_history WHERE username = %s ORDER BY timestamp ASC", (username,))
        # OpenAI API 형식에 맞게 변환
        return [{"role": row[0], "content": row[1]} for row in c.fetchall()]
    except mysql.connector.Error as err:
        print(f"채팅 기록 불러오기 오류: {err}")
        return []
    finally:
        if c: c.close()
        if conn: conn.close()

def hash_password(password):
    """비밀번호를 해시합니다."""
    return hashlib.sha256(password.encode()).hexdigest()
