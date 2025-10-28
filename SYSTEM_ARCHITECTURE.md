# 🏗️ 시스템 아키텍처 상세 분석

## 📋 목차
1. [Spring Boot Server](#1-spring-boot-server)
2. [Flask Server](#2-flask-server)
3. [MSA (Microservices Architecture)](#3-msa-architecture)
4. [통합 아키텍처 다이어그램](#4-통합-아키텍처-다이어그램)

---

## 1. Spring Boot Server

### 📍 **역할과 책임**

#### 1.1 핵심 역할
- **API Gateway 역할**: 프론트엔드와 Flask 서버 사이의 중개자
- **프록시 서버**: 클라이언트 요청을 Flask 서비스로 전달
- **사용자 인증 및 세션 관리**: 로그인, 로그아웃, 세션 유지
- **게시판 CRUD**: 자유게시판, 토론게시판, 메모 기능

#### 1.2 주요 기능

**✅ 정적 파일 서빙 (Port: 8080)**
```
- index.html (메인 페이지)
- chart.html (차트 페이지)
- news.html (뉴스 페이지)
- portfolio.html (포트폴리오 페이지)
- chatbot.html (AI 챗봇)
- board.html (게시판)
- login.html, register.html (인증)
```

**✅ RESTful API 엔드포인트**

| 엔드포인트 | 메서드 | 기능 | 프록시 여부 |
|-----------|--------|------|------------|
| `/api/chart/data/{ticker}/{interval}/{ema}/{rsi}` | GET | 차트 데이터 조회 | ✅ Flask |
| `/api/chart/analyze` | POST | 차트 패턴 분석 | ✅ Flask |
| `/api/chart/news/rss` | GET | RSS 뉴스 피드 | ✅ Flask |
| `/api/chart/news/analyze` | POST | 뉴스 감성 분석 | ✅ Flask |
| `/api/chart/ai/analyze` | POST | AI 예측 분석 | ✅ Flask |
| `/api/chart/chart/analyze-patterns` | POST | 차트 패턴 분석 | ✅ Flask |
| `/api/chart/portfolio/optimize` | POST | 포트폴리오 최적화 | ✅ Flask |
| `/api/chart/chatbot/chat` | POST | AI 챗봇 | ✅ Flask |
| `/api/chart/index-data` | GET | 주요 지수 데이터 | ✅ Flask |
| `/api/user/login` | POST | 로그인 | ❌ Native |
| `/api/user/logout` | POST | 로그아웃 | ❌ Native |
| `/api/user/status` | GET | 로그인 상태 | ❌ Native |
| `/api/board/*` | ALL | 게시판 CRUD | ❌ Native |

#### 1.3 데이터베이스 연동
- **MariaDB 연결** (Port: 3306)
- **JPA/Hibernate**: ORM 프레임워크
- **엔티티**: User, DiscussionPost, MemoPost
- **Repository**: UserRepository, DiscussionPostRepository, MemoPostRepository

#### 1.4 기술 스택
```java
- Java 17+
- Spring Boot 3.x
- Spring MVC
- JPA/Hibernate
- MariaDB JDBC
- RestTemplate (Flask 통신)
```

#### 1.5 설정 (application.properties)
```properties
server.port=8080
flask.service.url=http://localhost:5000
spring.datasource.url=jdbc:mariadb://localhost:3306/webdb
spring.jpa.hibernate.ddl-auto=none
```

---

## 2. Flask Server

### 📍 **역할과 책임**

#### 2.1 핵심 역할
- **데이터 분석 엔진**: Python 기반 데이터 처리 및 분석
- **AI/ML 서비스**: 머신러닝 모델 실행
- **외부 API 통합**: Yahoo Finance, OpenAI, RSS 피드 등
- **복잡한 계산**: 기술적 지표, 포트폴리오 최적화, 감성 분석

#### 2.2 주요 기능

**✅ Yahoo Finance 데이터 처리**
```python
# services/yahoo_finance_service.py
- 실시간 주가 데이터 수집 (yfinance)
- 캔들스틱 데이터 생성
- EMA (지수이동평균) 계산
- RSI (상대강도지수) 계산
- 거래량 분석
```

**✅ 뉴스 수집 및 분석**
```python
# services/news_rss_service.py
- RSS 피드 파싱 (한국경제, 매일경제, Nasdaq, MarketBeat)
- 웹 크롤링 (네이버, Google News)
- Yahoo Finance 뉴스 수집
- 실시간 뉴스 업데이트
```

**✅ AI/ML 분석**
```python
# services/ai_analysis_service.py
- Random Forest 모델 (주가 방향성 예측)
- 신뢰도 계산
- 가격 예측

# services/news_analysis_service.py
- TextBlob 기반 감성 분석
- 긍정/부정/중립 분류
```

**✅ 차트 패턴 분석**
```python
# services/chart_pattern_service.py
- 골든크로스/데드크로스 감지
- 이동평균선 돌파 분석
- 지지선/저항선 계산
```

**✅ 포트폴리오 최적화**
```python
# services/portfolio_optimizer_service.py
- 마코위츠 포트폴리오 이론 적용
- 효율적 경계 계산
- AI 기반 비중 조정
- 뉴스 감성 반영
```

**✅ AI 챗봇**
```python
# services/chatbot_service.py
- OpenAI GPT-3.5 Turbo 통합
- 금융 전문 프롬프트
- 대화 컨텍스트 관리
```

#### 2.3 RESTful API 엔드포인트 (Port: 5000)

| 엔드포인트 | 메서드 | 기능 | 데이터 소스 |
|-----------|--------|------|------------|
| `/api/data/<ticker>/<interval>/<ema>/<rsi>` | GET | 차트 데이터 | Yahoo Finance |
| `/api/analyze` | POST | 차트 분석 | 기술적 분석 |
| `/api/index-data` | GET | 주요 지수 | Yahoo Finance |
| `/api/news/rss` | GET | RSS 뉴스 | RSS + 크롤링 |
| `/api/news/analyze` | POST | 뉴스 분석 | 감성 분석 |
| `/api/ai/analyze` | POST | AI 분석 | Random Forest |
| `/api/chart/analyze-patterns` | POST | 차트 패턴 | 패턴 분석 |
| `/api/portfolio/available-stocks` | GET | 종목 리스트 | 내부 DB |
| `/api/portfolio/optimize` | POST | 포트폴리오 | 최적화 알고리즘 |
| `/api/chatbot/chat` | POST | 챗봇 응답 | OpenAI API |

#### 2.4 외부 서비스 연동

```python
# Yahoo Finance API
import yfinance as yf
stock = yf.Ticker("AAPL")
data = stock.history(period="1y")

# OpenAI API
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
response = client.chat.completions.create(...)

# RSS Feed
import feedparser
feed = feedparser.parse(url)

# Web Crawling
import requests
from bs4 import BeautifulSoup
response = requests.get(url)
soup = BeautifulSoup(html, 'lxml')
```

#### 2.5 기술 스택
```python
- Python 3.13
- Flask 3.x
- pandas, numpy (데이터 처리)
- yfinance (Yahoo Finance)
- scikit-learn (ML 모델)
- BeautifulSoup4 (크롤링)
- feedparser (RSS)
- OpenAI API
- TextBlob (감성 분석)
```

#### 2.6 설정 (.env)
```env
FLASK_PORT=5000
FLASK_DEBUG=True
OPENAI_API_KEY=your_key_here
```

---

## 3. MSA (Microservices Architecture)

### 📍 **아키텍처 개요**

이 프로젝트는 **하이브리드 마이크로서비스 아키텍처**를 채택합니다.

```
┌──────────────────────────────────────────────────────────────┐
│                       Client (Browser)                        │
│                    http://localhost:8080                      │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP
┌────────────────────────┴─────────────────────────────────────┐
│              API Gateway (Spring Boot)                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ • Request Routing                                      │  │
│  │ • Authentication (Session)                            │  │
│  │ • Load Balancing (Conceptual)                         │  │
│  │ • Logging & Monitoring                                │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────┬────────────────────────┬──────────────────────┘
               │                        │
               │ Proxy                  │ Direct Access
               ↓                        ↓
┌──────────────────────────┐  ┌──────────────────────────────┐
│  Data Analysis Service   │  │  User Management Service     │
│  (Flask)                │  │  (Spring Boot Native)         │
│  Port: 5000             │  │  Port: 8080                  │
│  ┌────────────────────┐  │  │  ┌────────────────────────┐ │
│  │ • Yahoo Finance    │  │  │  │ • Login/Logout         │ │
│  │ • News Collection  │  │  │  │ • Session Management   │ │
│  │ • AI/ML Analysis   │  │  │  │ • Board CRUD           │ │
│  │ • Portfolio Opt.   │  │  │  │ • User Repository      │ │
│  │ • Chatbot          │  │  │  └────────────────────────┘ │
│  └────────────────────┘  │  │                              │
└──────────────────────────┘  └──────────────────────────────┘
         │                                   │
         └──────────┬────────────────────────┘
                    ↓
         ┌──────────────────────┐
         │   MariaDB Database    │
         │   Port: 3306         │
         │   • Users            │
         │   • Posts            │
         │   • Memos            │
         └──────────────────────┘
```

### 🔍 **마이크로서비스 분리 원칙**

#### 3.1 비즈니스 도메인 분리

| 서비스 | 책임 | 기술 스택 |
|--------|------|-----------|
| **User Service** | 사용자 인증, 세션 관리, 게시판 | Spring Boot + JPA |
| **Data Analysis Service** | 데이터 수집, AI/ML 분석 | Flask + Python Libraries |
| **Chart Service** | 차트 렌더링, 시각화 | Lightweight Charts (Client) |

#### 3.2 데이터 처리 분리

**Spring Boot (Java 기반)**
- ✅ 빠른 API 라우팅
- ✅ 세션 관리 (HttpSession)
- ✅ 데이터베이스 CRUD
- ✅ 트랜잭션 관리
- ❌ 복잡한 데이터 분석
- ❌ 머신러닝 모델 실행

**Flask (Python 기반)**
- ✅ NumPy, Pandas (고성능 데이터 처리)
- ✅ Scikit-learn (ML 모델)
- ✅ BeautifulSoup (크롤링)
- ✅ OpenAI API (NLP)
- ❌ 세션 관리
- ❌ 엔터프라이즈 기능

#### 3.3 의존성 분리

```
Frontend
  ├─→ Spring Boot (Static Files + API Gateway)
  │     ├─→ MariaDB (User, Board Data)
  │     └─→ Flask (Data Analysis)
  │           ├─→ Yahoo Finance API
  │           ├─→ OpenAI API
  │           ├─→ RSS Feeds
  │           └─→ Web Crawling
```

### 🎯 **MSA 이점**

1. **기술 스택 독립성**
   - Java와 Python의 강점을 각각 활용
   - Java: 엔터프라이즈 기능
   - Python: 데이터 사이언스/ML

2. **스케일링 가능성**
   - Flask 서비스는 독립적으로 수평 확장 가능
   - Spring Boot는 안정적인 API 게이트웨이 역할

3. **개발 및 배포 독립성**
   - 각 서비스는 독립적으로 배포 가능
   - 버전 관리가 서비스별로 분리

4. **장애 격리**
   - Flask 서비스 장애 시 Spring Boot는 정상 동작
   - 게시판 기능은 Flask와 무관

### 📊 **서비스 간 통신**

#### Request Flow
```
Client Request
    ↓
Spring Boot Controller
    ├─→ Direct: User/Board API (Fast)
    └─→ Proxy: Analysis API
           ↓
        Flask Service
           ↓
        External APIs
           ↓
        Response → Client
```

#### CORS 설정
```python
# Flask app.py
CORS(app, origins=[
    "http://localhost:8080",
    "http://127.0.0.1:8080"
])
```

---

## 4. 통합 아키텍처 다이어그램

### 🔄 **완전한 시스템 흐름**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER (Browser)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │index.html│ │chart.html│ │news.html │ │portfolio │ │chatbot   │     │
│  │         │ │          │ │         │ │.html    │ │.html    │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                 │
│  │board.html│ │login.html│ │register  │ │myinfo    │                 │
│  │         │ │          │ │.html    │ │.html    │                 │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                 │
│  JavaScript: index.js, chart.js, news.js, portfolio.js, chatbot.js    │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │ HTTP/HTTPS (REST API)
┌────────────────────────────────┴──────────────────────────────────────┐
│                    API GATEWAY (Spring Boot)                         │
│                         Port: 8080                                   │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  ChartController                                               │  │
│  │  • GET /api/chart/data/*         → Proxy to Flask              │  │
│  │  • POST /api/chart/analyze        → Proxy to Flask              │  │
│  │  • GET /api/chart/news/rss       → Proxy to Flask              │  │
│  │  • POST /api/chart/news/analyze  → Proxy to Flask              │  │
│  │  • POST /api/chart/ai/analyze     → Proxy to Flask              │  │
│  │  • POST /api/chart/portfolio/*   → Proxy to Flask              │  │
│  │  • POST /api/chart/chatbot/chat  → Proxy to Flask              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  UserController                                                │  │
│  │  • POST /api/user/login          → Direct (JPA)                │  │
│  │  • POST /api/user/logout         → Direct (Session)            │  │
│  │  • GET /api/user/status          → Direct (Session)            │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  BoardController                                                │  │
│  │  • GET/POST /api/board/discussion → Direct (JPA)               │  │
│  │  • GET/POST /api/board/memo       → Direct (JPA)               │  │
│  └────────────────────────────────────────────────────────────────┘  │
└────────────────┬─────────────────────────┬─────────────────────────┘
                 │                         │
                 │ Proxy API Calls         │ Direct DB Access
                 ↓                         ↓
┌─────────────────────────────────┐  ┌──────────────────────────────┐
│   DATA ANALYSIS SERVICE         │  │  MariaDB Database           │
│   (Flask)                       │  │  Port: 3306                 │
│   Port: 5000                    │  │  ┌────────────────────────┐ │
│   ┌───────────────────────────┐ │  │  │ users                   │ │
│   │ yahoo_finance_service      │ │  │  │ • userId (PK)           │ │
│   │  • fetch_yahoo_data()      │ │  │  │ • username              │ │
│   │  • 계산 EMA/RSI            │ │  │  │ • password              │ │
│   └───────────────────────────┘ │  │  └────────────────────────┘ │
│   ┌───────────────────────────┐ │  │  ┌────────────────────────┐ │
│   │ news_rss_service          │ │  │  │ discussion_posts        │ │
│   │  • RSS 파싱                │ │  │  │ • postId (PK)          │ │
│   │  • 크롤링                   │ │  │  │ • username             │ │
│   └───────────────────────────┘ │  │  │ • title, content        │ │
│   ┌───────────────────────────┐ │  │  └────────────────────────┘ │
│   │ ai_analysis_service       │ │  │  ┌────────────────────────┐ │
│   │  • Random Forest 모델      │ │  │  │ memo_posts             │ │
│   │  • 방향성 예측             │ │  │  │ • memoId (PK)          │ │
│   └───────────────────────────┘ │  │  │ • username             │ │
│   ┌───────────────────────────┐ │  │  │ • memo_content         │ │
│   │ portfolio_optimizer       │ │  │  └────────────────────────┘ │
│   │  • 마코위츠 포트폴리오     │ │  └──────────────────────────────┘
│   │  • AI 비중 조정           │ │
│   └───────────────────────────┘ │
│   ┌───────────────────────────┐ │
│   │ chatbot_service           │ │
│   │  • OpenAI GPT-3.5         │ │
│   │  • 금융 전문 답변         │ │
│   └───────────────────────────┘ │
└────────┬────────────────────────┘
         │
         ↓
┌────────────────────────────────────────────────────────────┐
│              EXTERNAL SERVICES                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │Yahoo Finance │ │OpenAI API   │ │RSS Feeds    │       │
│  │(주가 데이터)  │ │(AI 챗봇)    │ │(뉴스)       │       │
│  │yfinance      │ │GPT-3.5 Turbo│ │7개 소스     │       │
│  └──────────────┘ └──────────────┘ └──────────────┘       │
└────────────────────────────────────────────────────────────┘
```

### 📝 **핵심 포인트**

1. **계층 분리**: Client → API Gateway → Microservices → External Services
2. **언어 분리**: Java (Infrastructure) vs Python (Data Science)
3. **독립성**: 각 서비스는 독립적으로 배포 및 스케일링 가능
4. **유연성**: 새로운 분석 기능 추가 시 Flask만 수정하면 됨

---

**문서 작성일**: 2025-10-27
**최종 업데이트**: 프로젝트 구조 분석 완료

