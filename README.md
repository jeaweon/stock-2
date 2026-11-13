# 🚀 미국 주식 챔피언 DNA 모의 투자 & 백테스트 분석 시스템

298개 미국 주식 전수 2년치(최대 730일) 1시간 봉 실증 백테스팅을 통해 도출된 **고수익 챔피언 DNA(변동성 파워 + 거시 정배열 + 신고가 돌파) 모멘텀 전략** 기반의 **실시간 모의 투자(Paper Trading) 시스템** 및 **백테스트 분석 대시보드**입니다.

GitHub 및 Streamlit Community Cloud에 즉시 배포하여 웹 브라우저에서 편리하게 구동할 수 있습니다.

---

## 🌟 주요 기능

### 1. 실전 1시간봉 모의투자 & 매매일지 시스템 (`paper_trading_app.py`)
- **기본 자본 `₩10,000,000` (1,000만원) & 실시간 원/달러 환율 연동**:
  - Yahoo Finance 실시간 환율(`KRW=X`)을 반영하여 원화 예수금/총 평가자산/수익률을 계산합니다.
- **실시간 1시간봉 API 대량 스캔 (150+ 종목)**:
  - 버튼 한 번으로 Yahoo Finance 최신 1시간봉(`90d, 1h`) 데이터를 직접 조회해 조건 검사:
    - 10일 거시선(65봉) > 30일 거시선(195봉) 골든크로스
    - 주가 > 200시간선(200봉) 장기 추세 안착
    - 시간당 변동성 $\text{ATR}(14) \ge 1.1\%$ (저변동 소외주 원천 차단)
    - 최근 60봉(약 10거래일) 최고가 대비 -6% 이내 돌파권 포착
- **최대 20종목 균등 분할 & 소수점 거래 불허용 (정수 1주 이상)**:
  - 1,000만원을 최대 20종목에 균등 배분(종목당 약 50만원)하며, 슬롯 예산 초과 고가주는 자동 건너뛰고 실제 매수 가능한 적격 종목을 우선 매수합니다.
  - 편도 0.1% (왕복 0.2%) 실거래 수수료 필수 차감
- **1시간봉 캔들 차트 뷰어 & 실전 매매일지**:
  - 종목별 실시간 1시간봉 캔들스틱, 10일/30일/200시간선, ATR 차트 제공
  - 완료된 모든 거래의 **매수 이유, 매도 이유, 진입/청산 지표, 승률, 수익률, Profit Factor, 회고 메모** 저장 및 CSV/JSON 백업 지원
- **엄격한 리스크 관리 & 영구 보존**:
  - **-12% 와이드 하드스탑** (개장 갭하락 시 시가 체결) + **10/30일선 데드크로스 전량 매도**
  - 로컬 JSON(`data/paper_portfolio.json`) 자동 저장 + JSON 백업/복원 + Streamlit Secrets(`GITHUB_TOKEN`) 설정 시 GitHub 저장소 자동 동기화

### 2. 백테스트 & AI 전략 분석 대시보드 (`app.py`)
- 298개 전 종목 리더보드, 인터랙티브 캔들스틱 매수/매도 타점 차트
- AI 심사역(Adversarial Critic) 평가 의견 및 파이썬 알고리즘 검증

---

## 🚀 빠른 시작 가이드 (Local Run)

### 1. 가상환경 활성화 및 패키지 설치
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. 실전 모의 투자 앱 실행
```bash
streamlit run paper_trading_app.py
```
브라우저에서 `http://localhost:8501`이 열리며 즉시 모의 투자를 시작할 수 있습니다.

### 3. 백테스트 대시보드 실행
```bash
streamlit run app.py
```

### 4. 터미널 퀀트 스크리너 및 백테스트 실행
```bash
# 실시간 챔피언 DNA 스크리너 실행
python3 run_screener.py --top 20

# 챔피언 DNA 조건 도달 종목 2년 과거 백테스트
python3 run_dna_backtest.py --group qualified
```

---

## 🌐 GitHub 푸시 및 Streamlit Cloud 배포 가이드

본 프로젝트는 GitHub 및 Streamlit Community Cloud에 즉시 연동되도록 구조화되어 있습니다.

### Step 1. 로컬 Git 리포지토리 초기화 및 커밋
```bash
git init
git add .
git commit -m "feat: Initial commit - Champion DNA Paper Trading & Backtesting System"
```

### Step 2. GitHub 원격 저장소 연결 및 푸시
```bash
# GitHub에서 새 repository 생성 후 (예: stock-champion-paper-trader)
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<REPO_NAME>.git
git push -u origin main
```

### Step 3. Streamlit Community Cloud 배포 (무료)
1. [share.streamlit.io](https://share.streamlit.io/)에 접속하여 GitHub 계정으로 로그인합니다.
2. **"Create app"** 클릭
3. **Repository**: `jeaweon/stock-2` 선택
4. **Branch**: `main`
5. **Main file path**: `paper_trading_app.py`
6. (선택) **Settings → Secrets**에 아래를 입력하면 클라우드 재부팅 시에도 매매일지가 GitHub에 자동 저장됩니다:
   ```toml
   GITHUB_TOKEN = "ghp_your_personal_access_token"
   GITHUB_REPO = "jeaweon/stock-2"
   ```
7. **"Deploy!"** 클릭 → 1~2분 내 전 세계 어디서나 접속 가능한 모의투자 웹 앱이 구동됩니다.

---

## 🧪 단위 테스트 검증
```bash
python3 -m unittest discover tests
```
총 35개의 단위 테스트(1,000만원 KRW 모의투자, 20종목 정수 분할 및 예산초과 스킵, 매매일지 이유/메모 보존, 갭하락 손절, 실시간 배치 추출, 백테스터)가 100% 통과합니다.

---

## 📁 디렉터리 구조
```
├── paper_trading_app.py       # 실전 모의 투자 Streamlit 메인 앱
├── app.py                     # 백테스트 & AI 심사 대시보드
├── run_screener.py            # 실시간 챔피언 DNA 퀀트 스크리너 CLI
├── run_dna_backtest.py        # 챔피언 DNA 과거 데이터 백테스트 CLI
├── requirements.txt           # 의존성 패키지 목록
├── .gitignore                 # 캐시/비밀키/가상환경 제외 설정
├── data/
│   ├── paper_portfolio.json   # 모의 투자 잔고 및 거래 이력 (영구 저장)
│   └── cache/                 # 1시간 봉 로컬 캐시 디렉터리
├── src/
│   ├── paper_trader.py        # 모의 투자 체결/수수료/포트폴리오 엔진
│   ├── screener.py            # 4대 챔피언 DNA 평가 및 랭킹 모듈
│   ├── timeframes.py          # 시간봉/일봉 EMA 엄밀 변환기
│   ├── backtester.py          # 갭하락 시가체결 & 수수료 반영 백테스터
│   ├── data_loader.py         # yfinance 속도제한 방어 & 로컬 캐시 로더
│   └── strategy_presets.py    # 공식 승인 1, 2, 3호 전략 프리셋
└── tests/                     # 31개 단위 테스트
```
