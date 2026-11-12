# 🚀 미국 주식 챔피언 DNA 모의 투자 & 백테스트 분석 시스템

298개 미국 주식 전수 2년치(최대 730일) 1시간 봉 실증 백테스팅을 통해 도출된 **고수익 챔피언 DNA(변동성 파워 + 거시 정배열 + 신고가 돌파) 모멘텀 전략** 기반의 **실시간 모의 투자(Paper Trading) 시스템** 및 **백테스트 분석 대시보드**입니다.

GitHub 및 Streamlit Community Cloud에 즉시 배포하여 웹 브라우저에서 편리하게 구동할 수 있습니다.

---

## 🌟 주요 기능

### 1. 실전 모의 투자 시스템 (`paper_trading_app.py`)
- **실시간 시장 스캔 & 시그널 포착**:
  - 10일 거시선(65봉) > 30일 거시선(195봉) 골든크로스
  - 주가 > 200시간선(200봉) 장기 추세 안착
  - 시간당 변동성 $\text{ATR}(14) \ge 1.1\%$ (저변동 소외주 원천 차단)
  - 최근 60봉(약 10거래일) 최고가 대비 -6% 이내 돌파권 포착
- **원클릭 수동 매수 & 100% 자동 매매 모드 지원**:
  - 포트폴리오 슬롯(기본 5개) 및 투자 비중 자동 분할 계산
  - 편도 0.1% (왕복 0.2%) 실거래 수수료 필수 차감
- **엄격한 리스크 관리**:
  - **-12% 와이드 하드스탑**: 개장 첫 봉 오버나이트 갭하락 시 지정가가 아닌 '개장 시가(Open)'로 청산 반영
  - **거시 추세 청산**: 10일선 $\le$ 30일선 데드크로스 발생 시 전량 청산 (하락장 100% 현금 방어)
- **실시간 포트폴리오 상태 영구 저장 (`data/paper_portfolio.json`)**:
  - 보유 종목 수량, 매수가, 현재가, 미실현 손익, 손절기준가 실시간 추적
  - 완료된 거래 내역, 실전 승률, 손익비(Profit Factor), 자산 변화 곡선(Equity Curve) 시각화

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
3. **Repository**: 방금 푸시한 저장소 선택
4. **Branch**: `main`
5. **Main file path**: `paper_trading_app.py`
6. **"Deploy!"** 클릭 $\rightarrow$ 1~2분 내 전 세계 어디서나 접속 가능한 모의투자 웹 앱이 구동됩니다.

---

## 🧪 단위 테스트 검증
```bash
python3 -m unittest discover tests
```
총 31개의 단위 테스트(모의 투자 엔진 잔고/체결/수수료 계산, 갭하락 손절, 데이터 로더, 백테스터, 스토리지)가 100% 통과합니다.

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
