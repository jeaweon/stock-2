# ⚡ 1시간봉 챔피언 DNA 실전 모의투자 프로그램 (`paper_trading/`)

기존 백테스트(`../backtest/`)와 완전히 독립된 실시간 모의투자 및 매매일지 애플리케이션입니다.  
실행 방식·원화 환산·GitHub Gist 영구 보존 흐름은 [jeaweon/stock-](https://github.com/jeaweon/stock-) 구성을 따르며, 매수·매도 및 종목 발굴에는 **1시간 봉 챔피언 DNA 4단계 스크리너**와 **유망 매매 방법 3호**를 사용합니다.

## 주요 규격
- **초기 자본**: `10,000,000 원` (1,000만원)
- **분산 투자 & 주문 단위**: 최대 20종목 균등 분할, **소수점 불가 (1주 이상 정수 매수만 허용)**, 편도 **0.1%** 수수료
- **시간 기준**: 1시간 봉 (`90d, 1h`) + 장외(프리/애프터/데이장) 1분봉 실시간 가격(`prepost=True`) + 시간별 원/달러 환율(`KRW=X`) 원화 환산
- **종목 검색 (스크리너)**: 157개 미국 주식/ETF 대상 4단계 DNA 점수화 (200시간선 상단 + 10/30일 정배열 + ATR 파워 + 오버나이트 갭 빈도 + 60봉 신고가 근접)
- **매수/매도 전략 (유망 매매 방법 3호)**:
  - 매수: `10일선(65h) > 30일선(195h)` & `주가 > 200시간선` & `ATR(14) >= 1.1%` & `60봉 최고가 대비 -6% 이내` (`BUY_READY`)
  - 매도: `10일선 <= 30일선` 데드크로스 시 전량 매도
  - 손절: `-12%` 하드스탑 (갭하락 시 개장 시가 체결 반영)

## 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud & GitHub Gist 영구 보존 + 1시간 자동매매 설정
### 1) Streamlit Cloud 연동
저장소 루트의 `app.py`가 자동으로 `paper_trading/app.py`를 실행하므로 `Main file path`가 `app.py` 또는 `paper_trading/app.py` 어느 쪽이든 동작합니다. **Settings → Secrets**에 아래 2줄을 입력하면 계좌 상태(`trader_state.json`)와 매매일지가 Gist에 영구 저장됩니다.
```toml
GIST_ID = "여기에_Gist_ID_입력"
GITHUB_TOKEN = "ghp_여기에_토큰_입력"
```

### 2) 24시간 무인 1시간봉 자동매매 (GitHub Actions)
`.github/workflows/hourly_paper_trade.yml`이 매시간(`cron: "5 * * * *"`) `paper_trading/auto_trade.py`를 자동 실행합니다.  
GitHub 저장소의 **Settings → Secrets and variables → Actions → New repository secret**에 아래 2개를 등록하세요 (`GITHUB_` 접두사는 GitHub 예약어이므로 토큰 이름은 반드시 `GIST_TOKEN`으로 등록):
- `GIST_ID`: 본인의 Gist ID
- `GIST_TOKEN`: Gist 읽기/쓰기(`gist`) 권한이 있는 GitHub Personal Access Token

브라우저를 열어둔 상태에서 웹 대시보드 좌측 사이드바의 **⏰ 웹 1시간 자동 갱신·매매** 토글도 함께 켜실 수 있습니다.
