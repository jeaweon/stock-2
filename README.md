# 📈 미국 주식 1시간봉 모의투자 & 백테스트 통합 저장소

이 저장소는 목적에 따라 **완전히 분리된 2개의 독립 프로그램**으로 구성되어 있습니다.

---

## 1. 실시간 모의투자 & 매매일지 프로그램 (`paper_trading/`)
> [jeaweon/stock-](https://github.com/jeaweon/stock-) 실행 방식(원화 환산 · 장외 1분봉 실시간 반영 · GitHub Gist 영구 보존) + **챔피언 DNA 4단계 스크리너** + **유망 매매 방법 3호(1시간봉)**

- **기본 자본 / 분할**: `10,000,000원` / 최대 20종목 균등 분할 / **정수 1주 이상 매수(소수점 불가)** / 편도 **0.1%** 수수료
- **종목 검색 & 매매**: 상단 버튼 클릭 시 157개 미국 주식/ETF의 1시간봉 API를 스캔하여 `BUY_READY` 적격 종목을 자동 매수하고, `-12%` 하드스탑 및 `10/30일선 데드크로스` 시 자동 매도
- **매매일지**: 매수·매도 상세 이유, 수익률, 승률, 손익비(PF), 전략별 성과 통계, CSV/JSON 백업 및 **Gist 클라우드 자동 동기화**

### 실행 및 Streamlit Cloud 배포
```bash
cd paper_trading
pip install -r requirements.txt
streamlit run app.py
```
- **Streamlit Cloud 배포**: Main file path를 `paper_trading/app.py`로 지정하고 `Settings → Secrets`에 `GIST_ID`, `GITHUB_TOKEN`을 설정하면 계좌 상태가 클라우드에 영구 보존됩니다.

---

## 2. 과거 데이터 백테스트 & AI 전략 연구 시스템 (`backtest/`)
> 298개 미국 주식 2년치 1시간 봉 백테스팅, Dual-Gemini AI 전략 최적화 & 적대적 심사역 검증기, 유망 매매 방법 1·2·3호 및 DNA 스크리너 연구실

```bash
cd backtest
pip install -r requirements.txt
streamlit run app.py
```

### 단위 테스트 실행
```bash
# 모의투자 엔진 테스트
cd paper_trading && python3 -m unittest discover tests -v

# 백테스트 엔진 테스트
cd backtest && python3 -m unittest discover tests -v
```
