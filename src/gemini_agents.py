"""Dual-Gemini AI Agent System: Strategy Optimizer & Adversarial Critic."""
import json
import os
import re
from typing import Any, Callable, Dict, Optional, Tuple
import pandas as pd

from src.timeframes import TERMINOLOGY_PROMPT

# 영구 폐기 판정 기준: 최소 검증 종목 수 (프로젝트 전역 통일)
MIN_UNIVERSE_SIZE: int = 15

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def extract_python_code(text: str) -> str:
    """Extract python code from markdown code block or raw text."""
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    # If no markdown fences, check if text defines generate_signals
    if "def generate_signals" in text:
        return text.strip()
    return text.strip()


def compile_strategy_code(code_str: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Safely compile and return the generate_signals function from code string."""
    clean_code = extract_python_code(code_str)
    
    # Safe namespace with pandas and numpy
    import numpy as np
    scope: Dict[str, Any] = {
        "pd": pd,
        "np": np,
    }
    
    try:
        exec(clean_code, scope)
    except Exception as e:
        raise ValueError(f"전략 코드 문법/컴파일 에러: {str(e)}")

    if "generate_signals" not in scope or not callable(scope["generate_signals"]):
        raise ValueError("코드 내에 'def generate_signals(df: pd.DataFrame) -> pd.DataFrame:' 함수가 정의되어 있지 않습니다.")

    # Validate function on a mini test dataframe
    test_df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0] * 5,
        "High": [101.0, 102.0, 103.0, 104.0, 105.0] * 5,
        "Low": [99.0, 100.0, 101.0, 102.0, 103.0] * 5,
        "Close": [100.5, 101.5, 102.5, 103.5, 104.5] * 5,
        "Volume": [1000] * 25,
    }, index=pd.date_range("2024-01-01", periods=25, freq="1h"))

    try:
        res_df = scope["generate_signals"](test_df.copy())
        if "signal" not in res_df.columns:
            raise ValueError("반환된 데이터프레임에 'signal' 열이 없습니다.")
    except Exception as e:
        raise ValueError(f"전략 함수 실행 검증 실패: {str(e)}")

    return scope["generate_signals"]


def compile_sizing_code(code_str: str) -> Optional[Callable[[pd.DataFrame], pd.Series]]:
    """Optimizer가 생성한 포지션 사이징 수식 함수 `compute_position_size(df) -> pd.Series`를 컴파일.

    함수가 정의되지 않았으면 None 반환. 반환된 함수는 백테스터에서 루프 전 1회 벡터 평가되며,
    봉마다 Gemini API를 호출하지 않습니다.
    """
    clean_code = extract_python_code(code_str)
    if "def compute_position_size" not in clean_code:
        return None

    import numpy as np
    scope: Dict[str, Any] = {"pd": pd, "np": np}
    try:
        exec(clean_code, scope)
    except Exception as e:
        raise ValueError(f"포지션 사이징 코드 컴파일 에러: {str(e)}")

    func = scope.get("compute_position_size")
    if not callable(func):
        return None

    test_df = pd.DataFrame({
        "Open": [100.0 + i for i in range(30)],
        "High": [101.0 + i for i in range(30)],
        "Low": [99.0 + i for i in range(30)],
        "Close": [100.5 + i for i in range(30)],
        "Volume": [1000] * 30,
    }, index=pd.date_range("2024-01-01", periods=30, freq="1h"))
    try:
        out = func(test_df.copy())
        out = pd.Series(out, index=test_df.index) if not isinstance(out, pd.Series) else out
        if len(out) != len(test_df):
            raise ValueError("반환 길이가 입력 df와 다릅니다.")
    except Exception as e:
        raise ValueError(f"포지션 사이징 함수 실행 검증 실패: {str(e)}")
    return func


class DualGeminiAgents:
    """Manages Optimizer Agent and Adversarial Critic Agent."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.8-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None

        if self.api_key and GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def is_api_ready(self) -> bool:
        return self.client is not None

    def _call_gemini(self, system_instruction: str, user_prompt: str, temperature: float = 0.5) -> str:
        """Call Gemini API using interactions or models endpoint with fallback."""
        if not self.client:
            raise RuntimeError("Gemini API 클라이언트가 초기화되지 않았습니다. API 키를 확인해주세요.")

        candidate_models = [self.model_name, "gemini-2.5-flash", "gemini-1.5-flash"]
        last_error = None

        for model in candidate_models:
            try:
                # First try client.interactions.create (Current SDK interactions pattern)
                try:
                    interaction = self.client.interactions.create(
                        model=model,
                        input=f"{system_instruction}\n\n사용자 요청:\n{user_prompt}",
                    )
                    if hasattr(interaction, "output_text") and interaction.output_text:
                        return interaction.output_text
                except Exception:
                    pass

                # Fallback to client.models.generate_content
                response = self.client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                    ) if hasattr(types, "GenerateContentConfig") else None,
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"Gemini API 호출 실패: {str(last_error)}")

    def parse_user_strategy_to_code(self, user_strategy_text: str) -> Dict[str, Any]:
        """Convert user's natural language strategy into executable python signals function."""
        if not self.is_api_ready():
            return self._mock_initial_strategy(user_strategy_text)

        system_prompt = (
            "당신은 퀀트 트레이딩 알고리즘 엔지니어입니다. "
            "사용자의 자연어 매수/매도 전략을 분석하여 1시간 봉 데이터에서 동작하는 Python 함수를 작성해야 합니다.\n"
            "반드시 아래 규격을 엄격히 준수하세요:\n"
            "1. 함수 시그니처: `def generate_signals(df: pd.DataFrame) -> pd.DataFrame:`\n"
            "2. 입력 df의 열: ['Open', 'High', 'Low', 'Close', 'Volume'] (DatetimeIndex)\n"
            "3. df에 반드시 'signal' 열 추가: 매수 시그널은 1, 매도 시그널은 -1, 홀드는 0\n"
            "4. df에 'position_size' 열 추가: AI가 시장 변동성(ATR 등)이나 신호 강도를 분석해 0.1~1.0 사이의 자본 비중을 결정\n"
            "5. 필요시 'stop_loss_pct'(예: 0.03), 'take_profit_pct'(예: 0.05) 열 추가\n"
            "6. 판다스 경고(SettingWithCopyWarning)가 발생하지 않도록 df = df.copy() 후 처리\n"
            "7. 지표 계산 시 lookahead bias(미래 데이터 참조) 절대 금지 (shift(1) 또는 과거 데이터만 참조)\n\n"
            + TERMINOLOGY_PROMPT + "\n"
            "응답은 반드시 JSON 코드블록 없이 순수 JSON 형식 또는 아래 태그 형식으로 응답하세요:\n"
            "<SUMMARY>\n"
            "매수 조건: ...\n"
            "매도 조건: ...\n"
            "투자 비중: ...\n"
            "</SUMMARY>\n"
            "```python\n"
            "# 파이썬 코드\n"
            "```"
        )

        resp_text = self._call_gemini(
            system_instruction=system_prompt,
            user_prompt=f"사용자 매수/매도 전략:\n{user_strategy_text}",
            temperature=0.3,
        )

        code = extract_python_code(resp_text)
        func = compile_strategy_code(code)

        summary_match = re.search(r"<SUMMARY>([\s\S]*?)</SUMMARY>", resp_text)
        summary = summary_match.group(1).strip() if summary_match else user_strategy_text

        return {
            "code": code,
            "func": func,
            "summary": summary,
            "raw_response": resp_text,
        }

    def propose_improved_strategy(
        self,
        original_strategy_desc: str,
        current_metrics: Dict[str, Any],
        critic_feedback: Optional[str] = None,
        iteration_round: int = 1,
    ) -> Dict[str, Any]:
        """Propose an improved strategy with python code to achieve higher returns & win rate."""
        if not self.is_api_ready():
            return self._mock_improved_strategy(iteration_round, current_metrics, critic_feedback)

        system_prompt = (
            "당신은 최고 수준의 헤지펀드 퀀트 전략 개선 AI (Optimizer)입니다.\n"
            "이전 백테스트 결과 지표와 심사역(Critic)의 비판 피드백을 깊이 분석하여, "
            "승률과 수익률을 개선하고 MDD(최대낙폭)를 낮추는 고도화된 전략을 제시하세요.\n"
            "규칙:\n"
            "1. 단순한 지표값 과최적화(커브피팅)는 Critic에게 즉시 탈락합니다. 경제적/시장 구조적 논리(추세 필터, 모멘텀 확인, 변동성 손절 등)를 도입하세요.\n"
            "2. AI 동적 투자 비중은 실시간 API 호출이 아닌 '수식 코드'로 제공하세요: "
            "`def compute_position_size(df: pd.DataFrame) -> pd.Series:` 함수를 같은 코드블록에 정의하여 "
            "ATR 기반 변동성 가중 등 순수 pandas/numpy 벡터 연산으로 0.1~1.0 비중 Series를 반환해야 합니다 "
            "(백테스터가 로컬 메모리에서 1회 평가). 미래 데이터 참조 금지.\n"
            "3. 함수 시그니처: `def generate_signals(df: pd.DataFrame) -> pd.DataFrame:`\n"
            "4. 반드시 df에 'signal'(1, -1, 0) 및 'position_size' 열이 존재해야 합니다.\n"
            f"5. 무작위 {MIN_UNIVERSE_SIZE}개 이상 다양한 자산군(Out-of-sample)에서 단순 보유(B&H) 평균을 넘어야 채택됩니다. "
            "특정 폭등주/레버리지 종목에만 맞춘 파라미터는 금지합니다.\n"
            + TERMINOLOGY_PROMPT +
            "6. 응답 포맷:\n"
            "<IMPROVEMENTS>\n"
            "- 개선 배경 및 원인 분석:\n"
            "- 구체적 개선 사항 (추가 필터, 지표, 손절 로직):\n"
            "- 동적 자금 관리(투자액 판단) 로직:\n"
            "</IMPROVEMENTS>\n"
            "```python\n"
            "# 파이썬 코드\n"
            "```"
        )

        user_content = (
            f"[전략 원본 또는 이전 버전 설명]\n{original_strategy_desc}\n\n"
            f"[현재 백테스트 성과 지표]\n{json.dumps(current_metrics, ensure_ascii=False, indent=2)}\n\n"
            f"[비판 심사역(Critic)의 피드백]\n{critic_feedback or '없음 (첫 번째 개선 라운드)'}\n\n"
            f"개선 라운드: {iteration_round}회차. 더 높은 수익률과 승률을 내면서 실전 리스크를 줄이는 개선 전략과 코드를 작성하세요."
        )

        resp_text = self._call_gemini(
            system_instruction=system_prompt,
            user_prompt=user_content,
            temperature=0.7,
        )

        code = extract_python_code(resp_text)
        func = compile_strategy_code(code)
        sizing_func = compile_sizing_code(code)

        imp_match = re.search(r"<IMPROVEMENTS>([\s\S]*?)</IMPROVEMENTS>", resp_text)
        improvements = imp_match.group(1).strip() if imp_match else "AI 자동 최적화 적용"

        return {
            "code": code,
            "func": func,
            "sizing_func": sizing_func,
            "improvements": improvements,
            "raw_response": resp_text,
        }

    def evaluate_with_adversarial_critic(
        self,
        strategy_desc: str,
        strategy_code: str,
        current_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Objective, skeptical risk evaluator. NEVER shows favoritism. Must be strict."""
        if not self.is_api_ready():
            return self._mock_critic_evaluation(current_metrics, baseline_metrics)

        system_prompt = (
            "당신은 독립적이고 냉철한 헤지펀드 리스크 심사역(Adversarial Critic) 제미나이입니다.\n"
            "절대 칭찬이나 편애를 하지 마십시오. 오직 실전 투자 가능성과 리스크 관점에서 가혹하고 객관적으로 평가해야 합니다.\n"
            "심사 기준:\n"
            f"1. 단순 보유(B&H) 평균 대비 초과 수익 창출 여부: {MIN_UNIVERSE_SIZE}개 이상의 다양한 종목을 검증했을 때, 전략의 평균 수익률이 단순 보유(Buy & Hold) 평균 수익률보다 낮으면 즉시 FAIL 처리하고 해당 조건을 영구 폐기해야 합니다.\n"
            "2. 과최적화(Overfitting) 위험: 특정 기간 1시간 봉 노이즈에 억지로 맞춘 규칙이나 파라미터가 있는가?\n"
            "   - 특정 폭등 주식/레버리지 종목에만 맞춰진 파라미터 과최적화(Curve Fitting)나 선택적 편향(Selection Bias)을 엄격히 감지하고 지적할 것.\n"
            f"   - 무작위 {MIN_UNIVERSE_SIZE}개 이상 다양한 자산군(Out-of-sample)에서 동일하게 유효하지 않으면 PASS를 주지 말 것. "
            f"검증 종목 수가 {MIN_UNIVERSE_SIZE}개 미만이거나 정보가 없으면 PASS 불가.\n"
            "3. 표본 수(거래 횟수)의 신뢰성: 2년치 1시간 봉에서 거래 횟수가 너무 적으면(예: 10회 미만) 통계적 우연에 불과함.\n"
            "4. MDD(최대낙폭) 및 손익비(Profit Factor): 승률만 높고 한번 물릴 때 폭망하는 구조인지 점검.\n"
            "5. 미래 참조 편향(Lookahead Bias): 코드에서 미래 봉을 참조하는 오류가 없는가?\n"
            "6. 동적 포지션 사이징 타당성: 무모한 베팅이나 비현실적인 투자 비중이 아닌가?\n"
            "7. 이평선 단위 오류: '20일 거시 EMA'(=130시간봉)와 '20시간봉 EMA'를 혼동한 코드가 없는가?\n\n"
            + TERMINOLOGY_PROMPT + "\n"
            "반드시 아래 형식으로 응답하세요:\n"
            "VERDICT: [PASS 또는 FAIL]\n"
            "SCORE: [0~100점]\n"
            "CRITIQUE:\n"
            "- 주요 취약점:\n"
            "- 과최적화 위험도:\n"
            "- 실전 운용 위험:\n"
            "ACTIONABLE_ADVICE:\n"
            "- 개선해야 할 점:\n"
        )

        user_content = (
            f"[심사 대상 전략 설명]\n{strategy_desc}\n\n"
            f"[전략 구현 코드]\n```python\n{strategy_code}\n```\n\n"
            f"[개선안 백테스트 성과 지표]\n{json.dumps(current_metrics, ensure_ascii=False, indent=2)}\n\n"
            f"[기준(원본) 전략 성과 지표]\n{json.dumps(baseline_metrics, ensure_ascii=False, indent=2)}\n\n"
            "이 개선안이 실제로 채택할 만한지 엄격히 심사하고 VERDICT(PASS/FAIL)를 판정하세요."
        )

        resp_text = self._call_gemini(
            system_instruction=system_prompt,
            user_prompt=user_content,
            temperature=0.2,
        )

        verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", resp_text, re.IGNORECASE)
        verdict = verdict_match.group(1).upper() if verdict_match else "FAIL"

        score_match = re.search(r"SCORE:\s*(\d+)", resp_text)
        score = int(score_match.group(1)) if score_match else (75 if verdict == "PASS" else 45)

        return {
            "verdict": verdict,
            "score": score,
            "full_critique": resp_text,
            "is_passed": verdict == "PASS",
        }

    # ---------------- Mock fallbacks for testing without API Key ---------------- #

    def _mock_initial_strategy(self, user_text: str) -> Dict[str, Any]:
        """Deterministic fallback strategy when API key is absent (Fast/Slow EMA + Dynamic ATR sizing)."""
        code = (
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "def generate_signals(df: pd.DataFrame) -> pd.DataFrame:\n"
            "    d = df.copy()\n"
            "    # 1. 이동평균선 산출 (단기 20시간봉 EMA, 장기 50시간봉 EMA - 일 기준 아님)\n"
            "    d['ema_fast'] = d['Close'].ewm(span=20, adjust=False).mean()\n"
            "    d['ema_slow'] = d['Close'].ewm(span=50, adjust=False).mean()\n"
            "    \n"
            "    # 2. 변동성(ATR) 산출 기반 동적 자금 관리 (투자액 판단)\n"
            "    high_low = d['High'] - d['Low']\n"
            "    high_close = (d['High'] - d['Close'].shift(1)).abs()\n"
            "    low_close = (d['Low'] - d['Close'].shift(1)).abs()\n"
            "    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)\n"
            "    atr = tr.rolling(14).mean()\n"
            "    norm_atr = (atr / d['Close']).fillna(0.02)\n"
            "    # 변동성이 높으면 비중 축소(0.3~0.5), 낮으면 비중 확대(0.8~1.0)\n"
            "    d['position_size'] = np.clip(1.0 - (norm_atr * 15.0), 0.3, 1.0)\n"
            "    \n"
            "    # 3. 매수/매도 시그널 생성 (골든크로스 매수, 데드크로스 매도)\n"
            "    d['signal'] = 0\n"
            "    prev_fast = d['ema_fast'].shift(1)\n"
            "    prev_slow = d['ema_slow'].shift(1)\n"
            "    buy_cond = (prev_fast <= prev_slow) & (d['ema_fast'] > d['ema_slow'])\n"
            "    sell_cond = (prev_fast >= prev_slow) & (d['ema_fast'] < d['ema_slow'])\n"
            "    d.loc[buy_cond, 'signal'] = 1\n"
            "    d.loc[sell_cond, 'signal'] = -1\n"
            "    \n"
            "    # 4. 손절/익절 (손절 3%, 익절 6%)\n"
            "    d['stop_loss_pct'] = 0.03\n"
            "    d['take_profit_pct'] = 0.06\n"
            "    return d\n"
        )
        return {
            "code": code,
            "func": compile_strategy_code(code),
            "summary": f"기본 파싱 전략: 20/50 EMA 골든크로스 매수 & 데드크로스 매도, ATR 기반 동적 비중 조절 (손절 3%, 익절 6%)",
            "raw_response": "Mock Strategy Initialized",
        }

    def _mock_improved_strategy(self, round_num: int, current_metrics: Dict[str, Any], feedback: Optional[str]) -> Dict[str, Any]:
        """Deterministic improved strategy fallback for testing."""
        # Round 1 adds RSI filter; Round 2 adds 200 EMA trend filter + trailing stop
        if round_num == 1:
            code = (
                "import pandas as pd\n"
                "import numpy as np\n\n"
                "def generate_signals(df: pd.DataFrame) -> pd.DataFrame:\n"
                "    d = df.copy()\n"
                "    d['ema20'] = d['Close'].ewm(span=20, adjust=False).mean()\n"
                "    d['ema60'] = d['Close'].ewm(span=60, adjust=False).mean()\n"
                "    \n"
                "    # RSI 필터 추가 (과열권 매수 방지)\n"
                "    delta = d['Close'].diff()\n"
                "    gain = (delta.where(delta > 0, 0)).rolling(14).mean()\n"
                "    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()\n"
                "    rs = gain / (loss + 1e-9)\n"
                "    d['rsi'] = 100 - (100 / (1 + rs))\n"
                "    \n"
                "    # 신호 생성: 골든크로스 + RSI 45~65 사이 건강한 모멘텀 구간에서만 진입\n"
                "    d['signal'] = 0\n"
                "    buy_cond = (d['ema20'] > d['ema60']) & (d['ema20'].shift(1) <= d['ema60'].shift(1)) & (d['rsi'] >= 45) & (d['rsi'] <= 65)\n"
                "    sell_cond = (d['ema20'] < d['ema60']) & (d['ema20'].shift(1) >= d['ema60'].shift(1))\n"
                "    d.loc[buy_cond, 'signal'] = 1\n"
                "    d.loc[sell_cond, 'signal'] = -1\n"
                "    \n"
                "    # AI 변동성 비중 조절 및 타이트한 손절\n"
                "    d['position_size'] = 0.75\n"
                "    d['stop_loss_pct'] = 0.025\n"
                "    d['take_profit_pct'] = 0.08\n"
                "    return d\n"
            )
            improvements = "1차 개선: RSI(14) 모멘텀 필터 도입(과매수 65 초과 시 추격매수 차단), 손절폭 2.5%로 축소 및 익절 8% 상향"
        else:
            code = (
                "import pandas as pd\n"
                "import numpy as np\n\n"
                "def generate_signals(df: pd.DataFrame) -> pd.DataFrame:\n"
                "    d = df.copy()\n"
                "    d['ema20'] = d['Close'].ewm(span=20, adjust=False).mean()\n"
                "    d['ema60'] = d['Close'].ewm(span=60, adjust=False).mean()\n"
                "    d['ema200'] = d['Close'].ewm(span=200, adjust=False).mean()\n"
                "    \n"
                "    # 200시간봉 EMA(약 30.8거래일) 추세 필터 적용 (역배열 하락장 매수 차단)\n"
                "    trend_up = d['Close'] > d['ema200']\n"
                "    \n"
                "    d['signal'] = 0\n"
                "    buy_cond = trend_up & (d['ema20'] > d['ema60']) & (d['ema20'].shift(1) <= d['ema60'].shift(1))\n"
                "    sell_cond = (d['ema20'] < d['ema60']) & (d['ema20'].shift(1) >= d['ema60'].shift(1))\n"
                "    d.loc[buy_cond, 'signal'] = 1\n"
                "    d.loc[sell_cond, 'signal'] = -1\n"
                "    \n"
                "    # 200선 상단 강한 추세장에서 적극 투자 비중(85%)\n"
                "    d['position_size'] = 0.85\n"
                "    d['stop_loss_pct'] = 0.03\n"
                "    d['take_profit_pct'] = 0.09\n"
                "    return d\n"
            )
            improvements = f"{round_num}차 개선: 200 EMA 장기 추세 필터 결합, 하락장 휩쏘 원천 차단 및 추세 추종 수익 극대화"

        # AI 생성 사이징 수식(코드) 예시: ATR 기반 변동성 역가중 (로컬 1회 평가)
        code += (
            "\n\ndef compute_position_size(df: pd.DataFrame) -> pd.Series:\n"
            "    high_low = df['High'] - df['Low']\n"
            "    high_close = (df['High'] - df['Close'].shift(1)).abs()\n"
            "    low_close = (df['Low'] - df['Close'].shift(1)).abs()\n"
            "    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)\n"
            "    norm_atr = (tr.rolling(14).mean() / df['Close']).fillna(0.02)\n"
            "    return np.clip(1.0 - norm_atr * 15.0, 0.3, 1.0)\n"
        )
        return {
            "code": code,
            "func": compile_strategy_code(code),
            "sizing_func": compile_sizing_code(code),
            "improvements": improvements,
            "raw_response": "Mock Improved Strategy",
        }

    def _mock_critic_evaluation(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic mock critic evaluation."""
        curr_return = current.get("total_return_pct", 0)
        curr_mdd = current.get("max_drawdown_pct", 100)
        curr_trades = current.get("total_trades", 0)

        # [영구 폐기 규칙] 15개 이상 종목 검증 지표가 있으면 B&H 평균 미달 시 즉시 FAIL
        universe_n = int(current.get("universe_count", 0) or 0)
        strat_avg = current.get("universe_avg_return_pct")
        bh_avg = current.get("universe_bh_avg_return_pct")
        if universe_n >= MIN_UNIVERSE_SIZE and strat_avg is not None and bh_avg is not None and strat_avg < bh_avg:
            feedback = (
                "VERDICT: FAIL\n"
                "SCORE: 20\n"
                "CRITIQUE:\n"
                f"- {universe_n}개 종목 검증 결과 전략 평균({strat_avg:+.2f}%)이 B&H 평균({bh_avg:+.2f}%)에 미달.\n"
                f"- {MIN_UNIVERSE_SIZE}개 이상 종목 B&H 평균 미달 규칙에 따라 즉시 FAIL 및 영구 폐기 대상.\n"
                "ACTIONABLE_ADVICE: 특정 종목 과최적화를 배제하고 다양한 자산군에서 유효한 구조적 규칙으로 재설계하십시오."
            )
            return {"verdict": "FAIL", "score": 20, "full_critique": feedback, "is_passed": False}

        # Realistic strict critique logic
        if curr_trades < 5:
            verdict = "FAIL"
            score = 40
            feedback = (
                "VERDICT: FAIL\n"
                "SCORE: 40\n"
                "CRITIQUE:\n"
                "- 거래 횟수가 너무 적음(5회 미만): 2년치 데이터에서 통계적으로 유의미하지 않으며 단순 우연에 기댄 결과일 확률이 높음.\n"
                "- 실전 적용 불가: 신호 발생 빈도가 극히 낮아 자본 회전율 저하.\n"
                "ACTIONABLE_ADVICE: 지나치게 엄격한 필터를 완화하여 최소 10회 이상의 거래 표본을 확보하십시오."
            )
        elif curr_return > baseline.get("total_return_pct", -999) and curr_mdd <= 25.0 and curr_trades >= 8:
            verdict = "PASS"
            score = 88
            feedback = (
                "VERDICT: PASS\n"
                "SCORE: 88\n"
                "CRITIQUE:\n"
                "- 기준 전략 대비 수익률과 승률이 유의미하게 개선되었음.\n"
                "- 최대낙폭(MDD)이 허용 범위(25% 이하) 내에서 안정적으로 제어됨.\n"
                "- 거래 횟수가 충분하여 통계적 신뢰성을 확보함.\n"
                "ACTIONABLE_ADVICE: 실전 운용 전 슬리피지 변동성을 감안하여 분할 체결을 권장함."
            )
        else:
            verdict = "FAIL"
            score = 55
            feedback = (
                "VERDICT: FAIL\n"
                "SCORE: 55\n"
                "CRITIQUE:\n"
                "- MDD 또는 손익비 개선 폭이 헤지펀드 실전 기준에 미흡함.\n"
                "- 횡보 구간에서의 불필요한 잦은 진입 손실이 여전히 관측됨.\n"
                "ACTIONABLE_ADVICE: 장기 추세 필터(200 EMA 등)를 결합하여 역추세 진입을 엄격히 배제하십시오."
            )

        return {
            "verdict": verdict,
            "score": score,
            "full_critique": feedback,
            "is_passed": verdict == "PASS",
        }
