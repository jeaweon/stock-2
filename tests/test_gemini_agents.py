"""Unit tests for Dual Gemini Agents."""
import unittest
import pandas as pd
from src.gemini_agents import DualGeminiAgents, extract_python_code, compile_strategy_code


class TestDualGeminiAgents(unittest.TestCase):
    def setUp(self):
        self.agents = DualGeminiAgents(api_key=None)  # Uses deterministic mock fallback

    def test_extract_python_code(self):
        text_with_fences = "Here is the code:\n```python\ndef generate_signals(df):\n    return df\n```\nDone."
        code = extract_python_code(text_with_fences)
        self.assertIn("def generate_signals", code)
        self.assertNotIn("```", code)

    def test_compile_strategy_code_valid(self):
        sample_code = (
            "import pandas as pd\n"
            "def generate_signals(df: pd.DataFrame) -> pd.DataFrame:\n"
            "    d = df.copy()\n"
            "    d['signal'] = 1\n"
            "    d['position_size'] = 0.8\n"
            "    return d\n"
        )
        func = compile_strategy_code(sample_code)
        self.assertTrue(callable(func))

    def test_compile_strategy_code_missing_signal_col_raises(self):
        invalid_code = (
            "import pandas as pd\n"
            "def generate_signals(df: pd.DataFrame) -> pd.DataFrame:\n"
            "    return df.copy()\n"  # missing 'signal' column
        )
        with self.assertRaises(ValueError):
            compile_strategy_code(invalid_code)

    def test_parse_user_strategy_mock(self):
        parsed = self.agents.parse_user_strategy_to_code("골든크로스 매수")
        self.assertIn("code", parsed)
        self.assertTrue(callable(parsed["func"]))
        self.assertIn("summary", parsed)

    def test_optimizer_and_critic_loop_mock(self):
        # 1. Base metrics
        base_metrics = {"total_return_pct": 5.0, "max_drawdown_pct": 12.0, "total_trades": 15}

        # 2. Optimizer generates improvement
        opt = self.agents.propose_improved_strategy(
            original_strategy_desc="20/50 골든크로스",
            current_metrics=base_metrics,
            critic_feedback=None,
            iteration_round=1,
        )
        self.assertIn("improvements", opt)
        self.assertTrue(callable(opt["func"]))

        # 3. Critic evaluates improvement
        improved_metrics = {"total_return_pct": 15.0, "max_drawdown_pct": 8.0, "total_trades": 20}
        critic = self.agents.evaluate_with_adversarial_critic(
            strategy_desc=opt["improvements"],
            strategy_code=opt["code"],
            current_metrics=improved_metrics,
            baseline_metrics=base_metrics,
        )
        self.assertIn(critic["verdict"], ["PASS", "FAIL"])
        self.assertIn("CRITIQUE", critic["full_critique"])


if __name__ == "__main__":
    unittest.main()
