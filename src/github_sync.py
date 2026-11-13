"""Optional GitHub Repository Sync for Streamlit Cloud Persistence.

Uses Python standard library (urllib) to read and commit `data/paper_portfolio.json`
to GitHub via the Contents API when `GITHUB_TOKEN` is configured in Streamlit Secrets
or environment variables. Falls back gracefully to local JSON persistence otherwise.
"""
import base64
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


DEFAULT_GITHUB_REPO = "jeaweon/stock-2"
DEFAULT_PORTFOLIO_REPO_PATH = "data/paper_portfolio.json"
DEFAULT_BRANCH = "main"


def _github_request(
    url: str,
    token: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "champion-dna-paper-trader",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data_bytes = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data_bytes = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def load_portfolio_from_github(
    token: str,
    repo: str = DEFAULT_GITHUB_REPO,
    file_path: str = DEFAULT_PORTFOLIO_REPO_PATH,
    branch: str = DEFAULT_BRANCH,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Fetch the latest portfolio JSON from GitHub Contents API."""
    if not token:
        return None, "GitHub 토큰 미설정"

    url = f"https://api.github.com/repos/{repo}/contents/{file_path}?ref={branch}"
    try:
        res = _github_request(url, token=token, method="GET")
        encoded = res.get("content", "")
        if not encoded:
            return None, "GitHub 파일 내용이 비어 있습니다."
        decoded_str = base64.b64decode(encoded).decode("utf-8")
        data = json.loads(decoded_str)
        return data, "GitHub에서 최신 계좌 및 매매일지를 불러왔습니다."
    except urllib.error.HTTPError as e:
        return None, f"GitHub 조회 실패 (HTTP {e.code})"
    except Exception as e:
        return None, f"GitHub 조회 오류: {e}"


def push_portfolio_to_github(
    portfolio: Dict[str, Any],
    token: str,
    repo: str = DEFAULT_GITHUB_REPO,
    file_path: str = DEFAULT_PORTFOLIO_REPO_PATH,
    branch: str = DEFAULT_BRANCH,
    commit_message: str = "chore: sync live paper portfolio & journal",
) -> Tuple[bool, str]:
    """Commit updated portfolio/journal JSON to GitHub via Contents API."""
    if not token:
        return False, "GitHub 토큰이 없어 로컬에만 저장되었습니다."

    api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    sha: Optional[str] = None
    try:
        current = _github_request(f"{api_url}?ref={branch}", token=token, method="GET")
        sha = current.get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            return False, f"GitHub SHA 확인 실패 (HTTP {e.code})"
    except Exception as e:
        return False, f"GitHub 연결 오류: {e}"

    json_str = json.dumps(portfolio, indent=2, ensure_ascii=False)
    content_b64 = base64.b64encode(json_str.encode("utf-8")).decode("ascii")
    payload: Dict[str, Any] = {
        "message": commit_message,
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    try:
        _github_request(api_url, token=token, method="PUT", payload=payload)
        return True, f"GitHub({repo}) 동기화 완료"
    except urllib.error.HTTPError as e:
        return False, f"GitHub 업로드 실패 (HTTP {e.code})"
    except Exception as e:
        return False, f"GitHub 업로드 오류: {e}"
