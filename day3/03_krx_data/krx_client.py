"""
KRX 공식 API 클라이언트
data.krx.co.kr 로그인 후 pykrx를 통해 데이터를 가져옵니다.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 .env 로드
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_KRX_ID = os.getenv("KRX_ID", "").strip()
_KRX_PW = os.getenv("KRX_PW", "").strip()

if not (_KRX_ID and _KRX_PW):
    raise EnvironmentError(".env 파일에 KRX_ID, KRX_PW를 설정해주세요.")

os.environ["KRX_ID"] = _KRX_ID
os.environ["KRX_PW"] = _KRX_PW

from pykrx import stock  # noqa: E402  (env 설정 후 import)

__all__ = ["stock"]
