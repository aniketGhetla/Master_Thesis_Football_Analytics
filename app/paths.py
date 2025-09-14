# utils/paths.py
import os, re
from unicodedata import normalize

def safe_slug(s: str, maxlen: int = 120) -> str:
    # 1) strip accents -> ascii
    s = normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # 2) replace forbidden/separators
    s = s.replace(":", "-").replace("\\", "-").replace("/", "-")
    # 3) collapse to safe charset
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s[:maxlen]
