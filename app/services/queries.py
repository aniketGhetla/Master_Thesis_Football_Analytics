# app/services/queries.py
from sqlalchemy import text
from app.services.db import get_engine

def count_rows(table_name: str) -> int:
    try:
        with get_engine().connect() as conn:
            res = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            return int(res.scalar() or 0)
    except Exception:
        return 0  # table may not exist yet

def count_distinct_match_ids(table_name: str) -> int:
    try:
        with get_engine().connect() as conn:
            res = conn.execute(text(f"SELECT COUNT(DISTINCT match_id) FROM {table_name}"))
            return int(res.scalar() or 0)
    except Exception:
        return 0
    
    
