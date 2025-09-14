# app/services/db.py
from sqlalchemy import create_engine

DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"

def get_engine():
    return create_engine(DB_URL)
