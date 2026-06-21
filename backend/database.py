import os
import time

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = None

for i in range(15):
    try:
        print(f"⏳ API waiting DB {i + 1}")
        engine = create_engine(DATABASE_URL)
        conn = engine.connect()
        conn.close()
        print("✅ API connected DB")
        break
    except Exception:
        time.sleep(2)

if engine is None:
    raise RuntimeError("DB not reachable")

SessionLocal = sessionmaker(bind=engine)
