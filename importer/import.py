import os
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources.import_csv import import_csv  # noqa: E402

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = None

# ✅ WAIT FOR DB
for i in range(15):
    try:
        print(f"⏳ Tentative connexion DB {i + 1}")
        engine = create_engine(DATABASE_URL)
        conn = engine.connect()
        conn.close()
        print("✅ DB CONNECTED")
        break
    except Exception:
        print("❌ DB not ready, retry...")
        time.sleep(2)

if engine is None:
    raise RuntimeError("❌ Impossible de se connecter à la DB")

import_csv()
