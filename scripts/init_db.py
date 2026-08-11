#!/usr/bin/env python
"""Inicializa/verifica el esquema PostgreSQL+PostGIS.

El docker-compose ejecuta los .sql automáticamente en el primer arranque. Este
script sirve para (a) verificar la conexión y PostGIS, o (b) (re)aplicar los
.sql de forma idempotente contra una BD ya en marcha.

    python scripts/init_db.py --check     # solo verifica conexión + PostGIS
    python scripts/init_db.py --apply      # aplica sql/*.sql en orden
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seis_peru.config import settings  # noqa: E402


def check() -> int:
    from seis_peru.storage import db
    print("Conectando a:", settings.dsn().replace(settings.pg_password, "***"))
    print("PostGIS:", db.ping())
    return 0


def apply() -> int:
    import psycopg2
    sql_dir = ROOT / "sql"
    files = sorted(sql_dir.glob("*.sql"))
    print(f"Aplicando {len(files)} archivos de {sql_dir} ...")
    with psycopg2.connect(settings.dsn()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for f in files:
                print(f"  -> {f.name}")
                cur.execute(f.read_text(encoding="utf-8"))
    print("Esquema aplicado.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = p.parse_args()
    if args.apply:
        return apply()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
