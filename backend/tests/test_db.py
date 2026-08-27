from sqlalchemy import inspect

from app.db.database import engine, init_db


def test_init_db_creates_tables():
    init_db()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "analyses" in tables
    assert "analysis_details" in tables
