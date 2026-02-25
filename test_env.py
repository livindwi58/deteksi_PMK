#!/usr/bin/env python
"""Quick test untuk environment variables dan MySQL connection"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("ENVIRONMENT VARIABLES TEST")
print("=" * 60)

variables = {
    'DB_HOST': 'localhost',
    'DB_PORT': '3306',
    'DB_USER': 'root',
    'DB_PASSWORD': '(hidden)',
    'DB_NAME': 'deteksi_pmk',
    'FLASK_SECRET': '(hidden)'
}

for var, default in variables.items():
    value = os.environ.get(var)
    if value:
        if 'PASSWORD' in var or 'SECRET' in var:
            print(f"✓ {var}: Set (hidden)")
        else:
            print(f"✓ {var}: {value}")
    else:
        print(f"✗ {var}: Not set (using implicit defaults)")

print()
print("=" * 60)
print("MYSQL CONNECTION TEST")
print("=" * 60)

try:
    from utils.mysql_db import get_engine
    engine = get_engine()
    with engine.connect() as conn:
        print("✓ Database connection: SUCCESS")
except Exception as e:
    print(f"✗ Database connection: FAILED")
    print(f"  Error: {e}")

print()
