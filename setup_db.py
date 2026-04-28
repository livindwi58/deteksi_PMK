"""
Setup script untuk initialize MySQL Database
Jalankan: python setup_db.py
"""

import os
import sys
from dotenv import load_dotenv
import pymysql
from pymysql import MySQLError

# Load environment variables
load_dotenv()

DB_HOST = os.environ.get('DB_HOST', 'localhost')

DB_PORT = os.environ.get('DB_PORT', '3306')
try:
    DB_PORT = int(DB_PORT)
except:
    DB_PORT = 3306

DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_NAME = os.environ.get('DB_NAME', 'deteksi_pmk')

def create_database():
    """Create database if not exists"""
    try:
        # Connect to MySQL server (without specifying database)
        if DB_PASSWORD:
            conn = pymysql.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                charset='utf8mb4'
            )
        else:
            conn = pymysql.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                charset='utf8mb4'
            )
        
        cursor = conn.cursor()
        
        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"✓ Database '{DB_NAME}' created/verified")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    except MySQLError as e:
        print(f"✗ Error creating database: {e}")
        return False


def init_tables():
    """Initialize database tables using SQLAlchemy"""
    try:
        # Import after ensuring database exists
        from utils.mysql_db import init_mysql_tables
        
        init_mysql_tables()
        print("✓ Database tables created/verified")
        return True
    except Exception as e:
        print(f"✗ Error initializing tables: {e}")
        return False


def main():
    print("=" * 60)
    print("MySQL Database Setup for Deteksi PMK")
    print("=" * 60)
    print()
    print(f"Configuration:")
    print(f"  Host: {DB_HOST}")
    print(f"  Port: {DB_PORT}")
    print(f"  User: {DB_USER}")
    print(f"  Database: {DB_NAME}")
    print()
    
    # Step 1: Create database
    print("Step 1: Creating database...")
    if not create_database():
        print("\n✗ Setup failed. Please check your MySQL connection settings in .env")
        sys.exit(1)
    
    print()
    
    # Step 2: Initialize tables
    print("Step 2: Initializing tables...")
    if not init_tables():
        print("\n✗ Setup failed. Please check your database configuration.")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("✓ Database setup completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Make sure database is running")
    print("2. Run: python app.py")
    print()


if __name__ == '__main__':
    main()
