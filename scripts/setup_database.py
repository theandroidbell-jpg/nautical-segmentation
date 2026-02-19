"""
Database Setup Script

Reads and executes the SQL schema file to create the PostGIS database schema
and tables for the nautical segmentation system.

Features:
- Schema creation from SQL file
- Table existence verification
- Optional drop-and-recreate mode
- Verification of successful creation
"""

import argparse
import logging
import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_tables(conn) -> dict:
    """
    Verify that all expected tables exist in the database.
    
    Args:
        conn: Database connection
        
    Returns:
        dict: Mapping of table names to existence status
    """
    expected_tables = [
        'charts',
        'ground_truth',
        'predicted_polygons',
        'output_files',
        'processing_log',
        'models',
        'tiles'
    ]
    
    table_status = {}
    
    try:
        with conn.cursor() as cur:
            for table_name in expected_tables:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = %s 
                        AND table_name = %s
                    )
                    """,
                    (Config.DB_SCHEMA, table_name)
                )
                exists = cur.fetchone()[0]
                table_status[table_name] = exists
                
    except Exception as e:
        logger.error(f"Error verifying tables: {e}")
    
    return table_status


def drop_schema(conn):
    """
    Drop the schema and all its contents.
    
    WARNING: This is a destructive operation!
    
    Args:
        conn: Database connection
    """
    try:
        with conn.cursor() as cur:
            logger.warning(f"Dropping schema {Config.DB_SCHEMA} CASCADE...")
            cur.execute(f"DROP SCHEMA IF EXISTS {Config.DB_SCHEMA} CASCADE")
        conn.commit()
        logger.info(f"Schema {Config.DB_SCHEMA} dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop schema: {e}")
        conn.rollback()
        raise


def execute_sql_file(conn, sql_file: Path):
    """
    Execute SQL commands from a file.
    
    Args:
        conn: Database connection
        sql_file: Path to SQL file
    """
    try:
        logger.info(f"Reading SQL file: {sql_file}")
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        logger.info("Executing SQL commands...")
        with conn.cursor() as cur:
            cur.execute(sql_content)
        
        conn.commit()
        logger.info("SQL execution completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to execute SQL file: {e}")
        conn.rollback()
        raise


def main():
    """Main entry point for database setup script."""
    parser = argparse.ArgumentParser(
        description='Setup PostGIS database schema for nautical segmentation'
    )
    parser.add_argument(
        '--drop-existing',
        action='store_true',
        help='Drop existing schema before creating (DANGEROUS!)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify tables, do not create'
    )
    parser.add_argument(
        '--sql-file',
        type=Path,
        default=Path(__file__).parent.parent / 'sql' / 'schema.sql',
        help='Path to SQL schema file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("Database Setup Script")
    logger.info("=" * 60)
    logger.info(f"Database: {Config.DB_NAME}")
    logger.info(f"Schema: {Config.DB_SCHEMA}")
    logger.info(f"SQL file: {args.sql_file}")
    logger.info("=" * 60)
    
    # Check if SQL file exists
    if not args.sql_file.exists():
        logger.error(f"SQL file not found: {args.sql_file}")
        sys.exit(1)
    
    # Connect to database
    try:
        conn = Config.get_db_connection()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    try:
        if args.verify_only:
            # Only verify tables
            logger.info("Verifying tables...")
            table_status = verify_tables(conn)
            
            logger.info("=" * 60)
            logger.info("TABLE VERIFICATION")
            logger.info("=" * 60)
            
            all_exist = True
            for table_name, exists in table_status.items():
                status = "EXISTS" if exists else "MISSING"
                logger.info(f"{table_name}: {status}")
                if not exists:
                    all_exist = False
            
            logger.info("=" * 60)
            
            if all_exist:
                logger.info("All tables exist!")
                sys.exit(0)
            else:
                logger.warning("Some tables are missing")
                sys.exit(1)
        
        else:
            # Create schema
            if args.drop_existing:
                logger.warning("DROP EXISTING MODE ENABLED!")
                response = input("This will DELETE all data in the schema. Type 'yes' to continue: ")
                if response.lower() != 'yes':
                    logger.info("Aborted by user")
                    sys.exit(0)
                
                drop_schema(conn)
            
            # Execute SQL file
            execute_sql_file(conn, args.sql_file)
            
            # Verify tables were created
            logger.info("Verifying tables...")
            table_status = verify_tables(conn)
            
            logger.info("=" * 60)
            logger.info("TABLE VERIFICATION")
            logger.info("=" * 60)
            
            all_exist = True
            for table_name, exists in table_status.items():
                status = "✓ CREATED" if exists else "✗ MISSING"
                logger.info(f"{table_name}: {status}")
                if not exists:
                    all_exist = False
            
            logger.info("=" * 60)
            
            if all_exist:
                logger.info("Database setup completed successfully!")
            else:
                logger.error("Some tables were not created")
                sys.exit(1)
        
    finally:
        conn.close()
        logger.info("Database connection closed")


if __name__ == '__main__':
    main()
