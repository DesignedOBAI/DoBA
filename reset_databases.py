#!/usr/bin/env python3
"""
DoBA Database Reset Script
Safely clears all existing data and reinitializes fresh databases
"""
import sqlite3
import os
import shutil
from datetime import datetime

def backup_existing_databases():
    """Backup existing databases before reset"""
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)

    db_files = [
        'ai_memory.db', 'consciousness_memory.db', 'emotional_memory.db',
        'intelligent_memory.db', 'nuclear_memory.db', 'nuclear_memory_enhanced.db',
        'memory.db'
    ]

    for db_file in db_files:
        if os.path.exists(db_file):
            shutil.copy2(db_file, os.path.join(backup_dir, db_file))
            print(f"✅ Backed up {db_file}")

    return backup_dir

def reset_nuclear_memory():
    """Reset nuclear memory database with correct schema"""
    if os.path.exists('nuclear_memory.db'):
        os.remove('nuclear_memory.db')

    conn = sqlite3.connect('nuclear_memory.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS nuclear_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(category, key) ON CONFLICT REPLACE
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS memory_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            ai_response TEXT,
            session_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Reset nuclear_memory.db")

def reset_intelligent_memory():
    """Reset intelligent memory database with correct schema"""
    if os.path.exists('intelligent_memory.db'):
        os.remove('intelligent_memory.db')

    conn = sqlite3.connect('intelligent_memory.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS intelligent_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_login TEXT,
            category TEXT,
            key TEXT,
            value TEXT,
            original_context TEXT,
            embedding_vector TEXT,
            confidence_score REAL DEFAULT 0.9,
            semantic_tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_login, key)
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS memory_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_login TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_message TEXT,
            ai_response TEXT,
            response_type TEXT DEFAULT 'single',
            model_used TEXT,
            importance_score INTEGER DEFAULT 1,
            extracted_facts TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Reset intelligent_memory.db")

def reset_other_databases():
    """Reset other database files"""
    other_dbs = [
        'ai_memory.db', 'consciousness_memory.db', 'emotional_memory.db',
        'nuclear_memory_enhanced.db', 'memory.db'
    ]

    for db_file in other_dbs:
        if os.path.exists(db_file):
            os.remove(db_file)
            # Create empty database with basic structure
            conn = sqlite3.connect(db_file)
            conn.execute('CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
            conn.commit()
            conn.close()
            print(f"✅ Reset {db_file}")

if __name__ == "__main__":
    print("🧹 DoBA Database Reset Tool")
    print("=" * 40)
    print("This will remove all conversation history and learned facts.")
    print("Previous data will be backed up automatically.")
    print()

    # Confirm with user
    response = input("Continue with database reset? (y/N): ")
    if response.lower() != 'y':
        print("❌ Reset cancelled")
        exit()

    print("\n🚀 Starting database reset...")

    # Backup existing data
    backup_dir = backup_existing_databases()
    print(f"📦 Backup created in: {backup_dir}")

    # Reset databases
    reset_nuclear_memory()
    reset_intelligent_memory()
    reset_other_databases()

    print("\n✅ All databases reset successfully!")
    print("🚀 DoBA is ready for fresh use!")
    print(f"📦 Previous data safely backed up to: {backup_dir}")