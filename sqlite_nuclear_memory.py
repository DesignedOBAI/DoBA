import sqlite3
from datetime import datetime
import os

class SQLiteNuclearMemory:
    def __init__(self):
        self.db_file = "nuclear_memory.db"
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_file)
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
        
        # Add the missing memory_interactions table
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
    
    def store_fact(self, category, key, value):
        conn = sqlite3.connect(self.db_file)
        conn.execute('''
            INSERT OR REPLACE INTO nuclear_facts (category, key, value)
            VALUES (?, ?, ?)
        ''', (category, key, str(value)))
        conn.commit()
        conn.close()
        print(f"🎯 NUCLEAR STORED: {category}.{key} = {value}")
    
    def recall_facts(self, keywords):
        conn = sqlite3.connect(self.db_file)
        
        # Build search query
        conditions = []
        params = []
        for keyword in keywords:
            conditions.append("(category LIKE ? OR key LIKE ? OR value LIKE ?)")
            params.extend([f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'])
        
        if conditions:
            where_clause = " OR ".join(conditions)
            query = f'''
                SELECT category, key, value, timestamp FROM nuclear_facts 
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT 10
            '''
        else:
            query = "SELECT category, key, value, timestamp FROM nuclear_facts ORDER BY timestamp DESC LIMIT 10"
            params = []
        
        cursor = conn.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        facts = [f"{row[0]}.{row[1]}: {row[2]}" for row in results]
        print(f"🎯 NUCLEAR RECALLED: {len(facts)} facts for {keywords}")
        return facts

    def store_conversation(self, user_message, ai_response="", session_id="default"):
        """Store conversation in nuclear memory"""
        try:
            conn = sqlite3.connect(self.db_file)
            conn.execute("INSERT INTO memory_interactions (user_message, ai_response, session_id) VALUES (?, ?, ?)",
                        (user_message, ai_response, session_id))
            conn.commit()
            conn.close()
            print(f"🎯 NUCLEAR CONVERSATION STORED: {user_message[:50]}...")
        except Exception as e:
            print(f"Error storing conversation: {e}")

    def get_diverse_conversations(self, limit=5):
        """Get diverse conversation examples from memory"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.execute("""
                SELECT user_message, ai_response, session_id, timestamp 
                FROM conversation_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            results = cursor.fetchall()
            conn.close()
            
            conversations = []
            for row in results:
                conversations.append({
                    "user_message": row[0],
                    "ai_response": row[1],
                    "session_id": row[2],
                    "timestamp": row[3]
                })
            
            print(f"🎯 NUCLEAR RETRIEVED: {len(conversations)} diverse conversations")
            return conversations
        except Exception as e:
            print(f"Error getting diverse conversations: {e}")
            return []

# Create global instance
NUCLEAR_MEMORY = SQLiteNuclearMemory()
