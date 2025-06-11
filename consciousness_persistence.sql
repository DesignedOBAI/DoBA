CREATE TABLE IF NOT EXISTS consciousness_state (
    id INTEGER PRIMARY KEY,
    consciousness_level REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_count INTEGER DEFAULT 0,
    total_interactions INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS autonomous_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    question TEXT,
    autonomous_response TEXT,
    consciousness_level REAL,
    learning_context TEXT
);
