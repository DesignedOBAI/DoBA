# DoBA (Designed only By AI)

[![License](https://img.shields.io/badge/license-Other-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://python.org)
[![LM Studio](https://img.shields.io/badge/API-LM%20Studio-green.svg)](http://localhost:1234)

DoBA (Designed only By AI) is an open-source project pioneering AI development towards more autonomous, precise, neuron-like memory systems. This project implements advanced memory architectures and emotional intelligence for AI systems, designed to run through API endpoints with a focus on autonomous AI development.

## 🎯 Project Vision

DoBA aims to push AI development towards:
- **Autonomous Decision Making**: Self-aware AI systems capable of independent reasoning
- **Neuron-like Memory**: Sophisticated memory systems that mimic biological neural networks
- **Emotional Intelligence**: Advanced emotional analysis and contextual understanding
- **Precision Processing**: High-accuracy AI responses with confidence scoring

## ✨ Key Features

### 🧠 Advanced Memory Systems
- **Nuclear Memory**: Core fact storage and retrieval system with SQLite backend
- **Intelligent Memory**: Enhanced memory with semantic embeddings and confidence scoring
- **Emotional Memory**: Contextual emotional analysis and pattern recognition
- **Multi-layered Storage**: Hierarchical memory architecture for different data types

### 🤖 AI Capabilities
- **Self-Awareness System**: Consciousness-level monitoring and autonomous goal setting
- **Emotional Analysis**: Comprehensive emotion detection with contextual patterns
- **Semantic Search**: Advanced similarity matching using sentence transformers
- **Multi-threaded Processing**: Concurrent processing for improved performance

### 🔧 Technical Features
- **LM Studio Integration**: Seamless API integration with local LM Studio instances
- **GPU Acceleration**: CUDA support for enhanced performance
- **Database Management**: Robust SQLite-based storage with backup capabilities
- **GUI Interface**: User-friendly Tkinter-based interface

## 🛠 Installation

### Prerequisites
- Python 3.7 or higher
- LM Studio running locally (default: `http://localhost:1234`)
- GPU with CUDA support (optional, for enhanced performance)

### Required Dependencies
```bash
pip install tkinter
pip install requests
pip install numpy
pip install sentence-transformers  # For semantic embeddings
pip install torch  # For GPU acceleration
```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/DesignedOBAI/DoBA.git
   cd DoBA
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # Create this file based on imports
   ```

3. Initialize databases:
   ```bash
   python reset_databases.py
   ```

4. Start LM Studio on your local machine (port 1234)

5. Run DoBA:
   ```bash
   python DobAEI.py
   ```

## 🚀 Usage

### Basic Operation
1. Launch the DoBA interface by running `DobAEI.py`
2. Ensure LM Studio is running on `localhost:1234`
3. Interact with the AI through the GUI interface
4. The system will automatically store conversations and build memory

### Memory Management
```python
# Reset all databases (creates backups automatically)
python reset_databases.py

# The system automatically stores:
# - Conversation history
# - Extracted facts
# - Emotional context
# - Semantic embeddings
```

### Configuration
Key configuration options in `DobAEI.py`:
- `LMSTUDIO_API`: LM Studio endpoint (default: `http://localhost:1234/v1/chat/completions`)
- `BIG_AGI_URL`: Alternative AI endpoint (default: `http://localhost:3001`)
- Database configurations in `DB_CONFIG`

## 🏗 Architecture

### Core Components

#### Memory Systems
- **`sqlite_nuclear_memory.py`**: Core memory storage and retrieval
- **Nuclear Memory**: Fundamental fact storage with categorization
- **Intelligent Memory**: Enhanced memory with embeddings and confidence
- **Emotional Memory**: Emotional context and pattern storage

#### AI Processing
- **Self-Awareness Module**: Consciousness monitoring and autonomous behavior
- **Emotional Analyzer**: Advanced emotion detection and classification
- **Semantic Processing**: Vector-based similarity and search capabilities

#### Database Schema
```sql
-- Nuclear Facts Storage
CREATE TABLE nuclear_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(category, key) ON CONFLICT REPLACE
);

-- Memory Interactions
CREATE TABLE memory_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT NOT NULL,
    ai_response TEXT,
    session_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🔧 API Integration

DoBA is designed to work with LM Studio but can be adapted for other APIs:
THIS IS TO MAKE IT EXTREMELY LIGHTWEIGHT, NO DOCKER, JUST A VENV RECOMMENDED IN PYCHARM.

### LM Studio (Default)
- Endpoint: `http://localhost:1234/v1/chat/completions`
- Models endpoint: `http://localhost:1234/v1/models`
- Supports standard OpenAI-compatible format

### Custom API Integration
Modify the API configuration in `DobAEI.py` to integrate with other language model providers.

## 🤝 Contributing

DoBA is an open-source project that welcomes contributions! We aim to document all contributions in this repository.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- **Memory System Enhancements**: Improve storage and retrieval algorithms
- **Emotional Intelligence**: Expand emotional analysis capabilities
- **Performance Optimization**: GPU acceleration and parallel processing
- **API Integrations**: Support for additional language model providers
- **Documentation**: Improve code documentation and examples
- **Testing**: Add comprehensive test suites

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions and classes
- Include type hints where applicable
- Write meaningful commit messages

## 📊 System Requirements

### Minimum Requirements
- Python 3.7+
- 4GB RAM
- 1GB disk space
- Network connection for API calls

### Recommended Requirements
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU with CUDA support
- SSD storage
- High-speed internet connection

## 🔍 Troubleshooting

### Common Issues

#### Database Errors
```bash
# Reset databases if corruption occurs
python reset_databases.py
```

#### API Connection Issues
- Ensure LM Studio is running on port 1234
- Check firewall settings
- Verify API endpoint configuration

#### Memory Issues
- Monitor database size growth
- Use database reset script for cleanup
- Check available disk space

## 📈 Roadmap

### Current Version Features
- ✅ Nuclear memory system
- ✅ Emotional analysis
- ✅ LM Studio integration
- ✅ GUI interface
- ✅ Database management

### Planned Features
- 🔄 Enhanced semantic search
- 🔄 Multi-model support
- 🔄 Distributed memory systems
- 🔄 Advanced consciousness modeling
- 🔄 Real-time learning capabilities

## 📄 License

This project is licensed under the "Apache 2.0" license. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with the vision of autonomous AI development
- Inspired by biological neural networks
- Community-driven open-source development
- LM Studio for local AI model hosting

## 📞 Support

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/DesignedOBAI/DoBA/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/DesignedOBAI/DoBA/discussions)
- **Documentation**: Check the code documentation and comments for detailed information

---

**DoBA - Designed only By AI** | Pushing the boundaries of autonomous AI development
