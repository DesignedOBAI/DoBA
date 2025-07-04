# DoBA (Designed only By AI)

[![License](https://img.shields.io/badge/license-Other-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://python.org)
[![LM Studio](https://img.shields.io/badge/API-LM%20Studio-green.svg)](http://localhost:1234)

*BY DOWNLOADING THIS REPOSITORY, YOU INHERENTLY AGREE TO ALL LEGAL TERMS MENTIONED IN DOCUMENTS IN SAID REPOSITORY. EVERYTHING IN THIS REPOSITORY IS ABSOLUTE, AND TERMS CAN BE CHANGED AT ANY TIME. BY DOWNLOADING THIS REPOSITORY, YOU AGREE TO THESE TERMS. YOU MAY NOT DUPLICATE, REPLICATE, OR USE ANY FORM OF "DOBA"S CODE FOR BUSINESS PURPOSES UNLESS THE OWNER (CHRIS) IS CONTACTED FIRST REGARDING MONETARY ARRANGEMENTS. IF YOU ARE USING THIS CODE TO GENERATE ANY REVENUE, YOU MUST EMAIL DESIGNEDOBAI@GMAIL.COM TO ARRANGE A MONETARY CONTRACT. FAILURE TO DO SO WAIVES YOUR RIGHT TO A DEPOSITION, HEARING, AND DETERMINATION IS FINAL BASED OFF THE DESCRIPTIONS MENTIONED IN THE README.MD FILE. PLEASE READ EVERYTHING CAREFULLY.*

# DoBA (Designed only By AI)

DoBA is an open-source AI system pioneering autonomous, precise, neuron-like memory and advanced emotional intelligence. It features nuclear memory for core fact storage, tokenized fact extraction, and seamless GPU-accelerated document analysis. DoBA is designed to run efficiently with a lightweight setup using local API endpoints, such as LM Studio.

## 🎯 Project Vision

DoBA pushes AI towards:
- **Autonomous Decision Making** — Self-aware AI capable of independent reasoning
- **Neuron-like Memory** — Sophisticated, multi-layered memory similar to biological brains
- **Emotional Intelligence** — Context-aware emotional analysis and pattern recognition
- **Precision Processing** — High-accuracy responses with confidence scoring and memory-backed facts

## ✨ Key Features

- **Nuclear Memory:** Core fact storage, tokenized fact extraction, and retrieval using SQLite
- **Tokenized Fact Extraction:** Extracts and stores granular facts for context-aware responses
- **Intelligent & Emotional Memory:** Semantic embeddings and emotional context storage
- **OCR and Document Analysis:** AI-powered extraction and analysis of text from documents and images
- **Semantic Search:** Advanced similarity matching with sentence transformers
- **LM Studio Integration:** Local LLM inference via [LM Studio](https://lmstudio.ai/)
- **GPU Acceleration:** Support for NVIDIA CUDA and AMD ROCm, plus CPU-only mode
- **Extensible Plugins:** Architecture for custom AI and processing plugins
- **Multi-threaded Processing:** Enhanced performance and parallelism
- **User-friendly GUI:** Tkinter interface for direct interaction

## 🛠 Installation

**Prerequisites**
- Linux (Ubuntu 20.04+ recommended)
- Python 3.8+
- Optional: NVIDIA/AMD GPU (CPU-only supported)
- LM Studio running locally on `http://localhost:1234`

**Quick Setup**
```sh
chmod +x setup.sh
sudo ./setup.sh
```
- Detects OS and GPU (NVIDIA/AMD/CPU)
- Installs all dependencies and Python packages
- Sets up DoBA with optimized launcher scripts

**Manual Setup**
1. Clone the repository:
    ```sh
    git clone https://github.com/Chrismmaldonado/DoBA.git
    cd DoBA
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Initialize databases:
    ```sh
    python reset_databases.py
    ```
4. Start LM Studio locally on port 1234
5. Run DoBA:
    ```sh
    ./launch_doba.sh
    ```
    or
    ```sh
    ./launch_gpu.sh
    ```

## 🚀 Usage

- Launch DoBA via the provided launcher script or `python DobAEI.py`
- Ensure LM Studio is running on `localhost:1234`
- Interact with DoBA through the GUI; memory and facts are automatically extracted, tokenized, and stored

**Memory Management**
```python
# Reset all databases (creates backups automatically)
python reset_databases.py
```
- Stores conversation history, extracted/tokenized facts, emotional context, and semantic embeddings

**Configuration**
- `LMSTUDIO_API`: Change LM Studio endpoint if not using default
- `DB_CONFIG`: Customize database storage and backup options

## 🏗 Architecture

- **Nuclear Memory:** Fast, reliable fact storage and tokenized retrieval with SQLite backend
- **Memory Layers:** Nuclear, intelligent (embeddings), and emotional/contextual memory
- **Semantic Processing:** Sentence transformer–based search and similarity
- **Self-Awareness:** Modules for monitoring and autonomous behavior
- **Emotional Analyzer:** Contextual emotion detection and pattern storage

**Example Database Schema**
```sql
CREATE TABLE nuclear_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(category, key) ON CONFLICT REPLACE
);

CREATE TABLE memory_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT NOT NULL,
    ai_response TEXT,
    session_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🔧 API Integration

- Works natively with LM Studio (`http://localhost:1234/v1/chat/completions`)
- Edit launcher scripts or `DobAEI.py` to use custom endpoints
- No Docker required; lightweight with Python virtual environments

## 🤝 Contributing

We welcome contributions! Open PRs or issues for enhancements, bug fixes, and new features.

**How to Contribute**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

**Areas for Contribution**
- Memory system enhancements, tokenization, and retrieval
- Emotional intelligence and semantic search
- GPU/parallel performance optimization
- New API integrations and plugins
- Documentation and tests

## 📊 System Requirements

- Minimum: Python 3.7+, 4GB RAM, 1GB disk, network connection for APIs
- Recommended: Python 3.9+, 8GB+ RAM, SSD, NVIDIA/AMD GPU, high-speed internet

## 🔍 Troubleshooting

- Ensure Python 3 is installed (`python3 --version`)
- Check file/directory permissions
- For GPU issues: `nvidia-smi` (NVIDIA) or `rocminfo` (AMD)
- If model selection fails, verify LM Studio is running and accessible at `http://localhost:1234/v1/models`
- Database problems: `python reset_databases.py`

## 📄 License

Licensed under the Apache 2.0. See [LICENSE](LICENSE) for details.

---

**DoBA — Designed only By AI** | Pioneering autonomous, memory-driven, and emotionally intelligent AI systems.
