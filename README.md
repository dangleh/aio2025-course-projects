# AIO-2025 Course Projects

This repository stores projects by [dangleh](https://github.com/dangleh) for the AIO-2025 course at AI VIET NAM  
Contact email: lehaidang2601@gmail.com

## 📁 Repository Structure

The repository is organized with a module structure containing 2 projects per module:

```
aio2025-course-projects/
├── module-1/
│   ├── project-1.1-streamlit/     # Streamlit Tutorial
│   └── project-1.2-rag-chatbot/   # RAG Chatbot
├── module-2/
│   ├── project-2.1/               # TBD
│   └── project-2.2/               # TBD
├── module-3/
│   ├── project-3.1/               # TBD
│   └── project-3.2/               # TBD
├── module-4/
│   ├── project-4.1/               # TBD
│   └── project-4.2/               # TBD
└── pyproject.toml                 # Root dependencies
```

## 🛠️ Installation and Usage

### System Requirements

- Python >= 3.8
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Run Project

```bash
# Navigate to project directory
cd module-1/project-1.1-streamlit

# Install dependencies
uv sync

# Run application
uv run streamlit run app.py
```

## 📚 Modules

### Module 1: Python Fundamentals

#### Project 1.1 - Streamlit Tutorial

- **Description**: Basic Streamlit tutorial with factorial calculator and grade analysis applications
- **Technologies**: Streamlit, Pandas, NumPy
- **Features**:
  - Factorial calculator application
  - Student grade analysis dashboard
  - Basic Streamlit components tutorial

#### Project 1.2 - RAG Chatbot

- **Description**: RAG Chatbot built from PDF files using LangChain
- **Technologies**: LangChain, OpenAI, FAISS, Streamlit
- **Features**:
  - PDF document processing and reading
  - Vector embeddings creation
  - RAG (Retrieval-Augmented Generation) chatbot
  - Web interface with Streamlit

### Module 2: TBD

_Projects will be updated in the future_

### Module 3: TBD

_Projects will be updated in the future_

### Module 4: TBD

_Projects will be updated in the future_

## 🔧 Development

### Project Structure

Each project follows a standard structure:

```
project-name/
├── pyproject.toml      # Dependencies and configuration
├── app.py             # Main application
├── src/               # Source code
├── tests/             # Unit tests
├── data/              # Sample data
└── README.md          # Project documentation
```

### Code Style

- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [Flake8](https://flake8.pycqa.org/) for linting
- Use [pytest](https://pytest.org/) for testing

### Run Tests

```bash
uv run pytest
```

### Format Code

```bash
uv run black .
```

## 📝 License

MIT License
