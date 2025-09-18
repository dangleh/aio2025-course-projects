# Project 1.1 - Streamlit Tutorial

Basic Streamlit tutorial with factorial calculator and grade analysis applications.

## Description
This project includes:
- **Factorial Calculator App**: Login-based factorial calculator with user authentication
- **Score Analysis App**: Excel file upload and grade distribution analysis with charts
- **Basic Streamlit Components**: Demonstrates various Streamlit widgets and features

## Features
- User authentication system
- Factorial calculation with input validation
- Excel file processing for grade analysis
- Interactive charts and visualizations
- Responsive UI with proper error handling

## Installation
```bash
# Install dependencies using uv
uv sync

# Run the main factorial calculator app
uv run streamlit run app.py

# Run the score analysis app
uv run streamlit run score_analysis.py
```

## Project Structure
```
project-1.1-streamlit/
├── app.py              # Main factorial calculator app
├── score_analysis.py   # Grade analysis application
├── src/                # Source code modules
│   ├── factorial.py    # Factorial calculation function
│   └── footer.py       # Footer component
├── data/               # Sample data
│   ├── scores.xlsx     # Sample grade data
│   └── users.txt       # User authentication data
├── tests/              # Unit tests
└── README.md
```

## Requirements
- Python >= 3.8
- Streamlit >= 1.28.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Pillow >= 10.0.0
- OpenPyXL >= 3.1.0