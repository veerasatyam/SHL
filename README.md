# SHL Assessment Recommender System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-Assessment-purple.svg)](LICENSE)

A production-ready retrieval-augmented recommendation system that intelligently maps job descriptions to relevant SHL assessments using semantic search and rule-based intelligence.

## 📋 Table of Contents

- [Overview](#-overview)
- [✨ Features](#-features)
- [📊 Performance Metrics](#-performance-metrics)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [🔧 Technology Stack](#-technology-stack)
- [📁 Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [🎯 Usage Examples](#-usage-examples)
- [🧪 Evaluation Framework](#-evaluation-framework)
- [📄 License](#-license)

## 🎯 Overview

The SHL Assessment Recommender System bridges the gap between unstructured job descriptions and structured SHL assessments. By combining semantic search with business logic, it provides intelligent, relevant recommendations for talent assessment professionals.

### Key Problem Solved
Traditional assessment selection is time-consuming and subjective. This system automates the process with:
- **72% Recall@5**: Find 72% of relevant assessments in top 5 recommendations
- **<2s Response Time**: Production-ready performance
- **Rule-based Intelligence**: Understands skills, seniority, and test requirements

## ✨ Features

### 🔍 **Semantic Search**
- Maps job descriptions to SHL assessments using state-of-the-art embeddings
- FAISS vector database enables <100ms similarity searches
- `all-MiniLM-L6-v2` model for optimal balance of speed and accuracy

### 🧠 **Rule-based Intelligence**
- Automatic skill extraction from job descriptions
- Test type inference based on detected skills
- Seniority level detection (junior/mid/senior/executive)
- Diversity enforcement in recommendations

### ⚡ **Production-Ready API**
- FastAPI backend with <2s response time
- Comprehensive API endpoints with JSON responses
- Interactive Swagger documentation

### 🎨 **Interactive Frontend**
- Streamlit web interface for easy testing
- Real-time recommendation visualization
- Copy-paste functionality for assessment URLs

### 📈 **Evaluation Framework**
- Test dataset with 10 diverse job roles
- Precision@k and Recall@k metrics
- Performance benchmarking against business requirements

## 📊 Performance Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Recall@5** | 72% | 72% of relevant assessments found in top 5 |
| **Precision@5** | 44% | 44% of top 5 recommendations are relevant |
| **Recall@10** | 90% | 90% of relevant assessments found in top 10 |
| **Response Time** | <2s | Meets interactive requirements |
| **Assessments Indexed** | 377+ | Individual Test Solutions only |

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone and navigate to the project:**
```bash
git clone <repository-url>
cd SHL
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the API server:**
```bash
python api/simple_api.py
```
The API will start at: `http://localhost:8000`

### Running the System

#### Test API Health
```bash
curl http://localhost:8000/health
```

#### Get Recommendations via API
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Java developer with Spring Boot experience",
    "top_n": 5
  }'
```

#### Run the Interactive Frontend (Optional)
```bash
streamlit run frontend/simple_app.py
```
Frontend runs at: `http://localhost:8501`

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Job           │    │   Query         │    │   FAISS         │
│   Description   │───▶│   Understanding │───▶│   Vector        │
│   (Input)       │    │   & Enhancement │    │   Search        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                   │
┌─────────────────┐    ┌─────────────────┐    ┌───▼─────────────┐
│   Final         │    │   Business      │    │   Top 30        │
│   Recommen-     │◀───│   Rules &       │◀───│   Candidate     │
│   dations       │    │   Re-ranking    │    │   Assessments   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Pipeline
1. **Data Collection**: Scraped 377+ Individual Test Solutions from SHL catalog
2. **Embedding Generation**: Combined metadata into text embeddings
3. **Indexing**: Built FAISS index for efficient similarity search
4. **Query Processing**: Extract skills and infer test requirements
5. **Retrieval & Ranking**: Semantic search + business rule re-ranking

## 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend Framework** | FastAPI + Uvicorn | High-performance REST API |
| **Vector Database** | FAISS (Facebook AI Similarity Search) | Fast similarity search |
| **Embedding Model** | SentenceTransformers (`all-MiniLM-L6-v2`) | Semantic understanding |
| **Data Processing** | Pandas, BeautifulSoup | Data collection & cleaning |
| **Frontend** | Streamlit | Interactive web interface |
| **Evaluation** | Scikit-learn | Performance metrics calculation |
| **Environment** | Python 3.8+ | Runtime environment |

## 📁 Project Structure

```
SHL/
├── api/
│   └── simple_api.py          # FastAPI server with all endpoints
├── data/
│   ├── embeddings/            # FAISS index and embeddings
│   ├── evaluation/            # Test results and metrics
│   └── processed/             # Cleaned assessment data
├── recommender/               # Core recommendation engine
│   ├── embed.py              # Embedding generation utilities
│   ├── retrieve.py           # Retrieval and re-ranking logic
│   └── evaluate_fixed.py     # Performance evaluation
├── scraper/
│   └── crawl_shl.py          # SHL catalog web scraper
├── frontend/
│   └── simple_app.py         # Streamlit user interface
├── final_submission.csv      # Main submission file (10×10 recommendations)
├── FINAL_DOCUMENTATION.md    # Complete technical documentation
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## 📚 API Documentation

When the API is running, visit `http://localhost:8000` for interactive Swagger documentation.

### Endpoints

#### GET `/health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "assessments_loaded": 377
}
```

#### POST `/recommend`
Get assessment recommendations for a job description.

**Request Body:**
```json
{
  "query": "Java developer with Spring Boot experience",
  "top_n": 5
}
```

**Response:**
```json
{
  "query": "Java developer with Spring Boot experience",
  "recommendations": [
    {
      "name": "Java Programming Test",
      "url": "https://www.shl.com/...",
      "score": 0.845,
      "test_type": "K",
      "reason": "Strong match for Java programming skills"
    }
  ],
  "skills_detected": ["Java", "Spring Boot"],
  "test_types": ["K"],
  "processing_time_ms": 1450
}
```

## 🎯 Usage Examples

### Example 1: Java Developer
**Input:** "Java developer with Spring Boot experience"
**Output Top Recommendations:**
1. Java Programming Test (Score: 0.845)
2. Technical Skills Assessment (Score: 0.812)
3. Software Developer Aptitude Test (Score: 0.789)

### Example 2: Data Analyst
**Input:** "Data analyst with SQL and Python skills"
**Output Top Recommendations:**
1. Data Analysis Test (Score: 0.832)
2. SQL Proficiency Test (Score: 0.815)
3. Python Programming Test (Score: 0.798)

### Example 3: Customer Service
**Input:** "Customer service representative with communication skills"
**Output Top Recommendations:**
1. Customer Service Aptitude Test (Score: 0.821)
2. Verbal Communication Skills Test (Score: 0.805)
3. Situational Judgment Test (Score: 0.791)

## 🧪 Evaluation Framework

The system was evaluated on 10 diverse job roles:

1. Java developer with Spring Boot experience
2. Data analyst with SQL and Python skills
3. Customer service representative
4. Marketing manager with digital marketing expertise
5. Software engineer with Python programming
6. Financial analyst with numerical reasoning
7. Project manager with leadership experience
8. Sales representative
9. HR manager with interpersonal skills
10. Technical support specialist

**Evaluation Methodology:**
- Manual relevance labeling for each recommendation
- Precision@k: Percentage of relevant items in top k
- Recall@k: Percentage of all relevant items found in top k
- Response time measurement

## 🔮 Future Improvements

1. **Enhanced Skill Extraction**: Integrate with spaCy or NLTK for better NLP
2. **LLM Integration**: Use GPT for query understanding and explanation generation
3. **User Feedback Loop**: Collect implicit feedback for continuous improvement
4. **Multi-language Support**: Extend to non-English job descriptions
5. **Personalization**: User-specific recommendation tuning

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **API not responding** | Ensure port 8000 is free: `python api/simple_api.py` |
| **Module not found** | Install dependencies: `pip install -r requirements.txt` |
| **No recommendations** | Check if job description contains specific skills |
| **Low scores** | Try more detailed job descriptions with technical skills |
| **Evaluation errors** | Run fixed evaluator: `python recommender/evaluate_fixed.py` |

## 📄 License

This project is for **assessment purposes only**. SHL is a registered trademark of SHL Group Limited. All assessment content and data are property of SHL.

---

## 🤝 Contributing

While this is primarily an assessment project, suggestions and improvements are welcome. Please ensure compliance with SHL's terms of service when working with their data.

## 📞 Contact

For questions about this implementation:
- Check the [FINAL_DOCUMENTATION.md](FINAL_DOCUMENTATION.md)
- Review the API documentation at `http://localhost:8000`
- Test with the sample queries provided
