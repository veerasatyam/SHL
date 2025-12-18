**File:** `SHL/README.md`

```markdown
# SHL Assessment Recommender System

A **retrieval-augmented recommendation system** that maps messy human job descriptions to structured SHL assessments with measurable accuracy.

## 🎯 Key Features

- **Semantic Search**: Maps job descriptions to SHL assessments using embeddings
- **Rule-based Intelligence**: Analyzes skills and recommends appropriate test types
- **FAISS Vector Database**: Enables <100ms similarity searches
- **FastAPI Backend**: Production-ready API with <2s response time
- **Streamlit Frontend**: Interactive web interface for testing
- **Evaluation Framework**: Measures Recall@k and Precision@k metrics

## 📊 Performance Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Recall@5** | 72% | 72% of relevant assessments found in top 5 |
| **Precision@5** | 44% | 44% of top 5 recommendations are relevant |
| **Recall@10** | 90% | 90% of relevant assessments found in top 10 |
| **Response Time** | <2s | Meets interactive requirements |
| **Assessments Indexed** | 377+ | Individual Test Solutions only |

## 📁 Project Structure

```
SHL/
├── api/                    # FastAPI backend
│   └── simple_api.py      # Production API server
├── data/                  # All data files
│   ├── embeddings/        # FAISS index + embeddings
│   ├── evaluation/        # Test results and metrics
│   └── processed/         # Cleaned assessment data
├── recommender/           # Core algorithms
│   ├── embed.py          # Embedding generation
│   ├── retrieve.py       # Retrieval + re-ranking
│   └── evaluate_fixed.py # Performance evaluation
├── scraper/              # Data collection
│   └── crawl_shl.py      # SHL catalog scraper
├── frontend/             # Optional UI
│   └── simple_app.py     # Streamlit interface
├── final_submission.csv  # Main submission file
├── FINAL_DOCUMENTATION.md # 2-page technical doc
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone and navigate:**
```bash
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

### Testing the System

**Check API health:**
```bash
curl http://localhost:8000/health
```

**Get recommendations:**
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "Java developer with Spring Boot", "top_n": 5}'
```

**Run evaluation:**
```bash
python recommender/evaluate_fixed.py
```

**Start the frontend (optional):**
```bash
streamlit run frontend/simple_app.py
```
Frontend runs at: `http://localhost:8501`

## 🛠️ Implementation Steps

### Phase 1: Data Collection
1. Scraped SHL Product Catalog (377+ Individual Test Solutions)
2. Filtered out "Pre-packaged Job Solutions"
3. Extracted: name, URL, description, skills, test type, category

### Phase 2: Embeddings & Indexing
1. Combined assessment fields into text: `name + description + skills + category`
2. Generated embeddings using `all-MiniLM-L6-v2`
3. Built FAISS index for fast similarity search

### Phase 3: Query Understanding
1. Rule-based skill extraction from job descriptions
2. Test type inference based on detected skills
3. Seniority level detection (junior/mid/senior/executive)

### Phase 4: Retrieval Pipeline
1. **First-pass**: Semantic search via FAISS (top 30 candidates)
2. **Re-ranking**: Business rules based on:
   - Test type matching (+50% boost)
   - Seniority adjustment
   - Diversity enforcement (max 3 per test type)

### Phase 5: API & Frontend
1. FastAPI with endpoints: `/health`, `/recommend`
2. Streamlit interface for interactive testing
3. JSON response format with scores and explanations

### Phase 6: Evaluation
1. Test dataset with 10 diverse job roles
2. Relevance based on test type/category matching
3. Metrics: Recall@k, Precision@k, Response time

## 📈 How It Works

```
Job Description → Skill Extraction → Enhanced Query → FAISS Search → Re-ranking → Recommendations
                     ↓                        ↓                      ↓
                 Hard/Soft Skills        Test Type Needs      Business Rules
                 Seniority Level     Category Requirements   Diversity Filters
```

### Example: "Java Developer with Spring Boot"
1. **Skill Extraction**: Hard: ["Java", "Spring Boot"], Soft: []
2. **Test Types**: ["K"] (Knowledge tests)
3. **Search**: Finds Java Programming Test (score: 0.845), Technical Skills Assessment (0.812)
4. **Re-ranking**: Boosts "K" type tests, ensures diversity
5. **Output**: Ranked list with URLs and relevance scores

## 🧪 Test Queries

The system was tested on 10 diverse job descriptions:

1. Java developer with Spring Boot experience
2. Data analyst with SQL and Python skills
3. Customer service representative with communication skills
4. Marketing manager with digital marketing expertise
5. Software engineer with Python programming
6. Financial analyst with numerical reasoning skills
7. Project manager with leadership experience
8. Sales representative with persuasive communication
9. HR manager with interpersonal skills
10. Technical support specialist with problem-solving skills

## 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI + Uvicorn | REST API server |
| **Vector DB** | FAISS | Fast similarity search |
| **Embeddings** | SentenceTransformers | Semantic understanding |
| **Data Processing** | Pandas + BeautifulSoup | Data collection & cleaning |
| **Frontend** | Streamlit | Interactive UI |
| **Evaluation** | Scikit-learn | Performance metrics |

## 📄 Submission Files

1. **`final_submission.csv`**: Main output with 10 queries × 10 recommendations
   - Format: `query, assessment_name, assessment_url`
   - 100 total recommendations

2. **`FINAL_DOCUMENTATION.md`**: 2-page technical documentation
   - Architecture overview
   - Performance metrics
   - Challenges & solutions
   - Improvement roadmap

## 🎯 Key Achievements

1. **Data Collection**: Successfully scraped 377+ Individual Test Solutions
2. **Semantic Search**: FAISS enables fast, accurate similarity matching
3. **Rule-based Intelligence**: Skill extraction without LLM dependency
4. **Production API**: FastAPI with <2s response time
5. **Measurable Accuracy**: Recall@10 of 90%, Precision@5 of 44%
6. **Correct Format**: Submission CSV matches required specification

## 🚨 Troubleshooting

### Common Issues:

1. **"API not responding"**
   ```bash
   # Check if API is running
   python api/simple_api.py
   # Should see: "API READY on http://localhost:8000"
   ```

2. **"Module not found"**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **"No recommendations"**
   - Ensure job description has specific skills
   - Check API response with: `curl http://localhost:8000/health`

4. **"Evaluation shows 0%"**
   - Run the fixed evaluator: `python recommender/evaluate_fixed.py`

## 📚 Documentation

- **`FINAL_DOCUMENTATION.md`**: Complete technical documentation (2 pages)
- **API Documentation**: Visit `http://localhost:8000` when API is running
- **Code Comments**: All major functions documented inline

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the technical documentation
3. Test with sample queries first
4. Contact me 

## 📄 License

This project is for assessment purposes only. SHL is a registered trademark.

---#   S H L  
 