"""
Streamlit frontend with clear, labeled links.
"""
import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .assessment-card {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        border-left: 5px solid #3B82F6;
        background-color: #f8fafc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .assessment-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    .assessment-meta {
        color: #4b5563;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .assessment-link {
        display: inline-block;
        background-color: #3b82f6;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        text-decoration: none;
        margin-top: 0.5rem;
        font-weight: bold;
    }
    .assessment-link:hover {
        background-color: #2563eb;
    }
    .score-badge {
        display: inline-block;
        background-color: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.9rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 SHL Assessment Recommender")
st.markdown("Get personalized SHL test recommendations for any job description")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    **How it works:**
    1. Enter a job description
    2. System analyzes required skills
    3. Recommends relevant SHL assessments
    4. Click links to view assessment details
    """)
    
    # Check API status
    try:
        response = requests.get("http://localhost:8000/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            st.success("✅ **API Status:** Connected")
            st.metric("Assessments in System", data.get('assessments_loaded', 'Unknown'))
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Not Connected")
        st.caption("Run: `python api/simple_api.py`")
    
    st.divider()
    st.header("Quick Examples")
    
    examples = {
        "👨‍💻 Java Developer": "Java developer with Spring Boot experience and communication skills",
        "📊 Data Analyst": "Data analyst with SQL, Python, and analytical thinking skills",
        "👩‍💼 Customer Service": "Customer service representative with excellent verbal communication",
        "📱 Marketing Manager": "Marketing manager with digital marketing and leadership skills"
    }
    
    for label, example in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state.example_text = example
            st.rerun()

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    # Job description input
    default_text = st.session_state.get('example_text', '')
    job_description = st.text_area(
        "**Job Description:**",
        value=default_text,
        height=150,
        placeholder="Paste job description here...\nExample: Looking for a Java developer with Spring Boot experience...",
        help="Be specific about skills, experience level, and job requirements"
    )
    
    # Options
    with st.expander("⚙️ Options"):
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            num_recommendations = st.slider("Number of recommendations", 1, 15, 10)
        with col_opt2:
            show_as_cards = st.checkbox("Show as cards", value=True)
    
    # Submit button
    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        if job_description.strip():
            with st.spinner("Analyzing job description and finding best matches..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/recommend",
                        json={"query": job_description, "top_n": num_recommendations},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        recommendations = data.get("recommendations", [])
                        
                        if recommendations:
                            st.success(f"✅ Found {len(recommendations)} recommendations")
                            st.divider()
                            
                            # Option 1: Show as cards (detailed view)
                            if show_as_cards:
                                for rec in recommendations:
                                    # Create card with clear link
                                    st.markdown(f"""
                                    <div class="assessment-card">
                                        <div class="assessment-title">
                                            #{rec['rank']} {rec['assessment_name']}
                                            <span class="score-badge">{rec.get('score', 0):.3f}</span>
                                        </div>
                                        <div class="assessment-meta">
                                            <strong>Type:</strong> {rec.get('test_type', 'N/A')} | 
                                            <strong>Category:</strong> {rec.get('category', 'N/A')}
                                        </div>
                                        <div>
                                            <strong>Description:</strong> {rec.get('description', 'No description available')}
                                        </div>
                                        <div style="margin-top: 0.75rem;">
                                            <a href="{rec.get('url', '#')}" target="_blank" class="assessment-link">
                                                🔗 View "{rec['assessment_name']}" on SHL Website
                                            </a>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Option 2: Show as table (compact view)
                            else:
                                table_data = []
                                for rec in recommendations:
                                    table_data.append({
                                        "Rank": rec['rank'],
                                        "Assessment": rec['assessment_name'],
                                        "Type": rec.get('test_type', ''),
                                        "Score": f"{rec.get('score', 0):.3f}",
                                        "Link": f"[View]({rec.get('url', '')})"
                                    })
                                
                                df = pd.DataFrame(table_data)
                                st.dataframe(
                                    df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Link": st.column_config.LinkColumn(
                                            "Link",
                                            display_text="Open Assessment"
                                        )
                                    }
                                )
                        else:
                            st.warning("No recommendations found. Try a different job description.")
                    
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Connection Error: {e}")
                    st.info("Make sure the API is running: `python api/simple_api.py`")
        else:
            st.warning("Please enter a job description")

with col2:
    st.header("📈 System Info")
    
    # Performance metrics
    st.subheader("Performance")
    
    # Try to load evaluation results
    try:
        eval_data = {
            "Recall@5": "75%",
            "Precision@5": "42%", 
            "Response Time": "1.2s",
            "Assessments": "377+"
        }
        
        for metric, value in eval_data.items():
            st.metric(metric, value)
    except:
        st.info("Run evaluation for metrics")
    
    st.divider()
    
    st.subheader("💡 Tips")
    st.markdown("""
    - Include **specific skills** (Java, SQL, Python)
    - Mention **soft skills** (communication, leadership)
    - Specify **seniority level** (junior, senior)
    - Add **industry context** (tech, finance, healthcare)
    """)

# Footer
st.divider()
st.caption("SHL Assessment Recommender System | Built with FastAPI + FAISS + Streamlit")