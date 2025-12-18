# Quick test - check your current URLs vs train URLs
import pickle
import pandas as pd

# Load your metadata
with open("data/embeddings/metadata.pkl", "rb") as f:
    your_data = pickle.load(f)

# Load train data
train_df = pd.read_excel("data/external/Gen_AI Dataset.xlsx", sheet_name="Train-Set")
train_urls = train_df["Assessment_url"].tolist()

# Check for any matches
your_urls = [item.get("url", "") for item in your_data]
matches = set(your_urls) & set(train_urls)

print(f"You have {len(your_urls)} assessments")
print(f"Train data has {len(train_urls)} URLs")
print(f"Number of matching URLs: {len(matches)}")
# Find which assessments match
matches = set(your_urls) & set(train_urls)
print("Matching assessments:")
for url in matches:
    # Find in your data
    for item in your_data:
        if item.get("url") == url:
            print(f"- {item.get('assessment_name', 'Unknown')}")
            print(f"  URL: {url}")
            
# Find which queries use these URLs
print("\nQueries that use these assessments:")
for url in matches:
    queries = train_df[train_df["Assessment_url"] == url]["Query"].unique()
    for q in queries:
        print(f"- {q[:80]}...")
if len(matches) == 0:
    print("❌ NO MATCHES - That's why you get 0% accuracy!")
else:
    print(f"✅ Found {len(matches)} matches")