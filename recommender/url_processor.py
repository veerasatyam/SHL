"""
Process job descriptions from URLs.
"""
import requests
from bs4 import BeautifulSoup
import re

class URLProcessor:
    """Extract job description text from URLs."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def extract_from_url(self, url: str) -> str:
        """
        Extract job description text from a URL.
        
        Args:
            url: URL to job posting
            
        Returns:
            Extracted job description text
        """
        try:
            # Fetch the webpage
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Try to find job description section
            # Common patterns for job postings
            job_keywords = [
                "job description", "requirements", "qualifications",
                "responsibilities", "about the role", "what you'll do"
            ]
            
            # Look for sections with job keywords
            found_sections = []
            for keyword in job_keywords:
                if keyword.lower() in text.lower():
                    # Find context around keyword
                    idx = text.lower().find(keyword.lower())
                    start = max(0, idx - 500)
                    end = min(len(text), idx + 1500)
                    section = text[start:end]
                    found_sections.append(section)
            
            # Use found sections or full text
            if found_sections:
                text = " ".join(found_sections)
            
            # Limit length
            if len(text) > 5000:
                text = text[:5000]
            
            return text
            
        except Exception as e:
            print(f"Error extracting from URL {url}: {e}")
            return f"Job description from URL: {url}"