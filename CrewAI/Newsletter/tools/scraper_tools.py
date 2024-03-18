from langchain_community.tools import tool
import requests
from bs4 import BeautifulSoup

class ScraperTools:
    @tool("Scrape website content, text only")
    def scrape_website(self, url: str) -> str:
        """Scrape the text content of a website."""
        try:
            # Send a GET request to the URL
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299'}
            response = requests.get(url, headers=headers)
            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Parse the HTML content of the page
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract all text from the page
                text = soup.get_text(separator=' ', strip=True)
                return text
            else:
                return f"Failed to retrieve content, status code: {response.status_code}"
        except Exception as e:
            return f"An error occurred: {e}"