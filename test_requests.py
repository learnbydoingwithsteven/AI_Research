import requests
import traceback

ARXIV_API_URL = "http://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results=1"

try:
    print(f"Attempting to connect to arXiv using requests library...")
    print(f"URL: {ARXIV_API_URL}")
    
    # Set a timeout to avoid hanging indefinitely
    response = requests.get(ARXIV_API_URL, timeout=15)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("Successfully connected and received a response.")
        print("Response content snippet:")
        print(response.text[:500] + "...")
    else:
        print("Failed to get a successful response.")
        print(f"Response: {response.text}")

except Exception as e:
    print("\nAn error occurred:")
    print(traceback.format_exc())
