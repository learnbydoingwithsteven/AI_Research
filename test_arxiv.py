import arxiv
import traceback
import logging

# Set up logging to capture everything from urllib3
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)

try:
    print("Attempting to connect to arXiv with DETAILED urllib3 logging...")
    
    client = arxiv.Client(
        page_size=10,
        delay_seconds=3.0,
        num_retries=3
    )

    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=1,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    result = next(client.results(search))
    
    print("\nSuccessfully fetched one paper:")
    print(f"Title: {result.title}")
    print(f"Published: {result.published.date()}")

except Exception as e:
    print("\nAn error occurred:")
    print(traceback.format_exc())
