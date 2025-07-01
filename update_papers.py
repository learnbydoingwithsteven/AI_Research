import requests
import feedparser
import datetime
import traceback
import re
from time import sleep

def fetch_papers(query, max_results_per_page=1000, max_total_results=10000):
    """Fetches papers from the arXiv API using requests and feedparser, with pagination."""
    base_url = 'http://export.arxiv.org/api/query?'
    all_papers = []
    start = 0
    
    print("Starting to fetch papers with pagination...")
    while True:
        search_query = (f'search_query={query}&sortBy=submittedDate&sortOrder=descending' + 
                        f'&start={start}&max_results={max_results_per_page}')
        
        print(f"Querying URL: {base_url}{search_query}")
        try:
            response = requests.get(base_url + search_query, timeout=30)
            if response.status_code != 200:
                print(f"Warning: Failed to fetch from arXiv API. Status: {response.status_code}. Retrying once...")
                sleep(5)
                response = requests.get(base_url + search_query, timeout=30)
                if response.status_code != 200:
                    print(f"Error: Failed to fetch from arXiv API on retry. Status: {response.status_code}. Aborting page fetch.")
                    break

            feed = feedparser.parse(response.content)
            page_papers = feed.entries
            print(f"Page fetched. Found {len(page_papers)} entries.")
            
            if not page_papers:
                print("No more papers found. Ending fetch.")
                break
            
            all_papers.extend(page_papers)
            start += len(page_papers)
            
            if start >= max_total_results:
                print(f"Reached max total results limit of {max_total_results}. Stopping fetch.")
                break

            # Be polite to the API
            sleep(3)

        except Exception as e:
            print(f"An error occurred during fetch: {e}")
            print(traceback.format_exc())
            break
            
    print(f"Finished fetching. Total papers retrieved: {len(all_papers)}")
    return all_papers

def format_papers_md(papers):
    """Formats a list of papers from the feed into Markdown."""
    papers_by_date = {}
    for paper in papers:
        # Use published_parsed for robust date handling
        date_obj = datetime.datetime(*paper.published_parsed[:6])
        date_str = date_obj.strftime('%Y-%m-%d')
        if date_str not in papers_by_date:
            papers_by_date[date_str] = []
        papers_by_date[date_str].append(paper)

    md_string = ""
    sorted_dates = sorted(papers_by_date.keys(), reverse=True)

    for date_str in sorted_dates:
        md_string += f'## <a name="{date_str.replace("-", "")}"></a>Papers Added on {date_str}\n\n'
        for paper in papers_by_date[date_str]:
            title = paper.title.replace('\n', ' ').strip()
            authors = ", ".join([author.name for author in paper.authors])
            summary = paper.summary.replace('\n', ' ').strip()
            pdf_link = ''
            for link in paper.links:
                if link.get('title') == 'pdf':
                    pdf_link = link.get('href')
                    break

            md_string += f'### [{title}]({pdf_link})\n\n'
            md_string += f'**Authors:** {authors}\n\n'
            md_string += f'{summary}\n\n---\n\n'

    return md_string

def rebuild_readme(papers):
    """Rebuilds the README.md with a fresh list of papers."""
    print("Rebuilding README.md...")
    
    # 1. Format the new paper content
    new_papers_md = format_papers_md(papers)
    
    # 2. Generate a new Table of Contents
    dates = sorted(list(set([datetime.datetime(*p.published_parsed[:6]).strftime('%Y-%m-%d') for p in papers])), reverse=True)
    new_toc = "## Table of Contents\n\n"
    for date_str in dates:
        new_toc += f'- [{date_str}](#{date_str.replace("-", "")})\n'
        
    # 3. Read the existing README to preserve the header
    with open('README.md', 'r', encoding='utf-8') as f:
        readme_content = f.read()
        
    # 4. Find where the paper list starts and cut everything after it
    # The paper list starts after the '---' following the Table of Contents
    toc_start_marker = "## Table of Contents"
    
    toc_start_index = readme_content.find(toc_start_marker)
    if toc_start_index == -1:
        # If no TOC, just use the whole file as header (fallback)
        print("Warning: '## Table of Contents' not found. Overwriting from top.")
        readme_analytics_part = ""
    else:
        readme_analytics_part = readme_content[:toc_start_index]
    
    final_readme = readme_analytics_part + new_toc + "\n---\n\n" + new_papers_md
    
    # 6. Write the new content to the file
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(final_readme)
    print("README.md has been successfully rebuilt.")


if __name__ == "__main__":
    try:
        QUERY = 'cat:cs.AI+OR+cat:cs.LG'
        print("Fetching recent papers using requests and feedparser...")
        all_fetched_papers = fetch_papers(QUERY, max_results_per_page=1000, max_total_results=10000)
        print(f"Total papers fetched: {len(all_fetched_papers)}")
        
        three_months_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=90)
        recent_papers = []
        print("Filtering papers for the last 3 months...")
        for paper in all_fetched_papers:
            # Use feedparser's parsed time object which is timezone-aware
            paper_date = datetime.datetime(*paper.published_parsed[:6], tzinfo=datetime.timezone.utc)
            if paper_date > three_months_ago:
                recent_papers.append(paper)
        print(f"Found {len(recent_papers)} papers from the last 3 months.")

        if recent_papers:
            rebuild_readme(recent_papers)
        else:
            print("No papers found in the last 3 months to update README.")

    except Exception as e:
        print("An error occurred:")
        print(traceback.format_exc())
