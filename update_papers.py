import requests
import feedparser
import datetime
import traceback
import re
from time import sleep

def fetch_papers(query, max_results=2000):
    """Fetches papers from the arXiv API using requests and feedparser."""
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}'
    
    print(f"Querying URL: {base_url}{search_query}")
    response = requests.get(base_url + search_query, timeout=30)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch from arXiv API. Status: {response.status_code}")

    print("Parsing feed...")
    feed = feedparser.parse(response.content)
    print(f"Feed parsed. Found {len(feed.entries)} entries.")
    return feed.entries

def format_papers_md(papers):
    """Formats a list of papers from the feed into Markdown."""
    papers_by_date = {}
    for paper in papers:
        date_str = datetime.datetime.strptime(paper.published, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
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

if __name__ == "__main__":
    try:
        QUERY = 'cat:cs.AI+OR+cat:cs.LG'
        print("Fetching recent papers using requests and feedparser...")
        all_fetched_papers = fetch_papers(QUERY)
        print(f"Total papers fetched: {len(all_fetched_papers)}")
        
        three_months_ago = datetime.datetime.now() - datetime.timedelta(days=90)
        recent_papers = []
        print("Filtering papers for the last 3 months...")
        for paper in all_fetched_papers:
            paper_date = datetime.datetime.strptime(paper.published, '%Y-%m-%dT%H:%M:%SZ')
            if paper_date > three_months_ago:
                recent_papers.append(paper)
        print(f"Found {len(recent_papers)} papers from the last 3 months.")

        if recent_papers:
            print(f"Found {len(recent_papers)} new papers to process.")
            with open('README.md', 'r', encoding='utf-8') as f:
                readme_content = f.read()

            final_md_output = ""
            new_papers_count = 0
            print("Checking for duplicates and preparing new content...")
            for paper in recent_papers:
                if paper.title.replace('\n', ' ').strip() not in readme_content:
                    final_md_output += format_papers_md([paper])
                    new_papers_count += 1
            
            if final_md_output:
                print(f"Found {new_papers_count} new, unique papers to add.")
                # Update TOC and add papers
                new_dates = sorted(list(set([datetime.datetime.strptime(p.published, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d') for p in recent_papers])), reverse=True)
                all_date_headers = re.findall(r'## <a name="\d+"></a>Papers Added on (\d{4}-\d{2}-\d{2})', readme_content)
                all_known_dates = sorted(list(set(all_date_headers + new_dates)), reverse=True)

                new_toc = "## Table of Contents\n\n"
                for date_str in all_known_dates:
                    new_toc += f'- [{date_str}](#{date_str.replace("-", "")})\n'

                toc_start_marker = "## Table of Contents"
                toc_start_index = readme_content.find(toc_start_marker)
                toc_end_marker = "\n---"
                toc_end_index = readme_content.find(toc_end_marker, toc_start_index)

                readme_with_new_toc = readme_content[:toc_start_index] + new_toc + readme_content[toc_end_index:]
                insertion_marker = readme_with_new_toc.find(toc_end_marker, toc_start_index) + len(toc_end_marker)
                final_readme = readme_with_new_toc[:insertion_marker] + "\n\n" + final_md_output + readme_with_new_toc[insertion_marker:]
                
                with open('README.md', 'w', encoding='utf-8') as f:
                    f.write(final_readme)
                print("README.md has been successfully updated.")
            else:
                print("No new, unique papers found to add.")
        else:
            print("No papers found in the last 3 months.")

    except Exception as e:
        print("An error occurred:")
        print(traceback.format_exc())
