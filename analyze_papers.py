import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
import seaborn as sns

def parse_readme(readme_path='README.md'):
    """Parses the README.md file to extract paper details."""
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find paper blocks with their date
    paper_blocks = re.findall(r'## <a name=".*?"></a>Papers Added on (\d{4}-\d{2}-\d{2})\n\n(.*?)(?=\n## <a name=|$)', content, re.DOTALL)
    
    papers = []
    for date_str, block in paper_blocks:
        # Regex to find individual papers within a block
        individual_papers = re.findall(r'### \[(.*?)\]\(.*?\)\n\n(.*?)\n\n---', block, re.DOTALL)
        for title, details in individual_papers:
            papers.append({'date': date_str, 'title': title.strip(), 'details': details.strip()})
    
    return pd.DataFrame(papers)

def analyze_topics(df):
    """Analyzes the distribution of topics based on keywords in titles."""
    topic_keywords = {
        'LLM/VLM': ['LLM', 'VLM', 'Large Language Model', 'Vision Language Model', 'HyperCLOVA'],
        'Agents': ['Agent', 'Agentic', 'Multi-Agent'],
        'Reinforcement Learning': ['Reinforcement Learning', 'RL', 'Actor-Critic'],
        'Generative Models': ['Generative', 'Diffusion', 'GAN', 'Flow Matching'],
        'Federated Learning': ['Federated Learning', 'PFL'],
        'Neural Networks': ['Neural Network', 'ReLU', 'Activation'],
        'Robotics/UAV': ['UAV', 'Robotics', 'Robot'],
        'Benchmark': ['Benchmark', 'Speedrunning']
    }

    topic_counts = Counter()
    for _, row in df.iterrows():
        text_to_search = row['title'] + ' ' + row['details']
        found_topic = False
        for topic, keywords in topic_keywords.items():
            if any(re.search(r'\b' + keyword + r'\b', text_to_search, re.IGNORECASE) for keyword in keywords):
                topic_counts[topic] += 1
                found_topic = True
        if not found_topic:
            topic_counts['Other'] += 1

    return topic_counts

def plot_topic_distribution(topic_counts, save_path='visualizations/topic_distribution.png'):
    """Plots and saves the topic distribution bar chart."""
    topics = list(topic_counts.keys())
    counts = list(topic_counts.values())

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 8))
    sns.barplot(x=counts, y=topics, palette='viridis', hue=topics, dodge=False, legend=False)
    plt.title('Distribution of AI Research Topics', fontsize=16)
    plt.xlabel('Number of Papers', fontsize=12)
    plt.ylabel('Topic', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Topic distribution plot saved to {save_path}")

def plot_papers_over_time(df, save_path='visualizations/papers_per_day.png'):
    """Plots and saves the number of papers added over time."""
    df['date'] = pd.to_datetime(df['date'])
    papers_per_month = df.groupby(df['date'].dt.to_period('M')).size()
    papers_per_month.index = papers_per_month.index.to_timestamp()

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 6))
    plt.plot(papers_per_month.index, papers_per_month.values, marker='o', linestyle='-')
    plt.title('AI Papers Added Per Month', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Papers over time plot saved to {save_path}")

if __name__ == "__main__":
    print("Starting analysis of README.md...")
    papers_df = parse_readme()
    if not papers_df.empty:
        print(f"Successfully parsed {len(papers_df)} papers.")
        
        # Generate and save topic distribution plot
        topics = analyze_topics(papers_df)
        plot_topic_distribution(topics)

        # Generate and save papers over time plot
        plot_papers_over_time(papers_df)
        
        print("Analysis complete. Visualizations have been updated.")
    else:
        print("No papers found in README.md. Skipping analysis.")
