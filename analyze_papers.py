import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import traceback
import logging

# Set up logging
logging.basicConfig(filename='analysis.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')

def parse_readme(readme_path='README.md'):
    """Parses the README.md file to extract paper details."""
    logging.info("Parsing README.md...")
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    paper_blocks = re.findall(r'## <a name=".*?"></a>Papers Added on (\d{4}-\d{2}-\d{2})\n\n(.*?)(?=\n## <a name=|$)', content, re.DOTALL)
    papers = []
    for date_str, block in paper_blocks:
        individual_papers = re.findall(r'### \[(.*?)\]\((.*?)\)\n\n\*\*Authors:\*\* (.*?)\n\n(.*?)\n\n---', block, re.DOTALL)
        for title, link, authors, summary in individual_papers:
            papers.append({'date': date_str, 'title': title.strip(), 'link': link, 'authors': authors, 'summary': summary.strip()})
    logging.info(f"Parsed {len(papers)} papers.")
    return pd.DataFrame(papers)

def save_to_csv(df, path='papers.csv'):
    """Saves the DataFrame to a CSV file."""
    logging.info(f"Saving data to {path}...")
    df.to_csv(path, index=False)
    logging.info(f"Data successfully saved to {path}.")

def analyze_topics(df):
    """Adds a 'topic' column to the DataFrame based on keywords."""
    logging.info("Analyzing topics...")
    df_copy = df.copy()
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
    df_copy['topic'] = 'Other'
    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            df_copy.loc[df_copy['title'].str.contains(keyword, case=False, regex=False) | df_copy['summary'].str.contains(keyword, case=False, regex=False), 'topic'] = topic
    logging.info("Topic analysis complete.")
    return df_copy

def plot_topic_distribution(topic_counts, save_path='visualizations/topic_distribution.png'):
    """Plots and saves the topic distribution bar chart."""
    logging.info(f"Generating topic distribution plot at {save_path}...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 8))
    sns.barplot(x=topic_counts.values, y=topic_counts.index, palette='viridis')
    plt.title('Distribution of AI Research Topics', fontsize=16)
    plt.xlabel('Number of Papers', fontsize=12)
    plt.ylabel('Topic', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info("Topic distribution plot saved.")

def plot_tsa_decomposition(df, save_path='visualizations/tsa_decomposition.png'):
    """Performs and plots time series decomposition."""
    logging.info(f"Generating TSA decomposition plot at {save_path}...")
    df['date'] = pd.to_datetime(df['date'])
    papers_per_day = df.set_index('date').resample('D').size().rename('paper_count')
    period = 7
    if len(papers_per_day) < 2 * period:
        logging.warning(f"Skipping TSA plot: not enough data. Need at least {2*period} days, but have {len(papers_per_day)}.")
        return
    decomposition = seasonal_decompose(papers_per_day, model='additive', period=period)
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.suptitle('Time Series Decomposition of Daily Paper Submissions', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    logging.info("TSA decomposition plot saved.")

def plot_monthly_topic_pie(df, save_path='visualizations/monthly_topic_pie.png'):
    """Plots a pie chart of topic distribution for the most recent month."""
    logging.info(f"Generating monthly topic pie chart at {save_path}...")
    df['date'] = pd.to_datetime(df['date'])
    latest_month = df['date'].max().to_period('M')
    monthly_df = df[df['date'].dt.to_period('M') == latest_month]
    topic_counts = monthly_df['topic'].value_counts()
    
    if topic_counts.empty:
        logging.warning("Skipping monthly pie chart: no topics found for the latest month.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 10))
    plt.pie(topic_counts, labels=topic_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(topic_counts)))
    plt.title(f'Topic Distribution for {latest_month.strftime("%B %Y")}', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info("Monthly topic pie chart saved.")

if __name__ == "__main__":
    try:
        logging.info("--- Starting Paper Analysis ---")
        papers_df = parse_readme()
        if not papers_df.empty:
            save_to_csv(papers_df)
            
            papers_with_topics_df = analyze_topics(papers_df)
            topic_counts = papers_with_topics_df['topic'].value_counts()
            
            plot_topic_distribution(topic_counts)
            plot_tsa_decomposition(papers_with_topics_df)
            plot_monthly_topic_pie(papers_with_topics_df)
            
            logging.info("--- Analysis complete. CSV and visualizations have been updated. ---")
        else:
            logging.warning("No papers found in README.md. Skipping analysis.")
    except Exception as e:
        logging.error("An error occurred during analysis:")
        logging.error(traceback.format_exc())
