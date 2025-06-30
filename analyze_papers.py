import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import traceback
import logging
import os
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(filename='analysis.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')

# Ensure visualizations directory exists
os.makedirs('visualizations', exist_ok=True)

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

def plot_papers_by_month(df, save_path='visualizations/papers_by_month.png'):
    """Plots a bar chart of papers published per month."""
    logging.info(f"Generating papers by month plot at {save_path}...")
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    papers_by_month = df.groupby('month').size()
    
    plt.figure(figsize=(12, 6))
    ax = papers_by_month.plot(kind='bar', color='steelblue')
    plt.title('Papers Published by Month', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info("Papers by month plot saved.")
    return papers_by_month

def plot_monthly_topic_trends(df, save_path='visualizations/monthly_topic_trends.png'):
    """Plots a stacked bar chart showing topic distribution by month."""
    logging.info(f"Generating monthly topic trends plot at {save_path}...")
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    # Get the last 3 months of data
    last_three_months = sorted(df['month'].unique())[-3:]
    filtered_df = df[df['month'].isin(last_three_months)]
    
    if filtered_df.empty:
        logging.warning("Skipping monthly topic trends: not enough monthly data.")
        return
    
    # Create pivot table for stacked bar chart
    topic_by_month = pd.crosstab(filtered_df['month'], filtered_df['topic'])
    
    plt.figure(figsize=(14, 8))
    topic_by_month.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Topic Distribution by Month (Last 3 Months)', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info("Monthly topic trends plot saved.")

def plot_author_collaboration_network(df, save_path='visualizations/author_network.png', max_authors=50):
    """Plots a simple collaboration network of the most frequent authors."""
    logging.info(f"Generating author collaboration network at {save_path}...")
    
    # Extract and count authors
    all_authors = []
    for author_list in df['authors']:
        authors = [a.strip() for a in author_list.split(',')]
        all_authors.extend(authors)
    
    author_counts = pd.Series(all_authors).value_counts()
    top_authors = author_counts.head(max_authors)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_authors.values, y=top_authors.index, palette='viridis')
    plt.title(f'Top {len(top_authors)} Most Prolific Authors', fontsize=16)
    plt.xlabel('Number of Papers', fontsize=12)
    plt.ylabel('Author', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info("Author network plot saved.")

def create_dashboard(df, save_path='visualizations/dashboard.png'):
    """Creates a comprehensive dashboard with multiple plots in a grid layout."""
    logging.info(f"Generating dashboard at {save_path}...")
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Topic Distribution (Top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    topic_counts = df['topic'].value_counts()
    sns.barplot(x=topic_counts.values, y=topic_counts.index, palette='viridis', ax=ax1)
    ax1.set_title('Distribution of AI Research Topics', fontsize=14)
    ax1.set_xlabel('Number of Papers', fontsize=10)
    ax1.set_ylabel('Topic', fontsize=10)
    
    # Plot 2: Papers Over Time (Top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    df_sorted = df.sort_values('date')
    papers_per_day = df_sorted.groupby('date').size().cumsum()
    ax2.plot(papers_per_day.index, papers_per_day.values, marker='o', linestyle='-', color='steelblue')
    ax2.set_title('Cumulative Papers Over Time', fontsize=14)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Number of Papers', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Monthly Topic Distribution (Bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    latest_month = df['date'].max().to_period('M')
    monthly_df = df[df['date'].dt.to_period('M') == latest_month]
    topic_counts = monthly_df['topic'].value_counts()
    ax3.pie(topic_counts, labels=topic_counts.index, autopct='%1.1f%%', 
           startangle=140, colors=sns.color_palette('viridis', len(topic_counts)))
    ax3.set_title(f'Topic Distribution for {latest_month.strftime("%B %Y")}', fontsize=14)
    ax3.axis('equal')
    
    # Plot 4: Papers by Month (Bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    df['month'] = df['date'].dt.to_period('M')
    papers_by_month = df.groupby('month').size()
    papers_by_month.plot(kind='bar', color='steelblue', ax=ax4)
    ax4.set_title('Papers Published by Month', fontsize=14)
    ax4.set_xlabel('Month', fontsize=10)
    ax4.set_ylabel('Number of Papers', fontsize=10)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Dashboard saved.")

if __name__ == "__main__":
    try:
        logging.info("--- Starting Paper Analysis ---")
        papers_df = parse_readme()
        if not papers_df.empty:
            save_to_csv(papers_df)
            
            papers_with_topics_df = analyze_topics(papers_df)
            topic_counts = papers_with_topics_df['topic'].value_counts()
            
            # Generate individual plots
            plot_topic_distribution(topic_counts)
            plot_tsa_decomposition(papers_with_topics_df)
            plot_monthly_topic_pie(papers_with_topics_df)
            plot_papers_by_month(papers_with_topics_df)
            plot_monthly_topic_trends(papers_with_topics_df)
            plot_author_collaboration_network(papers_with_topics_df)
            
            # Generate comprehensive dashboard
            create_dashboard(papers_with_topics_df)
            
            logging.info("--- Analysis complete. CSV and visualizations have been updated. ---")
        else:
            logging.warning("No papers found in README.md. Skipping analysis.")
    except Exception as e:
        logging.error("An error occurred during analysis:")
        logging.error(traceback.format_exc())
