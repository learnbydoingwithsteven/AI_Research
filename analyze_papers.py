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

def plot_topic_distribution(df, save_path='visualizations/topic_distribution.png'):
    """Plots and saves the topic distribution bar chart."""
    logging.info(f"Generating topic distribution plot at {save_path}...")
    topic_counts = df['topic'].value_counts()
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')
    
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 8))
    sns.barplot(x=topic_counts.values, y=topic_counts.index, palette='viridis')
    plt.title(f'Distribution of AI Research Topics\n({start_date} to {end_date})', fontsize=16)
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
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')
    papers_per_day = df.set_index('date').resample('D').size().rename('paper_count')
    period = 7
    if len(papers_per_day) < 2 * period:
        logging.warning(f"Skipping TSA plot: not enough data. Need at least {2*period} days, but have {len(papers_per_day)}.")
        return
    decomposition = seasonal_decompose(papers_per_day, model='additive', period=period)
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.suptitle(f'Time Series Decomposition of Daily Paper Submissions\n({start_date} to {end_date})', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    logging.info("TSA decomposition plot saved.")

def plot_monthly_topic_pies(df):
    """Plots pie charts of topic distribution for each of the last three months."""
    logging.info("Generating monthly topic pie charts for the last 3 months...")
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    if df['month'].empty:
        logging.warning("Skipping monthly pie charts: no data available.")
        return

    end_month = df['month'].max()
    start_month = end_month - 2
    month_range = pd.period_range(start=start_month, end=end_month, freq='M')

    for month in month_range:
        monthly_df = df[df['month'] == month]
        topic_counts = monthly_df['topic'].value_counts()
        save_path = f'visualizations/topic_pie_{month.strftime("%Y_%m")}.png'

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 10))

        if topic_counts.empty:
            logging.warning(f"No topics found for {month}. Generating empty plot.")
            ax.text(0.5, 0.5, f'No Papers Found for\n{month.strftime("%B %Y")}',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=20, color='grey', transform=ax.transAxes)
            ax.set_title(f'Topic Distribution for {month.strftime("%B %Y")}', fontsize=16)
            ax.axis('off')
        else:
            ax.pie(topic_counts, labels=topic_counts.index, autopct='%1.1f%%',
                   startangle=140, colors=sns.color_palette('viridis', len(topic_counts)))
            ax.set_title(f'Topic Distribution for {month.strftime("%B %Y")}', fontsize=16)
            ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Monthly topic pie chart saved for {month.strftime('%B %Y')} at {save_path}.")

def plot_papers_by_month(df, save_path='visualizations/papers_by_month.png'):
    """Plots a bar chart of papers published per month."""
    logging.info(f"Generating papers by month plot at {save_path}...")
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    papers_by_month = df.groupby('month').size()
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')
    
    plt.figure(figsize=(12, 6))
    papers_by_month.index = papers_by_month.index.strftime('%Y-%m')
    ax = papers_by_month.plot(kind='bar', color='steelblue')
    plt.title(f'Papers Published by Month\n({start_date} to {end_date})', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info("Papers by month plot saved.")

def plot_monthly_topic_trends(df, save_path='visualizations/monthly_topic_trends.png'):
    """Plots a stacked bar chart showing topic distribution by month."""
    logging.info(f"Generating monthly topic trends plot at {save_path}...")
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    if df['month'].empty:
        logging.warning("Skipping monthly topic trends plot: no data available.")
        return
        
    end_month = df['month'].max()
    start_month = end_month - 2
    month_range = pd.period_range(start=start_month, end=end_month, freq='M')
    
    filtered_df = df[df['month'].isin(month_range)]
    
    topic_by_month = pd.crosstab(filtered_df['month'], filtered_df['topic'])
    topic_by_month = topic_by_month.reindex(month_range, fill_value=0)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    topic_by_month.index = topic_by_month.index.strftime('%Y-%m')
    
    if topic_by_month.empty and not topic_by_month.columns.any():
        logging.warning("No topics to plot for the last 3 months. Generating empty plot.")
        ax.text(0.5, 0.5, 'No Data Available for the Last 3 Months', 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=16, color='grey')
        ax.set_title('Monthly Topic Distribution (Last 3 Months)', fontsize=16)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
    else:
        topic_by_month.plot(kind='bar', stacked=True, colormap='viridis', ax=ax, width=0.8)
        ax.set_title('Monthly Topic Distribution (Last 3 Months)', fontsize=16)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Topic', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info("Monthly topic trends plot saved.")

def plot_author_collaboration_network(df, save_path='visualizations/author_network.png', max_authors=50):
    """Plots a simple collaboration network of the most frequent authors."""
    logging.info(f"Generating author collaboration network at {save_path}...")
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')
    
    # Extract and count authors
    all_authors = []
    for author_list in df['authors']:
        authors = [a.strip() for a in author_list.split(',')]
        all_authors.extend(authors)
    
    author_counts = pd.Series(all_authors).value_counts()
    top_authors = author_counts.head(max_authors)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_authors.values, y=top_authors.index, palette='viridis')
    plt.title(f'Top {len(top_authors)} Most Prolific Authors\n({start_date} to {end_date})', fontsize=16)
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
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')
    
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(f'AI Research Paper Analysis ({start_date} to {end_date})', fontsize=20, y=1.02)
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Plot 1: Topic Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    topic_counts = df['topic'].value_counts()
    sns.barplot(x=topic_counts.values, y=topic_counts.index, palette='viridis', ax=ax1)
    ax1.set_title('Overall Topic Distribution', fontsize=16)
    ax1.set_xlabel('Number of Papers', fontsize=12)
    ax1.set_ylabel('Topic', fontsize=12)
    
    # Plot 2: Cumulative Papers Over Time
    ax2 = fig.add_subplot(gs[0, 1])
    papers_per_day = df.set_index('date').resample('D').size().cumsum()
    ax2.plot(papers_per_day.index, papers_per_day.values, marker='o', linestyle='-', color='steelblue', markersize=4)
    ax2.set_title('Cumulative Papers Added Over Time', fontsize=16)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Total Papers', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 3: Monthly Topic Trends
    ax3 = fig.add_subplot(gs[1, 0])
    df['month'] = df['date'].dt.to_period('M')
    
    if not df['month'].empty:
        end_month = df['month'].max()
        start_month = end_month - 2
        month_range = pd.period_range(start=start_month, end=end_month, freq='M')
        
        filtered_df = df[df['month'].isin(month_range)]
        
        topic_by_month = pd.crosstab(filtered_df['month'], filtered_df['topic'])
        topic_by_month = topic_by_month.reindex(month_range, fill_value=0)
        topic_by_month.index = topic_by_month.index.strftime('%Y-%m')

        if not topic_by_month.empty:
            topic_by_month.plot(kind='bar', stacked=True, colormap='viridis', ax=ax3, width=0.8)
            ax3.legend(title='Topic', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        else:
            ax3.text(0.5, 0.5, 'No Topic Data', horizontalalignment='center', verticalalignment='center', fontsize=12, color='grey')
    else:
        ax3.text(0.5, 0.5, 'No Date Data Available', horizontalalignment='center', verticalalignment='center', fontsize=12, color='grey')

    ax3.set_title('Topic Distribution (Last 3 Months)', fontsize=16)
    ax3.set_xlabel('Month', fontsize=12)
    ax3.set_ylabel('Number of Papers', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Top Authors
    ax4 = fig.add_subplot(gs[1, 1])
    all_authors = [a.strip() for author_list in df['authors'] for a in author_list.split(',')]
    author_counts = pd.Series(all_authors).value_counts().head(20)
    sns.barplot(x=author_counts.values, y=author_counts.index, palette='mako', ax=ax4)
    ax4.set_title('Top 20 Most Prolific Authors', fontsize=16)
    ax4.set_xlabel('Number of Papers', fontsize=12)
    ax4.set_ylabel('Author', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Dashboard saved.")

if __name__ == "__main__":
    try:
        logging.info("--- Starting Paper Analysis ---")
        papers_df = parse_readme()
        if not papers_df.empty:
            papers_df['date'] = pd.to_datetime(papers_df['date'])
            save_to_csv(papers_df)
            
            papers_with_topics_df = analyze_topics(papers_df)
            
            # Generate individual plots
            plot_topic_distribution(papers_with_topics_df)
            plot_tsa_decomposition(papers_with_topics_df)
            plot_monthly_topic_pies(papers_with_topics_df)
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
