import logging
import os
import pickle
import random

from collections import Counter

import tqdm
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy as shannon_entropy

import torch
from transformers import AutoTokenizer, AutoModel

from rank_bm25 import BM25Okapi

import nltk
from nltk.tokenize import word_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download nltk resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Global Settings
TOP_K_VALUES = [3, 5, 10, 15, 20]
ATTRIBUTES = ['age', 'cancer_diagnosis', 'departments', 'diagnosis',
              'document_year', 'notetype', 'sex']

SIMILARITY_METHODS = [
    'random', 
    'tfidf', 
    'bm25', 
    'embedding_roberta', 
    'embedding_clinicalbert',
]

EMBEDDING_MODELS = {
    'embedding_roberta': 'FacebookAI/roberta-large',
    'embedding_clinicalbert': 'medicalai/ClinicalBERT',
}

def create_doc_df_from_data(train_data, doc_infos):
    """Create a dataframe with document IDs, texts and their attributes"""
    # Extract document IDs and texts from train_data
    doc_data = []
    
    for item in train_data:
        doc_id = item['document_id']
        text = item['text']
        
        # Create a record for this document
        record = {'doc_id': doc_id, 'text': text}
        
        # Add attributes for this document
        for attribute, categories in doc_infos.items():
            for category, id_list in categories.items():
                if doc_id in id_list:
                    record[attribute] = category
                    break
        
        doc_data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(doc_data)
    
    # Check if we have all the attributes
    for attr in ATTRIBUTES:
        if attr not in df.columns:
            logger.warning(f"Attribute {attr} not found in dataframe, will be added as None")
            df[attr] = None
    
    return df

def get_tfidf_similarity(texts):
    """Calculate TF-IDF based similarity matrix"""
    logger.info("Calculating TF-IDF similarity...")
    tfidf = TfidfVectorizer()
    try:
        vectors = tfidf.fit_transform(texts)
        sim_matrix = cosine_similarity(vectors)
        return sim_matrix
    except Exception as e:
        logger.error(f"Error in TF-IDF calculation: {e}")
        return None

def get_bm25_similarity(texts):
    """Calculate BM25 based similarity matrix"""
    logger.info("Calculating BM25 similarity...")
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    
    bm25 = BM25Okapi(tokenized_texts)
    
    n = len(texts)
    sim_matrix = np.zeros((n, n))
    
    for i, text in enumerate(tqdm.tqdm(tokenized_texts, desc="BM25 Processing")):
        scores = bm25.get_scores(text)
        sim_matrix[i] = scores
    
    return sim_matrix

def get_embedding_similarity(texts, model_name):
    """Calculate embedding based similarity matrix"""
    logger.info(f"Calculating embedding similarity using {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    embeddings = []
    batch_size = 8
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                               max_length=512, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    
    all_embeddings = np.vstack(embeddings)
    sim_matrix = cosine_similarity(all_embeddings)
    return sim_matrix

def get_similarity_matrix(method, texts):
    """Get similarity matrix based on the method - with caching capability"""
    # Create similarity matrices directory if it doesn't exist
    save_dir = 'similarity_matrices'
    os.makedirs(save_dir, exist_ok=True)
    matrix_file = os.path.join(save_dir, f'{method}_similarity.pkl')
    
    # Check if similarity matrix already exists
    if os.path.exists(matrix_file):
        logger.info(f"Loading existing similarity matrix for {method} from {matrix_file}")
        with open(matrix_file, 'rb') as f:
            return pickle.load(f)
    
    # Otherwise calculate the similarity matrix
    logger.info(f"Calculating similarity matrix for {method}...")
    
    if method == 'random':
        return None
    elif method == 'tfidf':
        sim_matrix = get_tfidf_similarity(texts)
    elif method == 'bm25':
        sim_matrix = get_bm25_similarity(texts)
    elif method.startswith('embedding_'):
        model_name = EMBEDDING_MODELS[method]
        sim_matrix = get_embedding_similarity(texts, model_name)
    else:
        raise NotImplementedError(f"Method '{method}' not implemented")
    
    # Save the calculated similarity matrix
    with open(matrix_file, 'wb') as f:
        pickle.dump(sim_matrix, f)
    logger.info(f"Saved {method} similarity matrix to {matrix_file}")
    
    return sim_matrix

def compute_topk_neighbors(sim_matrix, doc_ids, method, k):
    """Compute top-k neighbors for each document"""
    neighbors = {}
    n = len(doc_ids)
    
    for i, doc_id in enumerate(doc_ids):
        if method == 'random':
            others = list(set(doc_ids) - {doc_id})
            neighbors[doc_id] = random.sample(others, min(k, len(others)))
        else:
            sims = sim_matrix[i]
            indices = np.argsort(sims)[::-1]
            if indices[0] == i:  # Skip self
                indices = indices[1:k+1]
            else:
                indices = indices[:k]
            neighbors[doc_id] = [doc_ids[j] for j in indices]
    
    return neighbors

def compute_domain_match_rate(source_val, neighbor_vals):
    """Calculate Domain Match Rate (DMR) - Previously DSFS"""
    if source_val is None or not neighbor_vals:
        return np.nan
    
    valid_neighbors = [v for v in neighbor_vals if v is not None]
    if not valid_neighbors:
        return np.nan
    
    return np.mean([v == source_val for v in valid_neighbors])

def compute_normalized_entropy(values):
    """Compute normalized Shannon entropy of a distribution"""
    valid_values = [v for v in values if v is not None]
    if not valid_values or len(valid_values) <= 1:
        return np.nan
    
    counter = Counter(valid_values)
    probs = np.array(list(counter.values())) / len(valid_values)
    raw_entropy = shannon_entropy(probs)
    
    # Normalize by maximum possible entropy (log(k))
    max_entropy = np.log(len(valid_values))
    if max_entropy > 0:
        return raw_entropy / max_entropy
    else:
        return 0.0

def evaluate_method_optimized(method, df, top_k_values):
    """Evaluate a similarity method for multiple k values with a single similarity matrix calculation"""
    logger.info(f"Evaluating {method} for all k values...")
    
    # Convert DataFrame to lists for easier processing
    texts = df['text'].tolist()
    doc_ids = df['doc_id'].tolist()
    
    # Get similarity matrix (calculate once)
    sim_matrix = get_similarity_matrix(method, texts) if method != 'random' else None
    
    # Store results for all k values
    all_k_results = []
    
    # Process each k value
    for top_k in top_k_values:
        logger.info(f"Processing {method} with top-k={top_k}...")
        
        # Find neighbors for each document for this k value
        neighbors = compute_topk_neighbors(sim_matrix, doc_ids, method, top_k)
        
        # Store results for each document
        results = []
        for idx, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Processing {method} k={top_k}"):
            doc_id = row['doc_id']
            neigh_ids = neighbors.get(doc_id, [])
            
            if not neigh_ids:
                continue
                
            # Get neighbor attributes
            neigh_rows = df[df['doc_id'].isin(neigh_ids)]
            
            # Create a record for this document
            record = {
                'method': method,
                'top_k': top_k,
                'doc_id': doc_id
            }
            
            # Evaluate each attribute
            for attr in ATTRIBUTES:
                if attr in row and attr in neigh_rows:
                    val = row[attr]
                    neigh_vals = neigh_rows[attr].tolist()
                    
                    # Calculate metrics
                    record[f'{attr}_dmr'] = compute_domain_match_rate(val, neigh_vals)
                    record[f'{attr}_norm_entropy'] = compute_normalized_entropy(neigh_vals)
            
            results.append(record)
        
        # Convert to DataFrame
        k_results = pd.DataFrame(results)
        all_k_results.append(k_results)
        
        # Save intermediate results
        os.makedirs('results', exist_ok=True)
        k_results.to_csv(f'results/domain_metrics_{method}_{top_k}.csv', index=False)
        logger.info(f"Saved intermediate results for {method} with k={top_k}")
    
    return pd.concat(all_k_results) if all_k_results else pd.DataFrame()

def run_evaluation(method, df):
    """Run evaluation for a single method and all k values - optimized version"""
    try:
        # Evaluate all k values at once with a single similarity matrix calculation
        results = evaluate_method_optimized(method, df, TOP_K_VALUES)
        return results
    except Exception as e:
        logger.error(f"Error evaluating {method} for all k values: {e}")
        return pd.DataFrame()

def run_all_methods(df):
    """Run evaluation for all methods"""
    all_results = []
    
    for method in SIMILARITY_METHODS:
        method_results = run_evaluation(method, df)
        all_results.append(method_results)
        
        # Save combined results so far
        combined = pd.concat(all_results)
        combined.to_csv(f'results/domain_metrics_combined.csv', index=False)
        
        # Also save as pickle for easier loading
        with open('results/domain_metrics_combined.pkl', 'wb') as f:
            pickle.dump(combined, f)
            
        logger.info(f"Added results for {method} to combined results")
    
    return pd.concat(all_results) if all_results else pd.DataFrame()

def summarize_results(all_df):
    """Summarize evaluation results by method and top_k"""
    if all_df.empty:
        logger.warning("Cannot summarize empty results dataframe")
        return pd.DataFrame()
    
    # Get all columns related to our metrics
    dmr_cols = [col for col in all_df.columns if col.endswith('_dmr')]
    entropy_cols = [col for col in all_df.columns if col.endswith('_norm_entropy')]
    
    # Group by method and top_k, then calculate average metrics
    summary = all_df.groupby(['method', 'top_k']).agg({
        **{col: 'mean' for col in dmr_cols},
        **{col: 'mean' for col in entropy_cols},
    }).reset_index()
    
    # Calculate aggregated metrics across all attributes
    summary['avg_dmr'] = summary[[col for col in dmr_cols]].mean(axis=1)
    summary['avg_norm_entropy'] = summary[[col for col in entropy_cols]].mean(axis=1)
    
    return summary

def create_attribute_summary(all_df):
    """Create a summary table for each attribute across all methods"""
    if all_df.empty:
        logger.warning("Cannot create attribute summary from empty results dataframe")
        return pd.DataFrame()
        
    # List to store attribute-level summaries
    attr_summaries = []
    
    # Process each attribute
    for attr in ATTRIBUTES:
        dmr_col = f'{attr}_dmr'
        entropy_col = f'{attr}_norm_entropy'
        
        # Skip if columns don't exist
        if dmr_col not in all_df.columns or entropy_col not in all_df.columns:
            continue
            
        # Group by method and top_k and calculate average metrics for this attribute
        attr_summary = all_df.groupby(['method', 'top_k']).agg({
            dmr_col: 'mean',
            entropy_col: 'mean'
        }).reset_index()
        
        # Rename columns to include attribute name
        attr_summary.rename(columns={
            dmr_col: f'dmr',
            entropy_col: f'norm_entropy'
        }, inplace=True)
        
        # Add attribute column
        attr_summary['attribute'] = attr
        
        # Append to list of summaries
        attr_summaries.append(attr_summary)
    
    # Combine all attribute summaries
    if attr_summaries:
        return pd.concat(attr_summaries)
    else:
        return pd.DataFrame()

def save_attribute_summary_tables(attr_summary_df, output_dir='results'):
    """Save attribute summary tables to CSV files and create formatted tables"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete attribute summary
    attr_summary_df.to_csv(f'{output_dir}/attribute_summary.csv', index=False)
    logger.info(f"Saved complete attribute summary to {output_dir}/attribute_summary.csv")
    
    # Create separate tables for each k value
    for k in TOP_K_VALUES:
        k_data = attr_summary_df[attr_summary_df['top_k'] == k]
        
        # Skip if no data for this k
        if k_data.empty:
            continue
        
        # Create pivot tables for domain match rate and entropy
        dmr_pivot = k_data.pivot_table(
            index='attribute', 
            columns='method', 
            values='dmr'
        ).reset_index()
        
        entropy_pivot = k_data.pivot_table(
            index='attribute', 
            columns='method', 
            values='norm_entropy'
        ).reset_index()
        
        # Save to CSV
        dmr_pivot.to_csv(f'{output_dir}/dmr_by_attribute_k{k}.csv', index=False)
        entropy_pivot.to_csv(f'{output_dir}/entropy_by_attribute_k{k}.csv', index=False)
        
        # Print formatted tables
        print(f"\nDomain Match Rate by Attribute (k={k}):")
        print(dmr_pivot.to_string(index=False))
        
        print(f"\nNormalized Entropy by Attribute (k={k}):")
        print(entropy_pivot.to_string(index=False))
        
    logger.info(f"Saved attribute summary tables for all k values")

def plot_metrics_combined(summary_df, output_dir='results'):
    """Create combined subplot for DMR and normalized entropy metrics"""
    import matplotlib.pyplot as plt
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    
    os.makedirs(output_dir, exist_ok=True)

    # Define colors for each method
    colors = {
        'random': 'gray',
        'tfidf': 'blue',
        'bm25': 'green',
        'embedding_roberta': 'red',
        'embedding_mpnet': 'purple',
        'embedding_clinicalbert': 'orange',
        'embedding_biomedrobertabase': 'magenta',
    }
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Plot for Domain Match Rate (DMR)
    for method in summary_df['method'].unique():
        data = summary_df[summary_df['method'] == method]
        # Sort by top_k to ensure correct line plotting
        data = data.sort_values('top_k')
        ax1.plot(data['top_k'], data['avg_dmr'], marker='o', label=method, color=colors.get(method, 'black'))
    
    ax1.set_ylabel('Domain Match Rate (DMR)')
    ax1.set_title('Domain Match Rate (DMR) ↑')  # Up arrow for "higher is better"
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot for Normalized Entropy
    for method in summary_df['method'].unique():
        data = summary_df[summary_df['method'] == method]
        # Sort by top_k to ensure correct line plotting
        data = data.sort_values('top_k')
        ax2.plot(data['top_k'], data['avg_norm_entropy'], marker='o', label=method, color=colors.get(method, 'black'))
    
    ax2.set_xlabel('Top-K Values')
    ax2.set_ylabel('Normalized Entropy')
    ax2.set_title('Normalized Entropy ↓')  # Down arrow for "lower is better"
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Place legend outside of plot box (right side)
    # We only need one legend for the entire figure
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.2), loc='center left', borderaxespad=0)
    
    plt.tight_layout()
    
    # Save as high-resolution PNG
    plt.savefig(f'{output_dir}/combined_metrics_plot.png', dpi=600, bbox_inches='tight')
    
    # Save as PDF with text as text (not as paths)
    with PdfPages(f'{output_dir}/combined_metrics_plot.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    
    return fig

def main(train_data, doc_infos):
    """Main function to run the evaluation"""
    logger.info("Starting domain structure evaluation...")
    
    # Create dataframe with document data
    df = create_doc_df_from_data(train_data, doc_infos)
    logger.info(f"Prepared data with {len(df)} documents")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run evaluation for all methods
    all_results = run_all_methods(df)
    
    # Save all results
    all_results.to_csv('results/domain_metrics_all_results.csv', index=False)
    with open('results/domain_metrics_all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"Saved all results with {len(all_results)} rows")
    
    # Summarize results
    summary = summarize_results(all_results)
    
    # Save summary
    summary.to_csv('results/domain_metrics_summary.csv', index=False)
    with open('results/domain_metrics_summary.pkl', 'wb') as f:
        pickle.dump(summary, f)
    logger.info("Saved summary results")
    
    # Create attribute-level summary
    attr_summary = create_attribute_summary(all_results)
    
    # Save attribute tables and print formatted results
    save_attribute_summary_tables(attr_summary)
    
    # Create plots
    plot_metrics_combined(summary)
    
    return all_results, summary, attr_summary

if __name__ == "__main__":
    from utils import import_train_valid_test_data_by_source, DOC_INFOS
    
    # Import data
    train_data, valid_data, test_data = \
        import_train_valid_test_data_by_source(document_source='inhouse')
    total_data = train_data + valid_data + test_data
    
    # Run evaluation
    all_results, summary, attr_summary = main(total_data, doc_infos=DOC_INFOS)
    
    # Print overall summary
    print("\nOverall Summary of Domain Metrics:")
    print(summary[['method', 'top_k', 'avg_dmr', 'avg_norm_entropy']].to_string(index=False))
