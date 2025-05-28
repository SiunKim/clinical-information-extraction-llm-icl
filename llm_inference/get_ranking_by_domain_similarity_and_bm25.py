import pickle
import os
import logging
import tqdm

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Settings
ATTRIBUTES = ['age', 'cancer_diagnosis', 'departments', 'diagnosis',
              'document_year', 'notetype', 'sex']

def create_doc_df_from_data(data, doc_infos):
    """Create a dataframe with document IDs, texts and their attributes"""
    # Extract document IDs and texts from data
    doc_data = []
    
    for item in data:
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
    
    # Check if we have all the attributes and count None values
    none_attributes = {}
    for attr in ATTRIBUTES:
        if attr not in df.columns:
            logger.warning(f"Attribute {attr} not found in dataframe, will be added as None")
            df[attr] = None
            none_attributes[attr] = len(df)  # All values are None
        else:
            none_count = df[attr].isna().sum()
            if none_count > 0:
                none_attributes[attr] = none_count
    
    # Print summary of None attributes if any found
    if none_attributes:
        print("\n=== None Attribute Detection Summary ===")
        print(f"Total documents: {len(df)}")
        for attr, count in none_attributes.items():
            percentage = (count / len(df)) * 100
            print(f"Attribute '{attr}': {count} None values ({percentage:.2f}%)")
        print("=======================================\n")
    
    return df

def get_bm25_model(texts):
    """Create BM25 model from texts"""
    logger.info("Creating BM25 model...")
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    return bm25, tokenized_texts

def compute_domain_match_rate(source_row, target_row):
    """Calculate domain match rate between source and target documents"""
    match_count = 0
    valid_attr_count = 0
    
    for attr in ATTRIBUTES:
        if attr in source_row and attr in target_row:
            source_val = source_row[attr]
            target_val = target_row[attr]
            
            # Skip if either value is None
            if source_val is None or target_val is None:
                continue
                
            valid_attr_count += 1
            if source_val == target_val:
                match_count += 1
    
    # Return match rate if there are valid attributes, otherwise 0
    return match_count / valid_attr_count if valid_attr_count > 0 else 0

def generate_domain_rankings(df):
    """Generate domain-based rankings for all documents"""
    logger.info("Generating domain-based rankings...")
    
    # Create BM25 model for text similarity
    texts = df['text'].tolist()
    doc_ids = df['doc_id'].tolist()
    bm25_model, tokenized_texts = get_bm25_model(texts)
    
    # Dictionary to store rankings
    rankings = {}
    
    # Process each document as an anchor
    for idx, anchor_row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
        anchor_id = anchor_row['doc_id']
        anchor_text_tokens = tokenized_texts[idx]
        
        # Calculate BM25 scores once for this anchor
        bm25_scores = bm25_model.get_scores(anchor_text_tokens)
        
        # Store document info with domain match rates and BM25 scores
        doc_info = []
        
        for target_idx, target_row in df.iterrows():
            target_id = target_row['doc_id']
            
            # Skip the anchor document itself
            if target_id == anchor_id:
                continue
                
            # Calculate domain match rate
            match_rate = compute_domain_match_rate(anchor_row, target_row)
            
            # Get BM25 score
            bm25_score = bm25_scores[target_idx]
            
            # Add to list
            doc_info.append({
                'doc_id': target_id,
                'match_rate': match_rate,
                'bm25_score': bm25_score
            })
        
        # Sort by match rate (descending) and then by BM25 score (descending)
        sorted_docs = sorted(doc_info, key=lambda x: (x['match_rate'], x['bm25_score']), reverse=True)
        
        # Store ranked document IDs
        rankings[anchor_id] = [doc['doc_id'] for doc in sorted_docs]
    
    return rankings

def generate_domain_rankings_by_group(df):
    """Generate domain-based rankings grouped by match rate"""
    logger.info("Generating domain-based rankings with grouping...")
    
    # Create BM25 model for text similarity
    texts = df['text'].tolist()
    doc_ids = df['doc_id'].tolist()
    bm25_model, tokenized_texts = get_bm25_model(texts)
    
    # Dictionary to store rankings
    rankings = {}
    
    # Process each document as an anchor
    for idx, anchor_row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
        anchor_id = anchor_row['doc_id']
        anchor_text_tokens = tokenized_texts[idx]
        
        # Calculate BM25 scores once for this anchor
        bm25_scores = bm25_model.get_scores(anchor_text_tokens)
        
        # Group documents by match rate
        match_rate_groups = defaultdict(list)
        
        for target_idx, target_row in df.iterrows():
            target_id = target_row['doc_id']
            
            # Skip the anchor document itself
            if target_id == anchor_id:
                continue
                
            # Calculate domain match rate
            match_rate = compute_domain_match_rate(anchor_row, target_row)
            
            # Get BM25 score
            bm25_score = bm25_scores[target_idx]
            
            # Add to appropriate match rate group
            match_rate_groups[match_rate].append({
                'doc_id': target_id,
                'bm25_score': bm25_score
            })
        
        # Sort each match rate group by BM25 score
        for match_rate in match_rate_groups:
            match_rate_groups[match_rate] = sorted(
                match_rate_groups[match_rate],
                key=lambda x: x['bm25_score'],
                reverse=True
            )
        
        # Flatten the groups in descending order of match rate
        ranked_docs = []
        for match_rate in sorted(match_rate_groups.keys(), reverse=True):
            for doc in match_rate_groups[match_rate]:
                ranked_docs.append(doc['doc_id'])
        
        # Store ranked document IDs
        rankings[anchor_id] = ranked_docs
    
    return rankings

def save_rankings(rankings, filename="domain_rankings.pkl"):
    """Save rankings to a pickle file"""
    logger.info(f"Saving rankings to {filename}...")
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, 'wb') as f:
        pickle.dump(rankings, f)
    
    logger.info(f"Rankings saved with {len(rankings)} anchor documents")

def main(data, doc_infos, output_file="results/domain_rankings.pkl", group_by_match_rate=True):
    """Main function to generate domain-based selection rankings"""
    logger.info("Starting domain-based selection ranking generation...")
    
    # Create dataframe with document data
    df = create_doc_df_from_data(data, doc_infos)
    logger.info(f"Prepared data with {len(df)} documents")
    
    # Generate rankings
    if group_by_match_rate:
        rankings = generate_domain_rankings_by_group(df)
    else:
        rankings = generate_domain_rankings(df)
    
    # Save rankings
    save_rankings(rankings, output_file)
    
    return rankings

if __name__ == "__main__":
    from utils import import_train_valid_test_data_by_source, DOC_INFOS
    
    # Import data
    train_data, valid_data, test_data = \
        import_train_valid_test_data_by_source(document_source='inhouse')
    total_data = train_data + valid_data + test_data
    
    # Generate and save rankings
    rankings = main(total_data, doc_infos=DOC_INFOS)
    
    # Print sample of rankings
    sample_id = list(rankings.keys())[0]
    print(f"\nSample ranking for document {sample_id}:")
    print(f"Top 10 ranked documents: {rankings[sample_id][:10]}")