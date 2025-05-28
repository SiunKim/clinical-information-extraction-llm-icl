'''explore_correlations_between_subgroups.py'''
import os
import pickle

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd



DIR_SOUBGROUPS = r'C:\Users\username\Documents\projects\ICLforclinicalIE\subgroups'
def set_doc_infos():
    """Set doc_infos by loading from pickle files and validate integer document IDs"""
    fnames_pkl = [fname for fname in os.listdir(DIR_SOUBGROUPS) if fname.endswith('.pkl')]
    doc_infos = {}
    for fname_pkl in fnames_pkl:
        subgroup_category = fname_pkl.split('by_')[1].split('.pkl')[0]
        with open(f"{DIR_SOUBGROUPS}/{fname_pkl}", 'rb') as f:
            document_ids_by = pickle.load(f)

        # Validate that all values in document_ids_by are integers
        for key, value_list in document_ids_by.items():
            if not all(isinstance(doc_id, int) for doc_id in value_list):
                raise ValueError(f"Non-integer document ID found in category '{subgroup_category}', group '{key}'")
        doc_infos[subgroup_category] = document_ids_by

    return doc_infos
DOC_INFOS = set_doc_infos()

def create_binary_matrix(doc_infos):
    """
    Create a binary matrix where each row represents a document
    and each column represents a subgroup
    """
    # Get all unique document IDs
    all_docs = set()
    subgroup_names = []
    domain_names = []
    
    for domain, subgroups in doc_infos.items():
        domain_names.append(domain)
        for subgroup, docs in subgroups.items():
            all_docs.update(docs)
            subgroup_names.append(f"{domain}_{subgroup}")
    
    # Create binary matrix
    matrix = np.zeros((len(all_docs), len(subgroup_names)))
    doc_list = sorted(list(all_docs))
    
    for i, doc_id in enumerate(doc_list):
        for j, (domain, subgroups) in enumerate(doc_infos.items()):
            for subgroup, docs in subgroups.items():
                col_idx = subgroup_names.index(f"{domain}_{subgroup}")
                if doc_id in docs:
                    matrix[i, col_idx] = 1
                    
    return matrix, subgroup_names, domain_names, doc_list

def analyze_domain_correlations(doc_infos):
    """
    Analyze and visualize correlations between different domains
    """
    domains = list(doc_infos.keys())
    n_domains = len(domains)
    correlation_matrix = np.zeros((n_domains, n_domains))
    
    # Calculate correlation between domains
    for i, domain1 in enumerate(domains):
        for j, domain2 in enumerate(domains):
            docs1 = set()
            docs2 = set()
            
            for docs in doc_infos[domain1].values():
                docs1.update(docs)
            for docs in doc_infos[domain2].values():
                docs2.update(docs)
                
            intersection = len(docs1.intersection(docs2))
            union = len(docs1.union(docs2))
            correlation_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                xticklabels=domains,
                yticklabels=domains,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f')
    plt.title('Domain Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix, domains

def analyze_subgroup_distributions(doc_infos):
    """
    Analyze and visualize the distribution of documents across subgroups
    within each domain
    """
    # Create figure with subplots for each domain
    n_domains = len(doc_infos)
    fig = plt.figure(figsize=(15, n_domains * 3))
    
    for idx, (domain, subgroups) in enumerate(doc_infos.items(), 1):
        # Calculate document counts for each subgroup
        counts = {subgroup: len(docs) for subgroup, docs in subgroups.items()}
        
        # Create subplot
        plt.subplot(n_domains, 1, idx)
        subgroup_names = list(counts.keys())
        doc_counts = list(counts.values())
        
        # Create bar plot
        sns.barplot(x=subgroup_names, y=doc_counts)
        plt.title(f'Document Distribution in {domain}')
        plt.xlabel('Subgroups')
        plt.ylabel('Number of Documents')
        plt.xticks(rotation=45)
        
        # Add count labels on top of bars
        for i, count in enumerate(doc_counts):
            plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
def analyze_cross_domain_distributions(doc_infos):
    """
    Analyze the distribution of documents across different domain combinations
    with percentage calculations for each column and chi-square test p-value
    """
    from scipy.stats import chi2
    domains = list(doc_infos.keys())
    results = {}
    
    for i, domain1 in enumerate(domains):
        for domain2 in domains[i+1:]:
            # Create cross-tabulation matrix
            cross_tab = defaultdict(lambda: defaultdict(int))
            
            # Count document overlap between subgroups
            for subgroup1, docs1 in doc_infos[domain1].items():
                for subgroup2, docs2 in doc_infos[domain2].items():
                    overlap = len(set(docs1) & set(docs2))
                    cross_tab[subgroup1][subgroup2] = overlap
            
            # Convert to DataFrame
            df = pd.DataFrame(cross_tab).fillna(0)
            
            # Calculate percentages for each column
            df_percentages = df.apply(lambda x: x / x.sum() * 100)
            
            # Create a new DataFrame with counts and percentages
            df_with_percentages = pd.DataFrame(index=df.index, columns=df.columns)
            for col in df.columns:
                for idx in df.index:
                    count = int(df.loc[idx, col])
                    percentage = df_percentages.loc[idx, col]
                    df_with_percentages.loc[idx, col] = f"{count} ({percentage:.1f}%)"
            
            # Calculate chi-square test
            chi2_stat = 0
            total = df.values.sum()
            row_sums = df.sum(axis=1)
            col_sums = df.sum(axis=0)
            
            # Calculate degrees of freedom
            df_rows = len(row_sums)
            df_cols = len(col_sums)
            dof = (df_rows - 1) * (df_cols - 1)
            
            # Calculate chi-square statistic and expected frequencies
            expected_freq = np.zeros((df_rows, df_cols))
            for i, row_idx in enumerate(df.index):
                for j, col_idx in enumerate(df.columns):
                    expected = (row_sums[row_idx] * col_sums[col_idx]) / total
                    expected_freq[i, j] = expected
                    if expected > 0:
                        chi2_stat += ((df.loc[row_idx, col_idx] - expected) ** 2) / expected
            
            # Calculate p-value
            p_value = 1 - chi2.cdf(chi2_stat, dof)
            
            results[f"{domain1}_vs_{domain2}"] = {
                'cross_tab': df_with_percentages,
                'raw_counts': df,
                'chi2': chi2_stat,
                'p_value': p_value,
                'dof': dof,
                'expected_frequencies': pd.DataFrame(
                    expected_freq, 
                    index=df.index, 
                    columns=df.columns
                )
            }
            
            # Print results
            print(f"\nCross-tabulation for {domain1} vs {domain2}:")
            print(df_with_percentages)
            print(f"Chi-square statistic: {chi2_stat:.2f}")
            print(f"Degrees of freedom: {dof}")
            print(f"P-value: {p_value:.4f}")
            
            # Print column totals
            col_totals = df.sum(axis=0)
            print("\nColumn totals:")
            for col in df.columns:
                print(f"{col}: {int(col_totals[col])}")
            
            # Add warning for small expected frequencies if any exist
            if (expected_freq < 5).any():
                print("\nWarning: Some expected frequencies are less than 5, "
                      "chi-square test results may not be reliable.")
    
    return results

def analyze_domain_distribution_for_department(doc_infos, target_category, target_value):
    """
    Analyze the distribution of documents across different domains for a specific department
    and print the results as text
    
    Args:
        doc_infos (dict): Dictionary containing document information by domains
        target_category (str): The category to filter by (e.g., 'departments')
        target_value (str): The specific value to filter for (e.g., 'hematology_oncology')
        
    Returns:
        dict: Dictionary containing distribution statistics for each domain
    """
    # Validate inputs
    if target_category not in doc_infos:
        raise ValueError(f"Target category '{target_category}' not found in doc_infos")
        
    if target_value not in doc_infos[target_category]:
        raise ValueError(f"Target value '{target_value}' not found in category '{target_category}'")
    
    # Get target documents
    target_docs = set(doc_infos[target_category][target_value])
    total_docs = len(target_docs)
    
    print(f"\nAnalyzing distribution for {target_value} in {target_category}")
    print(f"Total number of documents: {total_docs}\n")
    
    # Initialize results dictionary
    results = {}
    
    # Analyze each domain except the target category
    for domain, subgroups in doc_infos.items():
        if domain != target_category:
            print(f"\nDistribution across {domain}:")
            print("-" * 50)
            
            domain_stats = {
                'total_docs': total_docs,
                'distribution': defaultdict(int),
                'percentages': defaultdict(float)
            }
            
            # Calculate and print distribution for each subgroup
            for subgroup, docs in subgroups.items():
                overlap = len(target_docs.intersection(set(docs)))
                percentage = (overlap / total_docs) * 100
                
                domain_stats['distribution'][subgroup] = overlap
                domain_stats['percentages'][subgroup] = percentage
                
                print(f"{subgroup:25}: {overlap:4d} documents ({percentage:5.1f}%)")
            
            # Calculate and print unassigned documents if any
            assigned_docs = sum(domain_stats['distribution'].values())
            unassigned = total_docs - assigned_docs
            if unassigned > 0:
                print(f"{'Unassigned':25}: {unassigned:4d} documents ({(unassigned/total_docs)*100:5.1f}%)")
            
            results[domain] = domain_stats
    
    return results
def main(doc_infos):
    """
    Main function to run all analyses
    """
    print("Starting stratification analysis...")

    # Analyze domain correlations
    print("\nAnalyzing domain correlations...")
    correlation_matrix, domains = analyze_domain_correlations(doc_infos)
    
    # Analyze subgroup distributions
    print("\nAnalyzing subgroup distributions...")
    analyze_subgroup_distributions(doc_infos)
    
    # Analyze cross-domain distributions
    print("\nAnalyzing cross-domain distributions...")
    cross_domain_results = analyze_cross_domain_distributions(doc_infos)
    
    print("\nAnalysis complete!")
    
    # Example 1: Analyzing distributions for hematology_oncology
    print("\nAnalyzing distributions for hematology_oncology department:")
    results = analyze_domain_distribution_for_department(
        doc_infos,
        'departments',
        'hematology_oncology'
    )
    # Print summary statistics
    for domain, stats in results.items():
        print(f"\nDistribution for {domain}:")
        print(f"Total documents: {stats['total_docs']}")
        print("Distribution by subgroup:")
        for subgroup, count in stats['distribution'].items():
            percentage = stats['percentages'][subgroup]
            print(f"  {subgroup}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main(DOC_INFOS)