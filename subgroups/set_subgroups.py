"""set_subgroups.py"""
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def get_korean_ratio(text: str) -> float:
    """
    Calculate the ratio of Korean characters in a given string.
    
    Args:
        text (str): Input text to analyze
    
    Returns:
        float: Ratio of Korean characters (0.0 ~ 1.0)
    """
    if not text:
        return 0.0
    def is_korean(char: str) -> bool:
        # Check Korean Unicode ranges
        # 0xAC00-0xD7A3: Completed Korean characters
        # 0x3131-0x314E: Consonants
        # 0x314F-0x3163: Vowels
        code = ord(char)
        return (0xAC00 <= code <= 0xD7A3) or (0x3131 <= code <= 0x3163)

    korean_count = sum(1 for char in text if is_korean(char))
    return korean_count / len(text)

def load_and_preprocess_data(data_dir: str) -> pd.DataFrame:
    """
    Load and preprocess the medical data.
    
    Args:
        data_dir (str): Directory containing the data file
    
    Returns:
        pd.DataFrame: Preprocessed medical data
    """
    df = pd.read_csv(Path(data_dir) / "df_for_annotation.csv")
    # Process dates and calculate age
    df['생년월일'] = pd.to_datetime(df['생년월일'], format='%Y-%m-%d')
    df['서식작성일'] = pd.to_datetime(df['서식작성일'], format='%Y-%m-%d %H:%M:%S')
    df['age'] = df['서식작성일'].dt.year - df['생년월일'].dt.year
    # Adjust age for those who haven't had birthday this year
    not_had_birthday = (
        (df['서식작성일'].dt.month < df['생년월일'].dt.month) |
        ((df['서식작성일'].dt.month == df['생년월일'].dt.month) &
         (df['서식작성일'].dt.day < df['생년월일'].dt.day))
    )
    df.loc[not_had_birthday, 'age'] -= 1

    # Add text length and Korean ratio
    df['text_len'] = df['서식내용'].str.len()
    df['text_kor_ratio'] = df['서식내용'].apply(lambda x: get_korean_ratio(x) * 100)

    return df

def analyze_departments(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Analyze and group documents by department.
    
    Args:
        df (pd.DataFrame): Medical data
    
    Returns:
        Dict[str, List[int]]: Document IDs grouped by department
    """
    department_mapping = {
        'hematology_oncology': '혈액종양내과',
        'surgery': '외과',
        'medical_oncology': '종양내과센터종양내과'
    }
    document_ids = {}
    for key, dept in department_mapping.items():
        document_ids[key] = df[df['환자진료과'] == dept].index.tolist()

    # Add others category
    document_ids['others'] = df[~df['환자진료과'].isin(department_mapping.values())].index.tolist()

    return document_ids

def analyze_demographics(df: pd.DataFrame) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Analyze and group documents by sex and age.
    
    Args:
        df (pd.DataFrame): Medical data
    
    Returns:
        Tuple[Dict[str, List[int]], Dict[str, List[int]]]: Document IDs grouped by sex and age
    """
    sex_groups = {
        'M': df[df['성별'] == 'M'].index.tolist(),
        'F': df[df['성별'] == 'F'].index.tolist()
    }

    age_groups = {
        'under_65': df[df['age'] <= 65].index.tolist(),
        'over_65': df[df['age'] > 65].index.tolist()
    }

    return sex_groups, age_groups

def analyze_document_year(df: pd.DataFrame) -> Tuple[Dict[str, List[int]], \
    Dict[str, List[int]]]:
    """
    Analyze and group documents by text length and Korean ratio.
    
    Args:
        df (pd.DataFrame): Medical data
    
    Returns:
        Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        Document IDs grouped by text length and Korean ratio
    """
    document_year_groups = {
        2010: df[df['서식작성년도'] == 2010].index.tolist(),
        2015: df[df['서식작성년도'] == 2015].index.tolist(),
        2020: df[df['서식작성년도'] == 2020].index.tolist(),
    }

    return document_year_groups

def analyze_text_metrics(df: pd.DataFrame) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Analyze and group documents by text length and Korean ratio.
    
    Args:
        df (pd.DataFrame): Medical data
    
    Returns:
        Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        Document IDs grouped by text length and Korean ratio
    """
    text_len_groups = {
        'under_median_342': df[df['text_len'] <= 342].index.tolist(),
        'over_median_342': df[df['text_len'] > 342].index.tolist()
    }
    korean_ratio_groups = {
        'under_10%': df[df['text_kor_ratio'] < 10].index.tolist(),
        'over_10%': df[df['text_kor_ratio'] >= 10].index.tolist()
    }

    return text_len_groups, korean_ratio_groups

def analyze_note_types(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Analyze and group documents by note type (SOAP).
    
    Args:
        df (pd.DataFrame): Medical data
    
    Returns:
        Dict[str, List[int]]: Document IDs grouped by note type
    """
    return {
        note_type: df[df['노트종류'] == note_type].index.tolist()
        for note_type in ['S', 'O', 'A', 'P']
    }

def save_results(results: Dict[str, Dict[str, List[int]]], output_dir: str):
    """
    Save analysis results to pickle files.
    
    Args:
        results (Dict[str, Dict[str, List[int]]]): Analysis results
        output_dir (str): Directory to save the results
    """
    output_path = Path(output_dir)
    for category, data in results.items():
        with open(output_path / f'document_ids_by_{category}.pkl', 'wb') as f:
            pickle.dump(data, f)

def plot_distributions(df: pd.DataFrame):
    """
    Plot distributions of text length and Korean ratio.
    
    Args:
        df (pd.DataFrame): Medical data
    """
    # Plot text length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='text_len', bins=200)
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length')
    plt.ylabel('Count')
    plt.show()

    # Plot Korean ratio distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='text_kor_ratio', bins=100)
    plt.title('Distribution of Korean Text Ratio')
    plt.xlabel('Korean Text Ratio (%)')
    plt.ylabel('Count')
    plt.show()

def add_diagnosis_history_columns(df, diagnosis_dict):
    """
    Add columns to DataFrame indicating if patient has diagnosis history before note date
    for each diagnosis category. Removes rows for patients not found in diagnosis_dict.
    
    Args:
        df: DataFrame containing patient records with columns ['연구별 환자 ID', '서식작성일']
        diagnosis_dict: Dictionary mapping patient IDs to list of (diagnosis, date) tuples
        
    Returns:
        DataFrame with new columns for each diagnosis category,
        excluding patients not in diagnosis_dict
    """
    # Define all possible diagnosis categories
    diagnosis_categories = {
        'Blood/Immune', 'Cancer_Benign', 'Cancer_Blood', 'Cancer_Bone', 'Cancer_Breast',
        'Cancer_Digestive', 'Cancer_Endocrine', 'Cancer_FemaleGenital', 'Cancer_HeadNeck',
        'Cancer_InSitu', 'Cancer_MaleGenital', 'Cancer_NervousSystem', 'Cancer_Respiratory',
        'Cancer_Skin', 'Cancer_SoftTissue', 'Cancer_Unknown', 'Cancer_Unspecified',
        'Cancer_Urinary', 'Circulatory', 'Congenital', 'Digestive', 'Ear', 'Endocrine', 'Eye',
        'Genitourinary', 'Health_Factors', 'Infectious', 'Injury', 'Mental', 'Musculoskeletal',
        'Neurological', 'Perinatal', 'Pregnancy', 'Respiratory', 'Signs_Symptoms', 'Skin',
        'Special', 'Unknown'}

    # Create copy of DataFrame
    df = df.copy()
    # Filter out patients not in diagnosis_dict
    df = df[df['연구별 환자 ID'].isin(diagnosis_dict.keys())].copy()
    # Convert string date to datetime object
    df['서식작성일'] = pd.to_datetime(df['서식작성일'])

    def check_diagnosis_history(row):
        """
        Helper function to check patient's diagnosis history before note date
        Returns dict with diagnosis flags for each category
        """
        patient_id = row['연구별 환자 ID']
        note_date = row['서식작성일'].date()
        # Initialize diagnosis flags for all categories
        diagnosis_flags = {
            f'has_prior_{category.lower()}': False 
            for category in diagnosis_categories
        }
        # Add general diagnosis flag
        diagnosis_flags['has_prior_diagnosis'] = False

        # Get patient's diagnosis history
        diagnoses = diagnosis_dict[patient_id]  # Now safe to access directly
        # Check each diagnosis
        for diagnosis, diag_date in diagnoses:
            if diag_date < note_date:
                diagnosis_flags['has_prior_diagnosis'] = True
                # Convert diagnosis category to column name format
                column_key = f'has_prior_{diagnosis.lower()}'
                if column_key in diagnosis_flags:
                    diagnosis_flags[column_key] = True

        return diagnosis_flags

    # Apply the check_diagnosis_history function to each row
    # and create new columns from the results
    diagnosis_flags = df.apply(check_diagnosis_history, axis=1)
    diagnosis_flags_df = pd.DataFrame(diagnosis_flags.tolist(), index=df.index)

    # Add new columns to original DataFrame
    for column in diagnosis_flags_df.columns:
        df[column] = diagnosis_flags_df[column]

    return df

def print_diagnosis_stats(df):
    """
    Print statistics about diagnosis occurrences in the dataset
    
    Args:
        df: DataFrame with diagnosis history columns
    """
    print("Diagnosis Statistics:")
    print("-" * 50)
    # Get all diagnosis columns
    diagnosis_cols = [col for col in df.columns if col.startswith('has_prior_')]
    # Calculate and print statistics for each diagnosis type
    for col in sorted(diagnosis_cols):
        count = df[col].sum()
        percentage = (count / len(df)) * 100
        print(f"{col:<30}: {count:>5} cases ({percentage:>6.2f}%)")

def create_simplified_diagnosis_groups(df):
    """
    Create diagnosis groups with the following rules:
    1. Cancer groups: only one type of cancer, can have non-cancer diagnoses
    2. Non-cancer groups: only one type of non-cancer diagnosis, may or may not have cancer
    3. No cancer group: cases with no cancer diagnoses at all
    
    Args:
        df: DataFrame with diagnosis history columns
        
    Returns:
        Dictionary mapping group names to lists of row indices
    """
    # Get all diagnosis columns
    diagnosis_cols = [col for col in df.columns if col.startswith('has_prior_')]
    # Separate cancer and non-cancer columns
    cancer_cols = [col for col in diagnosis_cols
                  if 'cancer' in col.lower()
                  and col != 'has_prior_diagnosis']
    non_cancer_cols = [col for col in diagnosis_cols
                      if 'cancer' not in col.lower()
                      and col != 'has_prior_diagnosis']

    # Initialize groups dynamically
    diagnosis_groups = {
        'no_diagnosis': [],  # Start with no_diagnosis group
        'no_cancer': []      # Add no_cancer group
    }
    # Add cancer-only groups
    for col in cancer_cols:
        group_name = col.replace('has_prior_', '') + '_only'
        diagnosis_groups[group_name] = []
    # Add non-cancer groups with and without cancer
    for col in non_cancer_cols:
        base_name = col.replace('has_prior_', '')
        diagnosis_groups[base_name + '_only'] = []

    # Process each row
    for idx in df.index:
        row = df.loc[idx]
        # Check for no diagnosis first
        if not row['has_prior_diagnosis']:
            diagnosis_groups['no_diagnosis'].append(idx)
            continue

        # Get active diagnoses
        active_cancers = [col for col in cancer_cols if row[col]]
        active_non_cancers = [col for col in non_cancer_cols if row[col]]

        # Check for no cancer cases (but has other diagnoses)
        if not active_cancers and active_non_cancers:
            diagnosis_groups['no_cancer'].append(idx)

        # Process cancer groups - one cancer type only, can have other non-cancer diagnoses
        if len(active_cancers) == 1:
            cancer_col = active_cancers[0]
            group_name = cancer_col.replace('has_prior_', '') + '_only'
            diagnosis_groups[group_name].append(idx)

        # Process non-cancer groups
        for non_cancer_col in active_non_cancers:
            # Check if this is the only non-cancer diagnosis
            other_non_cancers = [col for col in active_non_cancers if col != non_cancer_col]
            if not other_non_cancers:  # No other non-cancer diagnoses
                base_name = non_cancer_col.replace('has_prior_', '')
                if active_cancers:  # Has some cancer
                    diagnosis_groups[base_name + '_only'].append(idx)
                else:  # No cancer
                    diagnosis_groups[base_name + '_only'].append(idx)

    return diagnosis_groups

def print_simplified_group_statistics(df, diagnosis_groups):
    """
    Print statistics about the simplified diagnosis groups
    
    Args:
        df: Original DataFrame
        diagnosis_groups: Dictionary of diagnosis groups and their indices
    """
    total_rows = len(df)
    print("\nDiagnosis Group Statistics:")
    print("-" * 70)
    print(f"{'Group Name':<40} {'Count':>8} {'Percentage':>12}")
    print("-" * 70)

    # Sort groups by type and count
    sorted_groups = sorted(diagnosis_groups.items(), 
                         key=lambda x: (-len(x[1]), x[0]))  # Sort by count (desc) then name
    # Print statistics for each group
    for group_name, indices in sorted_groups:
        count = len(indices)
        percentage = (count / total_rows) * 100
        print(f"{group_name:<40} {count:>8} {percentage:>11.2f}%")
    # Calculate and print summary statistics
    total_assigned = sum(len(indices) for indices in diagnosis_groups.values())
    print("\nSummary:")
    print("-" * 70)
    print(f"Total rows in dataset: {total_rows}")
    print(f"Total rows assigned to groups: {total_assigned}")
    # Some rows might be counted in multiple groups due to our grouping logic
    overlap = total_assigned - total_rows
    if overlap > 0:
        print(f"Rows assigned to multiple groups: {overlap}")
    # Count rows not assigned to any group
    all_assigned_indices = set()
    for indices in diagnosis_groups.values():
        all_assigned_indices.update(indices)
    unassigned = total_rows - len(all_assigned_indices)
    if unassigned > 0:
        print(f"Rows not assigned to any group: {unassigned}")

def analyze_diagnosis_correlation(df, cancer_only=False):
    """
    Create and visualize correlation matrix between different diagnoses
    
    Args:
        df: DataFrame with diagnosis history columns
        cancer_only: If True, only analyze cancer diagnoses. If False, analyze all diagnoses.
        
    Returns:
        correlation matrix DataFrame
    """
    # Get diagnosis columns
    diagnosis_cols = [col for col in df.columns if col.startswith('has_prior_')]
    if cancer_only:
        diagnosis_cols = [col for col in diagnosis_cols if 'cancer' in col.lower()]
        # Exclude the general 'has_prior_diagnosis' column
        diagnosis_cols = [col for col in diagnosis_cols if col != 'has_prior_diagnosis']
    else:
        # Exclude the general 'has_prior_diagnosis' column
        diagnosis_cols = [col for col in diagnosis_cols if col != 'has_prior_diagnosis']

    # Create correlation matrix
    corr_matrix = df[diagnosis_cols].astype(int).corr()
    # Create a larger figure for better readability
    plt.figure(figsize=(20, 16))

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .5})

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # Set title
    title = 'Cancer Diagnosis Correlation Matrix' \
        if cancer_only else 'All Diagnosis Correlation Matrix'
    plt.title(title, pad=20, fontsize=16)
    plt.tight_layout()
    plt.show()
    # Print co-occurrence statistics
    print("\nDiagnosis Co-occurrence Statistics:")
    print("-" * 50)
    for i in range(len(diagnosis_cols)):
        for j in range(i+1, len(diagnosis_cols)):
            col1 = diagnosis_cols[i]
            col2 = diagnosis_cols[j]
            # Calculate co-occurrence
            both = df[df[col1] & df[col2]].shape[0]
            if both > 0:  # Only print if there are co-occurrences
                only1 = df[df[col1] & ~df[col2]].shape[0]
                only2 = df[~df[col1] & df[col2]].shape[0]
                print(f"\n{col1} and {col2}:")
                print(f"Co-occurrence: {both} cases")
                print(f"Only {col1}: {only1} cases")
                print(f"Only {col2}: {only2} cases")

    return corr_matrix

def main():
    """Main function to run the analysis."""
    data_dir = r'C:\Users\username\Documents\projects\ICLforclinicalIE\data'
    data_patient_info = r'C:\Users\username\Documents\projects\ICLforclinicalIE\doc_infos'
    df = load_and_preprocess_data(data_dir)
    with open(f"{data_patient_info}/diagnosis_dict_by_patient_id.pkl", 'rb') as f:
        diagnosis_dict_by_patient_id = pickle.load(f)
    df_diagnosis_added = add_diagnosis_history_columns(df, diagnosis_dict_by_patient_id)
    print_diagnosis_stats(df_diagnosis_added)

    cancer_corr = analyze_diagnosis_correlation(df_diagnosis_added, cancer_only=True)
    all_corr = analyze_diagnosis_correlation(df_diagnosis_added, cancer_only=False)

    groups = create_simplified_diagnosis_groups(df_diagnosis_added)
    print_simplified_group_statistics(df_diagnosis_added, groups)

    document_ids_by_cancer_diagnosis = {
        key: groups[key] for key in ['cancer_blood_only',
                                     'cancer_digestive_only',
                                     'cancer_respiratory_only',
                                     'cancer_breast_only',
                                      'cancer_unknown_only',
                                      'no_cancer']
    }
    assert len(set.union(*[set(ids)
                            for ids in document_ids_by_cancer_diagnosis.values()])) \
                                == sum(len(ids)
                                       for ids in document_ids_by_cancer_diagnosis.values()), \
                                           "Lists contain overlapping elements"
    with open('document_ids_by_cancer_diagnosis.pkl', 'wb') as f:
        pickle.dump(document_ids_by_cancer_diagnosis, f)

    # Grouping by combining related diagnoses to achieve counts > 50
    document_ids_by_diagnosis = {
        'no_diagnosis': groups['no_diagnosis'],  # 140
        'digestive_only': groups['digestive_only'],
        'blood/immune_only': groups['blood/immune_only'],
        'endocrine_only': groups['endocrine_only'],
        'respiratory_and_circulatory': groups['circulatory_only'] + groups['respiratory_only'],
        'others': (groups['signs_symptoms_only'] + 
                   groups['genitourinary_only'] +
                   groups['injury_only'] +
                   groups['neurological_only'] +
                   groups['infectious_only'] +
                   groups['health_factors_only'] +
                   groups['musculoskeletal_only'] +
                   groups['skin_only'] +
                   groups['eye_only'] +
                   groups['mental_only'] +
                   groups['congenital_only'] +
                   groups['ear_only'] +
                   groups['special_only'] +
                   groups['unknown_only'])
    }
    for group_name, ids in document_ids_by_diagnosis.items():
        print(f"{group_name}: {len(ids)} cases")
    assert len(set.union(*[set(ids)
                        for ids in document_ids_by_diagnosis.values()])) \
        == sum(len(ids)
            for ids in document_ids_by_diagnosis.values()), \
        "Lists contain overlapping elements"
    with open('document_ids_by_diagnosis.pkl', 'wb') as f:
        pickle.dump(document_ids_by_diagnosis, f)

    # Perform analyses
    results = {
        'departments': analyze_departments(df),
        'sex': analyze_demographics(df)[0],
        'age': analyze_demographics(df)[1],
        'text_len': analyze_text_metrics(df)[0],
        'textkorratio': analyze_text_metrics(df)[1],
        'notetype': analyze_note_types(df),
        'document_year': analyze_document_year(df)
    }
    # Save results
    save_results(results, '.')
    # Plot distributions
    plot_distributions(df)

    # Print basic statistics
    print(f"Departments distribution: {Counter(df['환자진료과']).most_common(10)}")
    print(f"Sex distribution: {Counter(df['성별'])}")
    print(f"Median text length: {np.median(df['text_len'])}")
    print(f"Median Korean ratio: {np.median(df['text_kor_ratio'])}")

if __name__ == "__main__":
    main()
