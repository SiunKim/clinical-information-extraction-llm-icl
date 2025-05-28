import os
import pickle
from typing import Dict, Any

from datetime import datetime

import matplotlib.pyplot as plt

from inference_and_test_llm import (
    set_conditions_for_evaluation,
    run_evaluation_for_conditions,
    get_results_filename,
    FINAL_INPUT_PROMPTS_DATE_VERS,
    TEST_SAMPLE_SIZE
)
from settings import PPS

def visualize_results_with_ci(all_results: Dict[str, Any],
                              save_path: str = None,
                              figsize: tuple = (12, 6)):
    """
    Visualize results with confidence intervals for different parameter combinations
    
    Args:
        all_results: Dictionary containing results for each parameter combination
        save_path: Path to save the figure (optional)
        figsize: Figure size as tuple (width, height)
    """
    # Extract and organize data
    data_64 = []
    data_text = []
    penalties = sorted(set([eval(k)['repeat_penalty'] for k in all_results.keys()]))
    for penalty in penalties:
        for params_str, results in all_results.items():
            params = eval(params_str)
            if params['repeat_penalty'] == penalty:
                data_point = {
                    'penalty': penalty,
                    'f1': results['macro_f1'],
                    'ci_lower': results['bootstrap_results']['macro avg']['f1-score']['ci_lower'],
                    'ci_upper': results['bootstrap_results']['macro avg']['f1-score']['ci_upper']
                }
                if params['repeat_last_n'] == 64:
                    data_64.append(data_point)
                else:
                    data_text.append(data_point)

    # Create figure
    plt.figure(figsize=figsize)
    # Plot lines and confidence intervals
    def plot_group(data, color, label):
        x = [d['penalty'] for d in data]
        y = [d['f1'] for d in data]
        ci_lower = [d['ci_lower'] for d in data]
        ci_upper = [d['ci_upper'] for d in data]
        plt.plot(x, y, '-o', color=color, label=label, linewidth=2, markersize=8)
        plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.2)

    # Plot both groups
    plot_group(data_64, 'blue', 'repeat_last_n: 64')
    plot_group(data_text, 'green', 'repeat_last_n: text_length')

    # Customize plot
    plt.xlabel('Repeat Penalty', fontsize=12)
    plt.ylabel('Macro F1 Score', fontsize=12)
    plt.title('F1 Score vs Repeat Penalty with 95% CI', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    # Set y-axis limits with some padding
    all_values = ([d['f1'] for d in data_64] + [d['f1'] for d in data_text] +
                 [d['ci_lower'] for d in data_64] + [d['ci_lower'] for d in data_text] +
                 [d['ci_upper'] for d in data_64] + [d['ci_upper'] for d in data_text])
    plt.ylim(min(all_values) - 0.02, max(all_values) + 0.02)

    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def find_optimal_parameters(conditions, parameter_candidates, inference_n=100):
    """
    Find optimal parameters from candidates based on macro F1 score
    
    Args:
        conditions: Evaluation conditions (should be length 1)
        parameter_candidates: List of parameter dictionaries to test
        inference_n: Number of samples for inference
    
    Returns:
        tuple: (optimal_parameters, best_score, all_results)
    """
    assert len(list(conditions)) == 1, "Conditions length should be 1"
    best_score = -1
    optimal_params = None
    all_results = {}
    print(f"Testing {len(parameter_candidates)} parameter combinations...")
    for params in parameter_candidates:
        print("\nTesting parameters:", params)
        # Run evaluation with current parameters
        run_evaluation_for_conditions(
            conditions=conditions,
            optimal_llm_params=params,
            inference_n=inference_n,
            printout=False
        )
        # Load results for the condition
        results_filename = get_results_filename(
            list(conditions)[0], 
            inference_n,
            temperature=params['temperature'],
            repeat_penalty=params['repeat_penalty'],
            repeat_last_n=params['repeat_last_n']
        )

        full_path = os.path.join(
            PPS.dir_results_llm,
            FINAL_INPUT_PROMPTS_DATE_VERS,
            results_filename
        )
        with open(full_path, 'rb') as f:
            results = pickle.load(f)
        # Get macro F1 score and CI
        macro_f1 = results['class_report']['macro avg']['f1-score']
        print(f"Macro F1 score: {macro_f1:.4f}")

        all_results[str(params)] = {
            'macro_f1': macro_f1,
            'class_report': results['class_report'],
            'confusion_mat': results['confusion_mat'],
            'bootstrap_results': results['bootstrap_results'] 
        }
        # Update best if current is better
        if macro_f1 > best_score:
            best_score = macro_f1
            optimal_params = params.copy()

    print("\nOptimization completed!")
    print(f"Best parameters: {optimal_params}")
    print(f"Best macro F1 score: {best_score:.4f}")

    return optimal_params, best_score, all_results

# Example usage:
TEMPERATURE = 0.001
REPEAT_PENALTIES = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1]
REPEAT_LAST_NS = [64, 'text_length']
OPTIMAL_PARAMS_CANDIDATES = [
    {
        'temperature': TEMPERATURE,
        'repeat_penalty': p,
        'repeat_last_n': n
    }
    for p in REPEAT_PENALTIES
    for n in REPEAT_LAST_NS
]

def main():
    """main"""
    conditions = set_conditions_for_evaluation(
        sample_n=TEST_SAMPLE_SIZE,
        output_types_ner=['entity_list_with_start_end'],
        demonstration_selelction_methods=['Embedding'],
        demonstration_sample_ns=[15]
    )

    # Run optimization
    inference_n = 422
    optimal_params, _, all_results = find_optimal_parameters(
        conditions=conditions,
        parameter_candidates=OPTIMAL_PARAMS_CANDIDATES,
        inference_n=inference_n
    )

    # Save optimal parameters
    date_str = datetime.now().strftime('%m%d')
    with open(f'optimal_params_inferencen{inference_n}_{date_str}.pkl', 'wb') as f:
        pickle.dump(optimal_params, f)
    # Create visualization
    save_path = f'parameter_performance_inferencen{inference_n}_{date_str}.png'
    visualize_results_with_ci(all_results, save_path=save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == '__main__':
    main()
