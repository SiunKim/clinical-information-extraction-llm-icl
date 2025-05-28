"""
BERT model finetuning implementation for Named Entity Recognition (NER) and Relation Extraction (RE).
Includes hyperparameter tuning, model training, evaluation, and inference capabilities.
"""
import glob
import os
import pickle
from pathlib import Path
import copy
from itertools import product
from typing import Dict, Tuple, List, Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from preprocess_dataset import (
    preprocessing_bert_dataset,
    DOC_INFOS
    )
from models import NERModel, REModel, BiLSTM_CRF_NER  
from utils_bertfinetuning import (
    set_basefilename,
    save_results,
    save_results_excel,
    plot_losses,
    print_metrics,
    calculate_ner_metrics,
    calculate_re_metrics,
    load_best_model,
    inference,
    check_existing_hyperparameters,
    set_subgroups_from_to_and_size_n
)



# Global variables
MODEL_TYPE = 'BasicMLP' # or 'BiLSTM+CRF' 
TRAIN_ON_RE = False
DIR_BASE = r'C:\Users\username\Documents\projects\ICLforclinicalIE\bert_finetuning'
DIR_MODELS = f'{DIR_BASE}/best_models'
INDEPENDENT_TUNING = True
TEST_IN_BEST_TWO_ANNOTATORS = False
BEST_PARAMS_DEFULAT = {
    'epochs': 100,
    'batch_size': 256,
    'lr': 1e-05,
    'hidden_dim': 512,
    'dropout_rate': 0.3,
    'weight_decay': 0
    }
DISCHARGE_SUMMARY = False
TEST_ON_DISCHARGE = True


# Classes
class ModelConfig:
    """Configuration class for model hyperparameters and settings"""
    BERT_MODELS = [
        # 'kimsiun/kaers-bert-241101',
        # 'madatnlp/km-bert',
        # 'monologg/koelectra-small-v2-distilled-korquad-384',
        # 'kykim/electra-kor-base',
        'FacebookAI/xlm-roberta-large',
        ]
    SAMPLE_SIZES = [
        1480,
        # 50, 100, 150, 200, 400, 600, 800, 1000, 1200,
        ]
    HYPERPARAMS = {
        'EPOCHS': [100],
        'DROPOUTS': [0, 0.3],
        'BATCH_SIZES': [256],
        'LEARNING_RATES': [1e-5, 2e-6],
        'HIDDEN_DIMS': [256, 512],
        'WEIGHT_DECAYS': [0],
    }

class BERTTrainer:
    """Main trainer class for BERT-based NER and RE models"""
    def __init__(self, device: torch.device = None):
        """
        Initialize trainer with specified device
        
        Args:
            device: torch.device for model training (default: auto-detect)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_and_evaluate(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        task: str,
        tag_to_index: Dict[str, int],
        epochs: int = 10,
        patience: int = 3
    ) -> Tuple[nn.Module, Dict[str, float], List[float], List[float]]:
        """
        Train and evaluate model with early stopping
        
        Args:
            model: Neural network model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            task: Task type ('ner' or 're')
            tag_to_index: Mapping of tags to indices
            epochs: Maximum number of training epochs
            patience: Early stopping patience
            
        Returns:
            Tuple of (best_model, metrics, train_losses, val_losses)
        """
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_model = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            train_losses.append(epoch_train_loss)

            # Validation phase
            model.eval()
            epoch_val_loss, all_preds, all_labels = self._validate_epoch(
                model, val_loader, criterion)
            val_losses.append(epoch_val_loss)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {epoch_train_loss:.4f}")
            print(f"Validation Loss: {epoch_val_loss:.4f}")

            # Early stopping check
            if self._check_early_stopping(epoch_val_loss, best_val_loss):
                best_val_loss = epoch_val_loss
                best_model = copy.deepcopy(model)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                    break

        metrics = (calculate_ner_metrics(all_preds, all_labels, tag_to_index)
                    if task == 'ner' else calculate_re_metrics(all_preds, all_labels))

        return best_model, metrics, train_losses, val_losses

    def _train_epoch(self, model, train_loader, criterion, optimizer):
        epoch_loss = 0
        for batch in train_loader:
            inputs, labels = batch[:2]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            if MODEL_TYPE == 'BiLSTM+CRF':
                mask = batch[2].to(self.device) if len(batch) > 2 else None
                loss = model.loss(inputs, labels, mask=mask)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)


    def _validate_epoch(self, model, val_loader, criterion):
        epoch_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch[:2]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if MODEL_TYPE == 'BiLSTM+CRF':
                    mask = batch[2].to(self.device) if len(batch) > 2 else None
                    loss = model.loss(inputs, labels, mask=mask)
                    preds = model(inputs, mask=mask)
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().tolist())
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                epoch_loss += loss.item()
        return epoch_loss / len(val_loader), all_preds, all_labels

    def train_models_with_tuning(self,
                                 bert_name,
                                 n_samples_for_train,
                                 use_bio,
                                 train_dataset_ner,
                                 train_dataset_re,
                                 valid_dataset_ner,
                                 valid_dataset_re,
                                 tag_to_index,
                                 model_manager,
                                 best_params=None):
        """train_models_with_tuning with optional best_params"""
        def setup_training(model, lr, weight_decay):
            """setup_training"""
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            return criterion, optimizer

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_filename, _ = set_basefilename(bert_name, n_samples_for_train, use_bio)
        save_dir = Path(os.path.join(DIR_MODELS, base_filename))
        save_dir.mkdir(parents=True, exist_ok=True)

        best_ner_model, best_re_model = check_existing_hyperparameters(save_dir,
                                                                       ModelConfig.HYPERPARAMS)
        if isinstance(best_ner_model, torch.nn.Module) \
            and isinstance(best_re_model, torch.nn.Module):
            return best_ner_model, best_re_model
        best_ner_model, best_re_model = None, None
        best_ner_score, best_re_score = 0, 0
        best_overall_params = None
        all_results = []

        # If best_params is provided, use only those parameters
        if best_params is not None:
            param_combinations = [best_params]
        else:
            # Generate all combinations of hyperparameters
            param_combinations = [
                {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'hidden_dim': hidden_dim,
                    'dropout_rate': dropout_rate,
                    'weight_decay': weight_decay
                }
                for epochs, batch_size, lr, hidden_dim, dropout_rate, weight_decay in
                product(
                    ModelConfig.HYPERPARAMS['EPOCHS'],
                    ModelConfig.HYPERPARAMS['BATCH_SIZES'],
                    ModelConfig.HYPERPARAMS['LEARNING_RATES'],
                    ModelConfig.HYPERPARAMS['HIDDEN_DIMS'],
                    ModelConfig.HYPERPARAMS['DROPOUTS'],
                    ModelConfig.HYPERPARAMS['WEIGHT_DECAYS']
                )
            ]

        for params in param_combinations:
            print(f"Training with params: {params}")
            train_loader_ner, train_loader_re, val_loader_ner, val_loader_re = \
                self.create_data_loaders(
                    train_dataset_ner,
                    train_dataset_re,
                    valid_dataset_ner,
                    valid_dataset_re,
                    params['batch_size']
                )

            ner_model, re_model = \
                self.initialize_models(
                    train_dataset_ner,
                    train_dataset_re,
                    params['hidden_dim'],
                    tag_to_index,
                    params['dropout_rate'],
                    device
                )

            criterion_ner, optimizer_ner = \
                setup_training(ner_model, params['lr'], params['weight_decay'])
            criterion_re, optimizer_re = \
                setup_training(re_model, params['lr'], params['weight_decay'])

            ner_model, ner_metrics, ner_train_losses, ner_val_losses = \
                self.train_and_evaluate(
                    ner_model,
                    train_loader_ner,
                    val_loader_ner,
                    criterion_ner,
                    optimizer_ner,
                    'ner', 
                    tag_to_index,
                    epochs=params['epochs'],
                    patience=10
                )
            if TRAIN_ON_RE:
                re_model, re_metrics, re_train_losses, re_val_losses = \
                    self.train_and_evaluate(
                        re_model,
                        train_loader_re,
                        val_loader_re,
                        criterion_re,
                        optimizer_re,
                        're', 
                        tag_to_index,
                        epochs=params['epochs'],
                        patience=10
                    )
                if re_metrics['f1'] > best_re_score:
                    best_re_score = re_metrics['f1']
                    best_re_model = copy.deepcopy(re_model)
                    best_re_params = params
                    best_re_metrics = re_metrics
                    model_manager.save_best_model(
                        best_re_model,
                        best_re_params,
                        best_re_metrics,
                        n_samples_for_train,
                        're', 
                    )
            else:
                re_model = None
                re_metrics = {}
                re_train_losses, re_val_losses = [], []

            result = {
                'params': params,
                'ner_score': ner_metrics['f1_macro'],
                're_score': re_metrics['f1'],
            }
            all_results.append(result)

            plot_losses(ner_train_losses, ner_val_losses,
                        re_train_losses, re_val_losses, params, save_dir)
            print_metrics(ner_metrics, re_metrics)

            if ner_metrics['f1_macro'] > best_ner_score:
                best_ner_score = ner_metrics['f1_macro']
                best_ner_model = copy.deepcopy(ner_model)
                best_ner_params = params
                best_ner_metrics = ner_metrics
                model_manager.save_best_model(
                    best_ner_model,
                    best_ner_params,
                    best_ner_metrics,
                    n_samples_for_train,
                    'ner', 
                )
                best_overall_params = params
            if re_metrics['f1'] > best_re_score:
                best_re_score = re_metrics['f1']
                best_re_model = copy.deepcopy(re_model)
                best_re_params = params
                best_re_metrics = re_metrics
                model_manager.save_best_model(
                    best_re_model,
                    best_re_params,
                    best_re_metrics,
                    n_samples_for_train,
                    're', 
                )

        # Save all results and best parameters
        bert_name_clean = bert_name.replace('/', '-')
        best_params_file = save_dir / f'best_params_settings_{bert_name_clean}_{MODEL_TYPE}.pkl'
        with open(best_params_file, 'wb') as f:
            pickle.dump({
                'best_params': best_overall_params,
                'best_ner_score': best_ner_score,
                'best_re_score': best_re_score,
                'all_results': all_results
            }, f)

        return best_ner_model, best_re_model, best_overall_params

    def create_data_loaders(self, train_dataset_ner, train_dataset_re,
                            valid_dataset_ner,valid_dataset_re, batch_size):
        """create_data_loaders"""
        return (
            DataLoader(train_dataset_ner, batch_size=batch_size, shuffle=True),
            DataLoader(train_dataset_re, batch_size=batch_size, shuffle=True),
            DataLoader(valid_dataset_ner, batch_size=batch_size),
            DataLoader(valid_dataset_re, batch_size=batch_size)
        )

    def initialize_models(self, train_dataset_ner, train_dataset_re,
                        hidden_dim, tag_to_index, dropout_rate, device):
        """initialize_models"""
        input_dim_ner = train_dataset_ner[0][0].shape[-1]
        output_dim_ner = len(tag_to_index)

        if MODEL_TYPE == 'BiLSTM+CRF':
            ner_model = BiLSTM_CRF_NER(input_dim=input_dim_ner,
                                       hidden_dim=hidden_dim,
                                       output_dim=output_dim_ner,
                                       dropout_rate=dropout_rate).to(device)
        else:
            ner_model = NERModel(input_dim_ner, hidden_dim, output_dim_ner, dropout_rate).to(device)

        input_dim_re = train_dataset_re[0][0].shape[0]
        output_dim_re = 2
        re_model = REModel(input_dim_re, hidden_dim, output_dim_re, dropout_rate).to(device)

        return ner_model, re_model

    def perform_inference(self, best_ner_model, best_re_model,
                          test_dataset_ner_by_subgroups,
                          test_dataset_re_by_subgroups, tag_to_index, device):
        """perform_inference"""
        # NER model inference
        ner_test_metrics_by_subgroups = {}
        for subgroup_i, test_dataset_ner in test_dataset_ner_by_subgroups.items():
            print(f"subgroup_to_i: {subgroup_i}")
            if test_dataset_ner:
                test_loader_ner = DataLoader(test_dataset_ner, batch_size=256)
                ner_preds, ner_labels = inference(best_ner_model, test_loader_ner, device,
                                                  model_type=MODEL_TYPE)
                ner_test_metrics = calculate_ner_metrics(ner_preds, ner_labels, tag_to_index,
                                                        bootstrap=True)
                ner_test_metrics_by_subgroups[subgroup_i] = ner_test_metrics

        # RE model inference
        re_test_metrics_by_subgroups = {}
        if TRAIN_ON_RE:
            for subgroup_i, test_dataset_re in test_dataset_re_by_subgroups.items():
                print(f"subgroup_to_i: {subgroup_i}")
                if test_dataset_re:
                    test_loader_re = DataLoader(test_dataset_re, batch_size=256)
                    re_preds, re_labels = inference(best_re_model, test_loader_re, device,
                                                    model_type=MODEL_TYPE)
                    re_test_metrics = calculate_re_metrics(re_preds, re_labels, bootstrap=True)
                    re_test_metrics_by_subgroups[subgroup_i] = re_test_metrics

        return ner_test_metrics_by_subgroups, re_test_metrics_by_subgroups

    @staticmethod
    def _check_early_stopping(
        current_loss: float,
        best_loss: float,
    ) -> bool:
        """Check if early stopping criteria are met"""
        return current_loss < best_loss

class ModelManager:
    """Manager class for model saving and loading operations"""
    def __init__(self, save_dir: Path):
        """
        Initialize manager with save directory
        
        Args:
            save_dir: Directory path for saving models
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def check_existing_results(self,
                               ner_fname: str,
                               use_bio: bool,
                               subgroup_category: str) -> bool:
        """
        Check if results already exist for given parameters
        
        Args:
            ner_fname: Base filename for NER results
            use_bio: Whether BIO tagging scheme was used
            subgroup_category: Category for subgroup processing
            
        Returns:
            bool: True if results exist, False otherwise
        """
        # Convert ner_fname to string for string operations
        ner_fname_str = str(ner_fname)
        if not use_bio:
            ner_fname_str = ner_fname_str.replace('.txt', '_use_io.txt')
        _, subgroups_to, _ = set_subgroups_from_to_and_size_n(subgroup_category)
        for subgroup_to_i in subgroups_to:
            # First do all string replacements
            modified_fname = ner_fname_str.replace('.txt', '.pkl')
            modified_fname = modified_fname.replace('.pkl', f'_to{subgroup_to_i}.pkl')
            # Then create Path object for the final path
            pkl_path = self.save_dir / modified_fname
            if not pkl_path.exists():
                print(f"pkl_path doesn't exist! - {pkl_path}")
                return False

        print('All pkl_paths exist!')
        return True

    def save_best_model(
        self,
        model: nn.Module,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model_type: str,
        n_samples_for_train: int
    ) -> None:
        """Save best model with its parameters and metrics"""
        # Clear previous models of same type
        for file in glob.glob(os.path.join(self.save_dir, f'{model_type}_*.pt')):
            os.remove(file)

        # Create model filename
        model_name = \
            f'{MODEL_TYPE}_{model_type}_ntrainsamples{n_samples_for_train}_bs{params["batch_size"]}_lr{params["lr"]}_hd{params["hidden_dim"]}_dropout{params["dropout_rate"]}_wd{params["weight_decay"]}.pt'
        # Save model and metrics
        torch.save({
            'model': model,
            'hyperparameters': params,
            'metrics': metrics
        }, self.save_dir / model_name)

        # Save metrics separately
        with open(self.save_dir / model_name.replace('.pt', '.txt'), 'w',
                  encoding='utf-8') as f:
            f.write(str(metrics))

    def save_test_results(self, ner_fname, use_bio,
                          ner_test_metrics_by_subgroups,
                          re_test_metrics_by_subgroups):
        """save_test_results"""
        if use_bio is False:
            ner_fname = ner_fname.replace('.txt', '_use_io.txt')
        # Save NER results
        print("Test results - NER")
        for subgroup_i, ner_test_metrics in ner_test_metrics_by_subgroups.items():
            ner_fname_i = ner_fname.replace('.txt',
                                            f'_to{subgroup_i}_{MODEL_TYPE}.txt')
            with open(self.save_dir / ner_fname_i.replace('.txt', '.pkl'), 'wb') as f:
                pickle.dump(ner_test_metrics, f)
            save_results(ner_test_metrics, self.save_dir / ner_fname_i)
            save_results_excel(ner_test_metrics,
                               self.save_dir / ner_fname_i.replace('.txt', '.xlsx'))
        if TRAIN_ON_RE:
            # Save RE results
            print("Test results - RE")
            re_fname = ner_fname.replace('ner_', 're_')
            for subgroup_i, re_test_metrics in re_test_metrics_by_subgroups.items():
                re_fname_i = re_fname.replace('.txt',
                                            f'_to{subgroup_i}.txt')
                with open(self.save_dir / re_fname_i.replace('.txt', '.pkl'), 'wb') as f:
                    pickle.dump(re_test_metrics, f)
                save_results(re_test_metrics, self.save_dir / re_fname_i)
                save_results_excel(re_test_metrics,
                                self.save_dir / re_fname_i.replace('.txt', '.xlsx'))

def load_best_params_from_pickle(bert_name: str, n_samples: int) -> Dict:
    """
    Load best parameters from pickle file if it exists
    
    Args:
        bert_name: Name of BERT model
        n_samples: Number of training samples
        
    Returns:
        Dict of best parameters or None if file doesn't exist
    """
    bert_name_clean = bert_name.replace('/', '-')
    pickle_path = Path(DIR_MODELS) / f'best_params_{bert_name_clean}_samples_{n_samples}.pkl'
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            return data.get('best_params')
    return None

def save_best_params_to_pickle(bert_name: str, n_samples: int, best_params: Dict) -> None:
    """
    Save best parameters to pickle file
    
    Args:
        bert_name: Name of BERT model
        n_samples: Number of training samples
        best_params: Best hyperparameters
        best_ner_score: Best NER F1 score
        best_re_score: Best RE F1 score
        all_results: All training results
    """
    bert_name_clean = bert_name.replace('/', '-')
    pickle_path = Path(DIR_MODELS) / f'best_params_{bert_name_clean}_samples_{n_samples}.pkl'
    data = {
        'best_params': best_params
    }

    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)


def main(
    bert_name: str = 'google-bert/bert-base-uncased',
    n_samples_for_train: int = 1480,
    n_samples_for_valid: int = 50,
    use_bio: bool = True,
    saving_dataset: bool = False,
    sentence_entities: bool = False,
    dropped_entities: bool = True,
    small_dataset: bool = False,
    best_params: Dict = None,
    subgroup_category: str = None,
    subgroup_from_i: str = None,
    subgroups_to: list = None,
    valid_from_other_subgroup: bool = False,
    test_in_best_two_annotators: bool = False
) -> Tuple[Dict, Dict]:
    """
    Main function to run BERT finetuning
    
    Args:
        bert_name: Name of BERT model to use
        n_samples_for_train: Number of training samples
        use_bio: Whether to use BIO tagging scheme
        saving_dataset: Whether to save processed dataset
        sentence_entities: Whether to use sentence-level entities
        dropped_entities: Whether to use dropped entities
        small_dataset: Whether to use small dataset
        best_params: Best hyperparameters from previous run (if any)
    
    Returns:
        Tuple of (best_params, metrics)
    """
    # Initialize trainer and model manager
    trainer = BERTTrainer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = f'{DIR_BASE}/test_results/{bert_name.replace("/", "-")}_{MODEL_TYPE}'
    if DISCHARGE_SUMMARY:
        save_dir = save_dir.replace('test_results/',
                                    'test_results/discharge_summary/')
    if test_in_best_two_annotators:
        save_dir += '/test_in_two_best_annotators'
    model_manager = ModelManager(save_dir)

    # Check if we should load best_params from pickle
    if not INDEPENDENT_TUNING and best_params is None and \
        n_samples_for_train != max(ModelConfig.SAMPLE_SIZES):
        best_params = load_best_params_from_pickle(bert_name, max(ModelConfig.SAMPLE_SIZES))

    # Modify filename to include independent tuning information
    ner_fname = \
        f'ntrainsamples{n_samples_for_train}_sentenceent{sentence_entities}_dropent{dropped_entities}_{MODEL_TYPE}_ner'
    if INDEPENDENT_TUNING:
        ner_fname += '_indeptuning'
    if small_dataset:
        ner_fname += '_small'
    if subgroup_from_i:
        ner_fname += f'_subgroupfrom{subgroup_category}{subgroup_from_i}'
    ner_fname += '.txt'
    # Check if results already exist
    if model_manager.check_existing_results(ner_fname, use_bio, subgroup_category):
        print(f"Results already exist for {ner_fname}~~. Skipping training...")
        return best_params, None  # Or load and return existing results if needed

    # Load and preprocess datasets
    (train_dataset_ner, train_dataset_re, valid_dataset_ner, valid_dataset_re,
     test_dataset_ner_by_subgroups, test_dataset_re_by_subgroups, tag_to_index) = \
         preprocessing_bert_dataset(
             bert_name=bert_name,
             n_samples_for_train=n_samples_for_train,
             n_samples_for_valid=n_samples_for_valid,
             use_bio=use_bio,
             saving_dataset=saving_dataset,
             sentence_entities=sentence_entities,
             dropped_entities=dropped_entities,
             small_dataset=small_dataset,
             subgroup_category=subgroup_category,
             subgroup_from_i=subgroup_from_i,
             subgroups_to=subgroups_to,
             valid_from_other_subgroup=valid_from_other_subgroup,
             test_in_best_two_annotators=test_in_best_two_annotators,
             discharge_summary=DISCHARGE_SUMMARY
             )
    print('')

    # Train models and perform inferences
    best_ner_model, best_re_model, best_overall_params = \
        trainer.train_models_with_tuning(
            bert_name=bert_name,
            n_samples_for_train=n_samples_for_train,
            use_bio=use_bio,
            train_dataset_ner=train_dataset_ner,
            train_dataset_re=train_dataset_re,
            valid_dataset_ner=valid_dataset_ner,
            valid_dataset_re=valid_dataset_re,
            tag_to_index=tag_to_index,
            model_manager=model_manager,
            best_params=best_params  # Pass best_params to trainer
        )

    if TEST_ON_DISCHARGE:
        print("TEST_ON_DISCHARGE is True!")
        dir_dataset_tmp = \
            r'E:\NRF_CCADD\DATASET\240705\bert_finetuning\dataset\FacebookAI-xlm-roberta-large'
        fname_dataset_ner_tmp = \
            'ntrainsamples237_bio_test_ner_bysubgroups_dischargesummary.pkl'
        fname_dataset_re_tmp = \
            'ntrainsamples237_bio_test_re_bysubgroups_dischargesummary.pkl'
        with open(f"{dir_dataset_tmp}/{fname_dataset_ner_tmp}", 'rb') as f:
            test_dataset_ner_by_subgroups = pickle.load(f)
        with open(f"{dir_dataset_tmp}/{fname_dataset_re_tmp}", 'rb') as f:
            test_dataset_re_by_subgroups = pickle.load(f)
    if DISCHARGE_SUMMARY and not TEST_ON_DISCHARGE:
        print("DISCHARGE_SUMMARY is True and TEST_ON_DISCHARGE is False!")
        dir_dataset_tmp = \
            r'E:\NRF_CCADD\DATASET\240705\bert_finetuning\dataset\FacebookAI-xlm-roberta-large'
        fname_dataset_ner_tmp = 'ntrainsamples700_io_test_ner_bysubgroups.pkl'
        fname_dataset_re_tmp = 'ntrainsamples700_io_test_re_bysubgroups.pkl'
        with open(f"{dir_dataset_tmp}/{fname_dataset_ner_tmp}", 'rb') as f:
            test_dataset_ner_by_subgroups = pickle.load(f)
        with open(f"{dir_dataset_tmp}/{fname_dataset_re_tmp}", 'rb') as f:
            test_dataset_re_by_subgroups = pickle.load(f)

        from torch.utils.data import ConcatDataset
        test_dataset_ner_by_subgroups['total'] = \
            ConcatDataset([test_dataset_ner_by_subgroups['under_median_342'],
                           test_dataset_ner_by_subgroups['over_median_342']])
        test_dataset_re_by_subgroups['total'] = \
            ConcatDataset([test_dataset_re_by_subgroups['under_median_342'],
                           test_dataset_re_by_subgroups['over_median_342']])

    ner_test_metrics_by_subgroups, re_test_metrics_by_subgroups = \
        trainer.perform_inference(
            best_ner_model=best_ner_model,
            best_re_model=best_re_model if TRAIN_ON_RE else None,
            test_dataset_ner_by_subgroups=test_dataset_ner_by_subgroups,
            test_dataset_re_by_subgroups=test_dataset_re_by_subgroups,
            tag_to_index=tag_to_index,
            device=device
        )

    # Save best parameters to pickle if this is the largest sample size
    if n_samples_for_train == max(ModelConfig.SAMPLE_SIZES):
        save_best_params_to_pickle(
            bert_name,
            n_samples_for_train,
            best_overall_params
        )

    #else:
    model_manager.save_test_results(
        ner_fname, use_bio, ner_test_metrics_by_subgroups, re_test_metrics_by_subgroups
    )

    return best_overall_params, {'ner': ner_test_metrics_by_subgroups,
                                 're': re_test_metrics_by_subgroups}

def perform_experiment(best_param_default=False,
                       subgroup_category=None,
                       subgroup_from_i=None,
                       subgroups_to=None,
                       sample_sizes_for_train=None,
                       n_samples_for_valid=50,
                       valid_from_other_subgroup=False,
                       test_in_best_two_annotators=False):
    """perform_experiment"""
    for bert_name_i in ModelConfig.BERT_MODELS:
        #set best_param_default
        if best_param_default: # Initialize best_hyperparams
            best_hyperparams = BEST_PARAMS_DEFULAT
        else:
            best_hyperparams = None

        #set sample_sizes_fin
        if sample_sizes_for_train:
            sample_sizes_fin = sample_sizes_for_train
        else:
            sample_sizes_fin = ModelConfig.SAMPLE_SIZES

        #by n_samples
        print(f"Training {bert_name_i} in subgroup category - {subgroup_category}")
        for n_samples_for_train in sample_sizes_fin:
            print(f"n_samples: {n_samples_for_train}")
            # If INDEPENDENT_TUNING is True or it's the largest sample size,
            # don't pass best_hyperparams
            if INDEPENDENT_TUNING and n_samples_for_train == max(ModelConfig.SAMPLE_SIZES):
                best_hyperparams = None
            best_hyperparams, metrics = main(
                bert_name=bert_name_i,
                n_samples_for_train=n_samples_for_train,
                n_samples_for_valid=n_samples_for_valid,
                use_bio=False,
                saving_dataset=True,
                sentence_entities=False,
                dropped_entities=True,
                best_params=best_hyperparams,
                subgroup_category=subgroup_category,
                subgroup_from_i=subgroup_from_i,
                subgroups_to=subgroups_to,
                valid_from_other_subgroup=valid_from_other_subgroup,
                test_in_best_two_annotators=test_in_best_two_annotators
            )
            # Skip logging if results already existed
            if metrics is None:
                continue

            # Log results
            print(f"Completed training for {n_samples_for_train} samples")
            print(f"NER metrics: {metrics['ner']}")
            print(f"RE metrics: {metrics['re']}")
            print(f"Best hyperparameters: {best_hyperparams}")
            print("-" * 80)

def set_n_samples_settings(doc_ids):
    """set n_sample_settings"""
    n_samples_for_valid = 200 if len(doc_ids)>=400 else 50
    n_samples_for_train_max = len(doc_ids) - n_samples_for_valid
    valid_from_other_subgroup = False
    if n_samples_for_train_max < 50:
        n_samples_for_train_max = len(doc_ids)
        valid_from_other_subgroup = True
    return n_samples_for_train_max, n_samples_for_valid, valid_from_other_subgroup

def estimate_execution_time(time_per_sample=5, only_for_train_n_max=False):
    """
    Main function to estimate execution time without performing actual experiments
    """
    total_iterations = 0
    for subgroup_category, doc_infos_in_category in DOC_INFOS.items():
        print(f" - subgroup_category: {subgroup_category}")
        subgroups_from, _, size_n = \
            set_subgroups_from_to_and_size_n(subgroup_category)
        for subgroup_from_i in subgroups_from:
            if subgroup_from_i is None:
                print(f"Skipping subgroup_category({subgroup_category}) - not implemented yet")
                continue

            doc_ids = doc_infos_in_category[subgroup_from_i]
            print(f"  - subgroup_from_i: {subgroup_from_i}")
            print(f"   * document_ids length: {len(doc_ids)}")
            n_samples_for_train_max, n_samples_for_valid, valid_from_other_subgroup = \
                set_n_samples_settings(doc_ids)
            if n_samples_for_train_max>500:
                n_samples_step = 100
            else:
                n_samples_step = min(n_samples_for_train_max, 50)
            print(f"   * n_samples_for_train_max: {n_samples_for_train_max}")
            print(f"   * n_samples_for_valid: {n_samples_for_valid}")
            print(f"   * valid_from_other_subgroup: {valid_from_other_subgroup}")
            print(f"   * n_samples_step: {n_samples_step}")
            sample_sizes_for_train = list(range(n_samples_step,
                                                n_samples_for_train_max,
                                                n_samples_step))
            if n_samples_for_train_max>500:
                sample_sizes_for_train += [50]

            if len(sample_sizes_for_train)==0:
                sample_sizes_for_train = [n_samples_for_train_max]
            if only_for_train_n_max:
                sample_sizes_for_train = sample_sizes_for_train[-1:]
                if size_n:
                    sample_sizes_for_train = [size_n]
            else:
                if size_n:
                    sample_sizes_for_train = [s for s in sample_sizes_for_train
                                                if s <= size_n]

            # Calculate number of iterations for this subgroup
            n_iterations = len(sample_sizes_for_train)
            total_iterations += n_iterations
            # Calculate time for this subgroup
            subgroup_minutes = n_iterations * time_per_sample
            hours = subgroup_minutes // 60
            minutes = subgroup_minutes % 60
            time_str = f"{hours} hours {minutes} minutes" if hours > 0 else f"{minutes} minutes"
            print(f"   * Estimated execution time for this subgroup: {time_str}")
            print(f"   * Number of sample size iterations: {n_iterations}")
            print("   ----------------------------------------")

    # Calculate total time
    total_minutes = total_iterations * time_per_sample
    hours = total_minutes // 60
    minutes = total_minutes % 60
    total_time_str = f"{hours} hours {minutes} minutes" if hours > 0 else f"{minutes} minutes"
    print(f"\nTotal number of iterations: {total_iterations}")
    print(f"Total estimated execution time: {total_time_str}")
    print(f"Note: Estimation based on {time_per_sample} minutes per sample size")

INDEPENDENT_TUNING = False
def do_experiment(only_for_train_n_max=False):
    """main"""
    #estimate_execution_time
    estimate_execution_time(time_per_sample=7,
                            only_for_train_n_max=only_for_train_n_max)
    for subgroup_category, doc_infos_in_category in DOC_INFOS.items():
        print(f" - subgroup_category: {subgroup_category}")
        subgroups_from, subgroups_to, size_n = \
            set_subgroups_from_to_and_size_n(subgroup_category)
        for subgroup_from_i in subgroups_from:
            if subgroup_from_i is None:
                print(f"Skipping subgroup_category({subgroup_category}) - not implemented yet")
                continue

            doc_ids = doc_infos_in_category[subgroup_from_i]
            print(f"  - subgroup_from_i: {subgroup_from_i}")
            print(f"   * document_ids length: {len(doc_ids)}")
            n_samples_for_train_max, n_samples_for_valid, valid_from_other_subgroup = \
                set_n_samples_settings(doc_ids)
            if n_samples_for_train_max>500:
                n_samples_step = 100
            else:
                n_samples_step = min(n_samples_for_train_max, 50)
            print(f"   * n_samples_for_train_max: {n_samples_for_train_max}")
            print(f"   * n_samples_for_valid: {n_samples_for_valid}")
            print(f"   * valid_from_other_subgroup: {valid_from_other_subgroup}")
            print(f"   * n_samples_step: {n_samples_step}")
            sample_sizes_for_train = list(range(n_samples_step,
                                                n_samples_for_train_max,
                                                n_samples_step))
            if n_samples_for_train_max>500:
                sample_sizes_for_train += [50]
            if len(sample_sizes_for_train)==0:
                sample_sizes_for_train = [n_samples_for_train_max]
            if only_for_train_n_max:
                sample_sizes_for_train = sample_sizes_for_train[-1:]
                if size_n:
                    sample_sizes_for_train = [size_n]
            else:
                if size_n:
                    sample_sizes_for_train = [s for s in sample_sizes_for_train
                                                if s <= size_n]

            #perform_experiment
            if subgroup_from_i and subgroups_to and n_samples_for_valid:
                perform_experiment(best_param_default=True,
                                   subgroup_category=subgroup_category,
                                   subgroup_from_i=subgroup_from_i,
                                   subgroups_to=subgroups_to,
                                   sample_sizes_for_train=sample_sizes_for_train,
                                   n_samples_for_valid=n_samples_for_valid,
                                   valid_from_other_subgroup=valid_from_other_subgroup,
                                   test_in_best_two_annotators=TEST_IN_BEST_TWO_ANNOTATORS)
            else:
                print(f"Skipped training for subgroup_category({subgroup_category}),  subgroup_from_i({subgroup_from_i}), subgroups_to({subgroups_to}), sample_sizes_for_train({sample_sizes_for_train})!")

def train_for_total():
    """train_for_total"""
    print("Train for total dataset!")
    print(f"Data type :{'mimic discharge summary' if DISCHARGE_SUMMARY else 'in-house'}")
    perform_experiment(best_param_default=True)

DISCHARGE_SUMMARY = False
TEST_ON_DISCHARGE = False
if __name__ == "__main__":
    # SNUH to SNUH / diverse sample n from
    samplenfroms = [50, 100, 200, 300, 500, 900, 1400]
    for samplenfrom in samplenfroms:
        main(bert_name='FacebookAI/xlm-roberta-large',
             n_samples_for_train=samplenfrom,
             n_samples_for_valid=211,
             use_bio=False,
             saving_dataset=True)

    # MIMIC to MIMIC (TRAINING -> TESTING)
    # main(bert_name='FacebookAI/xlm-roberta-large',
    #      n_samples_for_train=237,
    #      n_samples_for_valid=60,
    #      use_bio=True,
    #      saving_dataset=True)

    # SNUH to MIMIC (TRAINING -> TESTING)
    # main(bert_name='FacebookAI/xlm-roberta-large',
    #      use_bio=False,
    #      saving_dataset=False)
