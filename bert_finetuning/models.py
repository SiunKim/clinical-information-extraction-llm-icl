"""
Neural Network Models for Information Extraction Tasks

This module provides PyTorch implementations of neural network models 
for Named Entity Recognition (NER) and Relation Extraction (RE) tasks.
Both models use a similar architecture consisting of fully connected
layers with batch normalization and dropout for regularization.

Models:
    - NERModel: A neural network for Named Entity Recognition that classifies tokens into predefined
                entity categories (e.g., person, organization, location).
    - REModel: A neural network for Relation Extraction that identifies relationships between
               entities in text (e.g., works_for, located_in, founded_by).
    - BiLSTM_CRF_NER: A BiLSTM-CRF model for Named Entity Recognition using transformers CRF implementation.

Architecture Details:
    Both basic models feature:
    - Three fully connected layers with decreasing dimensions
    - Batch normalization after first two layers
    - Dropout for regularization
    - ReLU activation functions
"""
import torch
from torch import nn

class NERModel(nn.Module):
    """
    Neural Network model for Named Entity Recognition (NER) task.
    This model consists of three fully connected layers with batch normalization and dropout
    for regularization.
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden layer
        output_dim (int): Dimension of output (number of NER classes)
        dropout_rate (float): Dropout rate for regularization (default: 0.5)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(NERModel, self).__init__()
        # First fully connected layer with batch normalization and dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second fully connected layer with reduced dimensions
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the NER model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        self.eval()
        # First layer: FC -> BatchNorm -> ReLU -> Dropout
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        # Second layer: FC -> BatchNorm -> ReLU -> Dropout
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        # Output layer: FC only
        x = self.fc3(x)
        return x


class REModel(nn.Module):
    """
    Neural Network model for Relation Extraction (RE) task.
    This model has the same architecture as the NER model but is used for
    identifying relationships between entities.
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden layer
        output_dim (int): Dimension of output (number of relation classes)
        dropout_rate (float): Dropout rate for regularization (default: 0.5)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(REModel, self).__init__()
        # First fully connected layer with batch normalization and dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second fully connected layer with reduced dimensions
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the RE model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        self.eval()
        # First layer: FC -> BatchNorm -> ReLU -> Dropout
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        # Second layer: FC -> BatchNorm -> ReLU -> Dropout
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        # Output layer: FC only
        x = self.fc3(x)
        return x

class BiLSTM_CRF_NER(nn.Module):
    """
    BiLSTM-CRF model for Named Entity Recognition (NER) task.
    This model uses pre-computed embeddings as input, a Bidirectional LSTM
    to capture contextual information, and a Conditional Random Field (CRF)
    for sequence labeling.
    
    Args:
        input_dim (int): Dimension of input features (pre-computed embeddings)
        hidden_dim (int): Dimension of LSTM hidden state
        output_dim (int): Dimension of output (number of NER classes)
        num_layers (int): Number of LSTM layers (default: 1)
        dropout_rate (float): Dropout rate for regularization (default: 0.5)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_rate=0.5):
        super(BiLSTM_CRF_NER, self).__init__()
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Linear layer to map from LSTM output to emission scores
        self.hidden2tag = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Custom CRF layer for sequence labeling
        self.num_tags = output_dim
        # Transition matrix
        self.transitions = nn.Parameter(torch.randn(output_dim, output_dim))
        # Initialize special transitions from START and to END states
        self.start_transitions = nn.Parameter(torch.randn(output_dim))
        self.end_transitions = nn.Parameter(torch.randn(output_dim))
        
    def _get_lstm_features(self, x):
        """
        Get LSTM features from input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
                             with pre-computed embeddings
        
        Returns:
            torch.Tensor: LSTM emissions of shape (batch_size, seq_len, output_dim)
        """
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)  # (batch_size, seq_len, output_dim)
        
        return emissions
        
    def forward(self, x, mask=None):
        """
        Forward pass without computing loss (for inference).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
                             with pre-computed embeddings
            mask (torch.ByteTensor, optional): Mask tensor of shape (batch_size, seq_len)
                                            with 1s for valid positions and 0s for padding
        
        Returns:
            list: List of predicted tag sequences
        """
        emissions = self._get_lstm_features(x)
        
        # Convert mask to valid format if provided
        if mask is not None:
            mask = mask.bool()
            
        # Decode to get the best paths
        return self._viterbi_decode(emissions, mask)
        
    def _viterbi_decode(self, emissions, mask=None):
        """
        Uses Viterbi algorithm to find the most likely tag sequence.
        
        Args:
            emissions (torch.Tensor): Emission scores tensor of shape (batch_size, seq_len, num_tags)
            mask (torch.ByteTensor, optional): Mask tensor of shape (batch_size, seq_len)
                                            with 1s for valid positions and 0s for padding
        
        Returns:
            list: List of best tag sequences for each batch
        """
        batch_size, seq_length, num_tags = emissions.shape
        
        # In case the mask is not provided, we use a tensor of ones
        if mask is None:
            mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=emissions.device)
            
        best_tags_list = []
        
        # For each batch
        for i in range(batch_size):
            # Initialize score with start_transitions
            score = self.start_transitions + emissions[i, 0]
            history = []
            
            # For each step except the first one
            for j in range(1, seq_length):
                # If the position is padding, just use previous score
                if not mask[i, j]:
                    history.append(torch.zeros(num_tags, dtype=torch.long, device=emissions.device))
                    continue
                    
                # Broadcast score for every possible next tag
                broadcast_score = score.unsqueeze(1)
                
                # Calculate score for next step (emission + transition)
                next_score = broadcast_score + self.transitions
                
                # Find the indices of the maximum score and the maximum score itself
                best_score, best_tags = next_score.max(0)
                
                # Add emission score for current position
                score = best_score + emissions[i, j]
                
                # Save the best tags
                history.append(best_tags)
                
            # Add end_transitions score
            score += self.end_transitions
            
            # Find the best final score and the corresponding tag
            _, best_last_tag = score.max(0)
            best_last_tag = best_last_tag.item()
            
            # Follow the back pointers to build the best path
            best_tags = [best_last_tag]
            for hist in reversed(history):
                if len(hist) > 0:  # Skip padding positions
                    best_last_tag = hist[best_last_tag].item()
                    best_tags.append(best_last_tag)
                    
            # Reverse the order to get the correct tag sequence
            best_tags.reverse()
            best_tags_list.append(best_tags)
            
        return best_tags_list
    
    def loss(self, x, tags, mask=None):
        """
        Compute the CRF loss.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
                            with pre-computed embeddings
            tags (torch.LongTensor): Ground truth tags of shape (batch_size, seq_len)
            mask (torch.ByteTensor, optional): Mask tensor of shape (batch_size, seq_len)
                                            with 1s for valid positions and 0s for padding
        
        Returns:
            torch.Tensor: Negative log likelihood loss
        """
        emissions = self._get_lstm_features(x)
        
        # 수정: 차원 확인 및 재구성
        if len(tags.shape) == 1:
            tags = tags.unsqueeze(0)  # [seq_len] -> [1, seq_len]
            emissions = emissions.unsqueeze(0) if len(emissions.shape) == 2 else emissions
        
        # Convert mask to valid format if provided
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 1:
                mask = mask.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        else:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
            
        # Calculate negative log likelihood (loss)
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        # Return negative log likelihood as loss
        return -log_likelihood.mean()

    def _compute_log_likelihood(self, emissions, tags, mask):
        """
        Compute the log-likelihood of the given tag sequence under the CRF model.
        
        Args:
            emissions (torch.Tensor): Emission scores tensor of shape (batch_size, seq_len, num_tags)
            tags (torch.LongTensor): Ground truth tags of shape (batch_size, seq_len)
            mask (torch.ByteTensor): Mask tensor of shape (batch_size, seq_len)
                                    with 1s for valid positions and 0s for padding
        
        Returns:
            torch.Tensor: Log-likelihood for each sequence, shape (batch_size,)
        """
        # 수정: tags 차원 확인 및 재구성
        if len(tags.shape) == 1:
            # 1차원 tags를 2차원으로 변환 (배치 크기가 1인 경우)
            tags = tags.unsqueeze(0)  # [seq_len] -> [1, seq_len]
            
        batch_size, seq_length = tags.shape
        log_likelihood = torch.zeros(batch_size, device=emissions.device)
        
        # For each batch
        for i in range(batch_size):
            # 중요: in-place 연산(+=) 대신 일반 연산(+)을 사용
            # Score for the target path
            score = self.start_transitions[tags[i, 0]]
            score = score + emissions[i, 0, tags[i, 0]]
            
            # Add transition and emission scores for the rest of the sequence
            for j in range(1, seq_length):
                # Skip if mask is 0 (padding)
                if not mask[i, j]:
                    continue
                
                # Add transition score from previous tag to current tag
                trans_score = self.transitions[tags[i, j-1], tags[i, j]]
                emit_score = emissions[i, j, tags[i, j]]
                score = score + trans_score + emit_score
                
            # Add transition to END tag
            last_tag_idx = mask[i].sum().long() - 1
            score = score + self.end_transitions[tags[i, last_tag_idx]]
            
            # Compute the total score for all possible paths
            forward_score = self._forward_algorithm(
                emissions[i, :mask[i].sum().long()], 
                mask[i, :mask[i].sum().long()]
            )
            
            # The log-likelihood is the score of the target path minus the score of all paths
            log_likelihood[i] = score - forward_score
            
        return log_likelihood
    
    def _forward_algorithm(self, emissions, mask):
        """
        Calculate the partition function Z using the forward algorithm.
        
        Args:
            emissions (torch.Tensor): Emission scores tensor of shape (seq_len, num_tags)
            mask (torch.ByteTensor): Mask tensor of shape (seq_len)
        
        Returns:
            torch.Tensor: Partition function Z (scalar)
        """
        seq_length = emissions.shape[0]
        
        # Initialize forward variables with start_transitions
        forward_var = self.start_transitions + emissions[0]
        
        # Iterate through the rest of the sequence
        for i in range(1, seq_length):
            # Skip if position is padding
            if not mask[i]:
                continue
                
            # Create a new forward variable as a combination of all possible transitions
            alphas_t = []
            
            for next_tag in range(self.num_tags):
                # The emission scores for all possible previous tags leading to next_tag
                emit_score = emissions[i, next_tag].unsqueeze(0).expand(self.num_tags)
                # The transition scores from all possible previous tags
                trans_score = self.transitions[:, next_tag]
                # The score of being in each possible tag at position i-1 and then transitioning to next_tag at i
                next_tag_var = forward_var + trans_score + emit_score
                # Use log-sum-exp to add these scores
                alphas_t.append(torch.logsumexp(next_tag_var, dim=0))
            
            # Update forward variable
            forward_var = torch.stack(alphas_t)
            
        # Add end_transitions to get the final score
        terminal_var = forward_var + self.end_transitions
        # The partition function is the log-sum-exp of all possible tag sequences
        alpha = torch.logsumexp(terminal_var, dim=0)
        
        return alpha
