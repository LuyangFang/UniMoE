import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import argparse
import random
import hashlib
import time


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with learnable routing and load balancing
    """
    def __init__(self, input_dim, hidden_dim, num_experts, num_tasks, dropout_rate=0.1, task_embedding_dim=64):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, input_dim),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_experts)
        ])
        
        # Task embeddings for task-aware routing
        self.task_embeddings = nn.Embedding(num_tasks, task_embedding_dim)
        
        # Gating network (now takes concatenated input + task embedding)
        self.gate = nn.Sequential(
            nn.Linear(input_dim + task_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, task_id):
        # Get task embedding
        task_emb = self.task_embeddings(task_id)  
        gate_input = torch.cat([x, task_emb], dim=-1)  
        
        # Get gating weights
        gate_weights = self.gate(gate_input)  
        
        # Calculate load balancing loss
        avg_gate_weights = gate_weights.mean(dim=0) 
        uniform_target = 1.0 / self.num_experts
        load_balance_loss = ((avg_gate_weights - uniform_target) ** 2).mean()
        
        # Get expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) 

        gate_weights_expanded = gate_weights.unsqueeze(-1)  
        output = torch.sum(gate_weights_expanded * expert_outputs, dim=1) 
        
        return output, gate_weights, load_balance_loss


class MultiTaskBERTWithMoE(nn.Module):
    """
    Multi-task BERT model with Mixture of Experts
    """
    def __init__(self, bert_model_name, task_configs, num_experts=4, moe_hidden_dim=512, dropout_rate=0.1, task_embedding_dim=64):
        super(MultiTaskBERTWithMoE, self).__init__()
        
        # Shared BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size
        
        # Task configurations
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        self.num_tasks = len(self.task_names)
        self.task_to_id = {task_name: i for i, task_name in enumerate(self.task_names)}
        
        # MoE layers with task-aware routing
        self.moe_layers = nn.ModuleList([
            MixtureOfExperts(bert_dim, moe_hidden_dim, num_experts, self.num_tasks, dropout_rate, task_embedding_dim)
            for _ in range(2)  # Using 2 MoE layers
        ])
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task_name: nn.Sequential(
                nn.Linear(bert_dim, bert_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(bert_dim // 2, config['num_labels'])
            )
            for task_name, config in task_configs.items()
        })
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(bert_dim)
        
    def forward(self, input_ids, attention_mask, token_type_ids, task_name):
        # Get task ID
        task_id = self.task_to_id[task_name]
        task_id_tensor = torch.tensor([task_id] * input_ids.size(0), device=input_ids.device)
        
        # Shared BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = bert_outputs.pooler_output
        
        # Pass through MoE layers with residual connections
        moe_output = pooled_output
        gate_weights_list = []
        load_balance_losses = []
        
        for moe_layer in self.moe_layers:
            moe_result, gate_weights, lb_loss = moe_layer(moe_output, task_id_tensor)
            moe_output = self.layer_norm(moe_output + moe_result)  # Residual connection
            gate_weights_list.append(gate_weights)
            load_balance_losses.append(lb_loss)
        
        # Task-specific head
        logits = self.task_heads[task_name](moe_output)
        
        # Average load balance loss across layers
        total_lb_loss = torch.stack(load_balance_losses).mean()
        
        return logits, gate_weights_list, total_lb_loss


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task learning with optional teacher predictions
    """
    def __init__(self, data_dict, tokenizer, max_length=100):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Combine all task data
        for task_name, task_data in data_dict.items():
            teacher_probs = task_data.get('teacher_probs', None)
            
            for idx in range(len(task_data['sentences'])):
                item = {
                    'task_name': task_name,
                    'sentence': task_data['sentences'][idx],
                    'labels': task_data['labels'][idx]
                }
                
                # Add teacher probabilities if available
                if teacher_probs is not None and idx < len(teacher_probs):
                    item['teacher_probs'] = teacher_probs[idx]
                else:
                    item['teacher_probs'] = None
                    
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['sentence'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'task_name': item['task_name'],
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'labels': torch.tensor(item['labels'], dtype=torch.float)
        }
        
        # Add teacher probabilities if available
        if item['teacher_probs'] is not None:
            result['teacher_probs'] = torch.tensor(item['teacher_probs'], dtype=torch.float)
        
        return result


def collate_fn(batch):
    """
    Custom collate function for multi-task batches with teacher probabilities
    """
    # Group by task
    task_groups = {}
    for item in batch:
        task_name = item['task_name']
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append(item)
    
    # Create mini-batches for each task
    task_batches = {}
    for task_name, items in task_groups.items():
        batch_data = {
            'input_ids': torch.stack([item['input_ids'] for item in items]),
            'attention_mask': torch.stack([item['attention_mask'] for item in items]),
            'token_type_ids': torch.stack([item['token_type_ids'] for item in items]),
            'labels': torch.stack([item['labels'] for item in items])
        }
        
        # Add teacher probabilities if all items have them
        if all('teacher_probs' in item for item in items):
            batch_data['teacher_probs'] = torch.stack([item['teacher_probs'] for item in items])
        
        task_batches[task_name] = batch_data
    
    return task_batches


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=3, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        """Saves model when validation loss decreases"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class MultiTaskTrainer:
    """
    Trainer for multi-task learning with MoE and Knowledge Distillation
    """
    def __init__(self, model, device, task_configs, load_balance_weight=0.01, 
                 kd_weight=0.5, kd_temperature=3.0, verbose=True):
        self.model = model
        self.device = device
        self.task_configs = task_configs
        self.load_balance_weight = load_balance_weight
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.verbose = verbose  # Control progress bar display
        self.model.to(device)
        
    def compute_kd_loss(self, student_logits, teacher_probs):
        """
        Compute knowledge distillation loss for multi-label classification
        """

        student_probs = torch.sigmoid(student_logits / self.kd_temperature)
        teacher_probs = torch.clamp(teacher_probs, 1e-7, 1 - 1e-7)

        kd_loss = F.binary_cross_entropy(
            student_probs,
            teacher_probs,
            reduction='mean'
        )
        kd_loss = kd_loss * (self.kd_temperature ** 2)
        
        return kd_loss
        
    def train_epoch(self, train_loader, optimizer, scheduler, loss_fn):
        self.model.train()
        total_loss = 0
        total_lb_loss = 0
        total_kd_loss = 0
        task_losses = {task: 0 for task in self.task_configs}
        task_kd_losses = {task: 0 for task in self.task_configs}
        task_counts = {task: 0 for task in self.task_configs}
        kd_counts = {task: 0 for task in self.task_configs}
        
        # Create progress bar with less frequent updates or disable if not verbose
        if self.verbose:
            progress_bar = tqdm(train_loader, desc="Training", mininterval=2.0, maxinterval=10.0)
            update_frequency = max(1, len(train_loader) // 20)  
        else:
            progress_bar = train_loader
            update_frequency = max(1, len(train_loader) // 5) 
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            batch_loss = 0
            batch_lb_loss = 0
            batch_kd_loss = 0
            
            # Process each task in the batch
            for task_name, task_batch in batch.items():
                # Move to device
                input_ids = task_batch['input_ids'].to(self.device)
                attention_mask = task_batch['attention_mask'].to(self.device)
                token_type_ids = task_batch['token_type_ids'].to(self.device)
                labels = task_batch['labels'].to(self.device)
                
                # Forward pass
                logits, _, lb_loss = self.model(input_ids, attention_mask, token_type_ids, task_name)
                
                # Calculate task loss (BCE with logits for multi-label)
                num_labels = self.task_configs[task_name]['num_labels']
                task_loss = loss_fn(logits.view(-1, num_labels), labels.view(-1, num_labels))
                
                # Calculate KD loss if teacher predictions are available
                kd_loss = 0
                if 'teacher_probs' in task_batch:
                    teacher_probs = task_batch['teacher_probs'].to(self.device)
                    # Ensure teacher probs have same shape as logits
                    if teacher_probs.shape != logits.shape:
                        print(f"Warning: Shape mismatch - logits: {logits.shape}, teacher: {teacher_probs.shape}")
                    else:
                        kd_loss = self.compute_kd_loss(logits, teacher_probs)
                        batch_kd_loss +=  0.05 * kd_loss.item()
                        task_kd_losses[task_name] += kd_loss.item()
                        kd_counts[task_name] += 1
                
                # Combined loss
                combined_loss = task_loss + self.load_balance_weight * lb_loss
                if kd_loss != 0:
                    combined_loss += self.kd_weight * kd_loss * 0.05
                
                batch_loss += combined_loss
                batch_lb_loss += self.load_balance_weight * lb_loss.item()
                task_losses[task_name] += task_loss.item()
                task_counts[task_name] += 1
            
            # Backward pass
            if batch_loss.requires_grad:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            total_loss += batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss
            total_lb_loss += batch_lb_loss
            total_kd_loss += batch_kd_loss
            
            # Update progress bar only at specified intervals
            if batch_idx % update_frequency == 0 or batch_idx == len(train_loader) - 1:
                if self.verbose:
                    progress_dict = {
                        'loss': batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss,
                        'lb_loss': batch_lb_loss
                    }
                    if batch_kd_loss > 0:
                        progress_dict['kd_loss'] = batch_kd_loss
                    progress_bar.set_postfix(progress_dict)
                else:
                    # Print status without progress bar
                    print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {batch_loss:.4f}, LB: {batch_lb_loss:.6f}, KD: {batch_kd_loss:.4f}")
        
        # Calculate average losses
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_lb_loss = total_lb_loss / num_batches if num_batches > 0 else 0
        avg_kd_loss = total_kd_loss / num_batches if num_batches > 0 else 0
        
        avg_task_losses = {
            task: task_losses[task] / task_counts[task] if task_counts[task] > 0 else 0
            for task in self.task_configs
        }
        
        # Print KD usage statistics
        total_kd_samples = sum(kd_counts.values())
        total_samples = sum(task_counts.values())
        if total_samples > 0:
            kd_coverage = total_kd_samples / total_samples * 100
            print(f"\nKD Coverage: {kd_coverage:.1f}% of samples had teacher predictions")
        
        return avg_loss, avg_task_losses, avg_lb_loss, avg_kd_loss
    
    def evaluate(self, eval_loader, loss_fn):
        """
        Evaluate the model on validation/test data
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        task_metrics = {task: {'preds': [], 'labels': []} for task in self.task_configs}
        
        # Create progress bar with less frequent updates for evaluation or disable if not verbose
        with torch.no_grad():
            if self.verbose:
                progress_bar = tqdm(eval_loader, desc="Evaluating", mininterval=2.0, maxinterval=10.0)
                eval_update_frequency = max(1, len(eval_loader) // 10) 
            else:
                progress_bar = eval_loader
                eval_update_frequency = max(1, len(eval_loader) // 3)  
            
            for batch_idx, batch in enumerate(progress_bar):
                for task_name, task_batch in batch.items():
                    # Move to device
                    input_ids = task_batch['input_ids'].to(self.device)
                    attention_mask = task_batch['attention_mask'].to(self.device)
                    token_type_ids = task_batch['token_type_ids'].to(self.device)
                    labels = task_batch['labels'].to(self.device)
                    
                    # Forward pass
                    logits, _, _ = self.model(input_ids, attention_mask, token_type_ids, task_name)
                    
                    # Calculate loss
                    num_labels = self.task_configs[task_name]['num_labels']
                    loss = loss_fn(logits.view(-1, num_labels), labels.view(-1, num_labels))
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Store predictions (using sigmoid for multi-label)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    task_metrics[task_name]['preds'].append(preds.cpu().numpy())
                    task_metrics[task_name]['labels'].append(labels.cpu().numpy())
                
                # Update progress bar only at specified intervals
                if batch_idx % eval_update_frequency == 0 or batch_idx == len(eval_loader) - 1:
                    if self.verbose:
                        progress_bar.set_postfix({'avg_loss': total_loss / max(num_batches, 1)})
                    else:
                        print(f"  Eval Batch {batch_idx+1}/{len(eval_loader)} - Avg Loss: {total_loss / max(num_batches, 1):.4f}")
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Calculate metrics for each task
        task_scores = {}
        for task_name in self.task_configs:
            if len(task_metrics[task_name]['preds']) > 0:
                # Concatenate all predictions and labels
                preds = np.vstack(task_metrics[task_name]['preds'])
                labels = np.vstack(task_metrics[task_name]['labels'])
                
                # Ensure correct data types
                preds = preds.astype(np.int32)
                labels = labels.astype(np.int32)
                
                # Calculate kappa scores for each label
                per_label_kappa = []
                per_label_acc = []
                for i in range(preds.shape[1]):
                    try:
                        kappa = cohen_kappa_score(labels[:, i], preds[:, i])
                        per_label_kappa.append(kappa)
                    except:
                        per_label_kappa.append(0.0)  #
                    
                    # Calculate per-label accuracy
                    label_acc = accuracy_score(labels[:, i], preds[:, i])
                    per_label_acc.append(label_acc)
                
                # Calculate average kappa and accuracy
                avg_kappa = np.mean(per_label_kappa)
                avg_per_label_acc = np.mean(per_label_acc)
                
                # Calculate metrics
                task_scores[task_name] = {
                    'micro_f1': f1_score(labels, preds, average='micro', zero_division=0),
                    'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
                    'exact_match': np.mean(np.all(preds == labels, axis=1)),
                    'avg_per_label_acc': avg_per_label_acc,
                    'avg_kappa': avg_kappa,
                    'per_label_kappa': per_label_kappa,
                    'per_label_acc': per_label_acc,
                    'num_samples': len(labels)
                }
                
                # Add per-label F1 scores
                label_names = self.task_configs[task_name].get('label_names', 
                               [f'Label_{i}' for i in range(preds.shape[1])])
                per_label_f1 = {}
                for i, label_name in enumerate(label_names[:preds.shape[1]]):
                    per_label_f1[label_name] = f1_score(
                        labels[:, i], 
                        preds[:, i], 
                        average='binary', 
                        zero_division=0
                    )
                task_scores[task_name]['per_label_f1'] = per_label_f1
            else:
                task_scores[task_name] = {
                    'micro_f1': 0, 
                    'macro_f1': 0, 
                    'exact_match': 0,
                    'avg_per_label_acc': 0,
                    'avg_kappa': 0,
                    'per_label_kappa': [],
                    'per_label_acc': [],
                    'num_samples': 0,
                    'per_label_f1': {}
                }
        
        return avg_loss, task_scores


def load_teacher_predictions(teacher_pred_dir, task_name, split='train'):
    """
    Load teacher model predictions from CSV files
    """
    csv_path = os.path.join(teacher_pred_dir, task_name, f'{split}_probs.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: Teacher predictions not found at {csv_path}")
        return None
    
    teacher_df = pd.read_csv(csv_path)
    prob_cols = [col for col in teacher_df.columns if col.startswith('p(') and col.endswith('=yes)')]
    teacher_probs = teacher_df[prob_cols].values
    
    print(f"  Loaded {len(teacher_probs)} teacher predictions with {len(prob_cols)} labels")
    
    return teacher_probs


def load_dataset_with_teacher_probs(file_path, teacher_pred_dir, dataset_name):
    """
    Load dataset and merge with teacher predictions
    """
    # Load the training data
    df = pd.read_csv(file_path)

    df.rename(columns={df.columns[0]: "Sentence"}, inplace=True)
    if df.columns[-1] == "Group":
        df.drop(columns=["Group"], inplace=True)
    
    # Extract label columns
    label_cols = [col for col in df.columns if col != "Sentence"]
    
    print(f"  Found {len(label_cols)} labels: {label_cols}")

    teacher_probs = None
    if teacher_pred_dir and os.path.exists(teacher_pred_dir):
        teacher_probs = load_teacher_predictions(teacher_pred_dir, dataset_name, 'train')

    sentences = df['Sentence'].tolist()
    labels = df[label_cols].values.tolist()
    
    return {
        'sentences': sentences,
        'labels': labels,
        'teacher_probs': teacher_probs,
        'num_labels': len(label_cols),
        'label_names': label_cols
    }


def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_config_hash(config):
    """Generate a unique hash for the configuration"""
    # Create a string representation of the configuration
    config_str = f"experts_{config['num_experts']}_lb_{config['load_balance_weight']}_kd_{config['kd_weight']}_temp_{config['kd_temperature']}_lr_{config['learning_rate']}_seed_{config['seed']}"
    # Generate hash
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return config_hash


def create_unique_output_dir(config, output_dir):
    """Create a unique output directory that won't be overwritten"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = generate_config_hash(config)
    
    # Create descriptive directory name
    dir_name = f"exp_{config['num_experts']}_lb_{config['load_balance_weight']}_kd_{config['kd_weight']}_seed_{config['seed']}_{timestamp}_{config_hash}"
    config_dir = os.path.join(output_dir, dir_name)

    counter = 1
    original_dir = config_dir
    while os.path.exists(config_dir):
        config_dir = f"{original_dir}_v{counter}"
        counter += 1
    
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def train_model_with_config(config, train_loader, val_loader, task_configs, device, output_dir):
    """
    Train a model with specific hyperparameter configuration
    """
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"\n{'='*60}")
    print(f"TRAINING STARTED")
    print(f"{'='*60}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTraining model with configuration:")
    print(f"  - Number of experts: {config['num_experts']}")
    print(f"  - Load balance weight: {config['load_balance_weight']}")
    print(f"  - KD weight: {config['kd_weight']}")
    print(f"  - KD temperature: {config['kd_temperature']}")
    print(f"  - Seed: {config['seed']}")
    print(f"  - Verbose: {config.get('verbose', True)}")
    
    # Set random seeds for reproducibility
    set_random_seeds(config['seed'])
    
    # Create unique output directory for this configuration
    config_dir = create_unique_output_dir(config, output_dir)
    print(f"  - Results will be saved to: {config_dir}")
    
    # Save configuration to file immediately
    config_with_times = config.copy()
    config_with_times['start_time'] = start_datetime.isoformat()
    with open(os.path.join(config_dir, 'config.json'), 'w') as f:
        json.dump(config_with_times, f, indent=2)
    
    # Initialize model
    model = MultiTaskBERTWithMoE(
        bert_model_name='bert-base-uncased',
        task_configs=task_configs,
        num_experts=config['num_experts'],
        moe_hidden_dim=512,
        task_embedding_dim=64
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config['max_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Initialize trainer
    trainer = MultiTaskTrainer(
        model, 
        device, 
        task_configs, 
        load_balance_weight=config['load_balance_weight'],
        kd_weight=config['kd_weight'],
        kd_temperature=config['kd_temperature'],
        verbose=config.get('verbose', True)  # Control progress bar verbosity
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
    temp_model_path = os.path.join(config_dir, 'temp_model.pt')
    
    # Training loop
    train_losses = []
    val_losses = []
    train_lb_losses = []
    train_kd_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    val_scores_history = []
    
    for epoch in range(config['max_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['max_epochs']}")
        
        # Train
        train_loss, train_task_losses, train_lb_loss, train_kd_loss = trainer.train_epoch(
            train_loader, optimizer, scheduler, loss_fn
        )
        train_losses.append(train_loss)
        train_lb_losses.append(train_lb_loss)
        train_kd_losses.append(train_kd_loss)
        
        # Evaluate
        val_loss, val_task_scores = trainer.evaluate(val_loader, loss_fn)
        val_losses.append(val_loss)
        val_scores_history.append(val_task_scores)
        
        # Calculate metrics
        avg_val_f1 = np.mean([scores['micro_f1'] for scores in val_task_scores.values()])
        avg_val_kappa = np.mean([scores['avg_kappa'] for scores in val_task_scores.values()])
        avg_val_acc = np.mean([scores['avg_per_label_acc'] for scores in val_task_scores.values()])
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Avg Val - F1: {avg_val_f1:.4f}, Kappa: {avg_val_kappa:.4f}, Acc: {avg_val_acc:.4f}")
        print(f"Component losses - LB: {train_lb_loss:.6f}, KD: {train_kd_loss:.6f}")
        
        # Print detailed task scores
        print("Task-specific validation scores:")
        for task, scores in val_task_scores.items():
            print(f"  {task}: F1_micro={scores['micro_f1']:.4f}, F1_macro={scores['macro_f1']:.4f}, "
                  f"EM={scores['exact_match']:.4f}, Acc={scores['avg_per_label_acc']:.4f}, Kappa={scores['avg_kappa']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'task_configs': task_configs,
                'config': config,
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                'val_task_scores': val_task_scores
            }, os.path.join(config_dir, 'best_model.pt'))
            print("Best model saved!")
        
        early_stopping(val_loss, model, temp_model_path)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Clean up temporary model file
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
    
    # Record end time and calculate duration
    end_time = time.time()
    end_datetime = datetime.now()
    training_duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training duration: {training_duration/3600:.2f} hours ({training_duration/60:.1f} minutes)")
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch+1})")
    
    # Calculate final metrics
    final_val_scores = val_scores_history[best_epoch] if best_epoch < len(val_scores_history) else val_scores_history[-1]
    final_avg_f1 = np.mean([scores['micro_f1'] for scores in final_val_scores.values()])
    final_avg_kappa = np.mean([scores['avg_kappa'] for scores in final_val_scores.values()])
    final_avg_acc = np.mean([scores['avg_per_label_acc'] for scores in final_val_scores.values()])
    
    print(f"Final average metrics:")
    print(f"  - Micro F1: {final_avg_f1:.4f}")
    print(f"  - Kappa: {final_avg_kappa:.4f}")
    print(f"  - Per-label Accuracy: {final_avg_acc:.4f}")
    
    # Save training history and results
    results = {
        'config': config,
        'timing': {
            'start_time': start_datetime.isoformat(),
            'end_time': end_datetime.isoformat(),
            'duration_seconds': training_duration,
            'duration_minutes': training_duration / 60,
            'duration_hours': training_duration / 3600
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'lb_losses': train_lb_losses,
            'kd_losses': train_kd_losses
        },
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_val_scores': final_val_scores,
        'final_metrics': {
            'avg_micro_f1': final_avg_f1,
            'avg_kappa': final_avg_kappa,
            'avg_per_label_acc': final_avg_acc
        },
        'epochs_trained': len(train_losses)
    }
    
    # Save results to JSON
    with open(os.path.join(config_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create training curves plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', marker='o')
    axes[0, 0].plot(val_losses, label='Val Loss', marker='s')
    axes[0, 0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Model')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Component losses
    axes[0, 1].plot(train_lb_losses, label='Load Balance Loss', color='orange', marker='d')
    axes[0, 1].plot(train_kd_losses, label='KD Loss', color='green', marker='^')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 scores over time
    f1_scores = [[scores[task]['micro_f1'] for task in task_configs.keys()] for scores in val_scores_history]
    avg_f1_scores = [np.mean(scores) for scores in f1_scores]
    axes[1, 0].plot(avg_f1_scores, label='Avg F1', color='blue', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Average F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Kappa scores over time
    kappa_scores = [[scores[task]['avg_kappa'] for task in task_configs.keys()] for scores in val_scores_history]
    avg_kappa_scores = [np.mean(scores) for scores in kappa_scores]
    axes[1, 1].plot(avg_kappa_scores, label='Avg Kappa', color='purple', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Kappa Score')
    axes[1, 1].set_title('Average Kappa Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to: {config_dir}")
    print(f"  - best_model.pt: Best model checkpoint")
    print(f"  - training_results.json: Complete training results")
    print(f"  - training_curves.png: Training visualization")
    print(f"  - config.json: Experiment configuration")
    
    return results


def main():
    # Record overall start time
    overall_start_time = time.time()
    overall_start_datetime = datetime.now()
    
    parser = argparse.ArgumentParser(description='Train Multi-Task BERT with MoE')
    parser.add_argument('--num_experts', type=int, required=True, help='Number of experts')
    parser.add_argument('--load_balance_weight', type=float, default=0.005, help='Load balance weight')
    parser.add_argument('--kd_weight', type=float, required=True, help='Knowledge distillation weight')
    parser.add_argument('--kd_temperature', type=float, default=3.0, help='KD temperature')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_length', type=int, default=100, help='Max sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', default=False, help='Show detailed progress bars')
    parser.add_argument('--data_dir', type=str, default='../PASTA_data/new_processed_data/', help='Data directory')
    parser.add_argument('--teacher_pred_dir', type=str, default='../results_oss', help='Teacher predictions directory')
    parser.add_argument('--output_dir', type=str, default='./PASTA-Models/', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='', help='Optional experiment name prefix')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"MULTI-TASK BERT WITH MOE TRAINING")
    print(f"{'='*80}")
    print(f"Overall start time: {overall_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Configuration
    config = {
        'num_experts': args.num_experts,
        'load_balance_weight': args.load_balance_weight,
        'kd_weight': args.kd_weight,
        'kd_temperature': args.kd_temperature,
        'learning_rate': args.learning_rate,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'seed': args.seed,
        'verbose': args.verbose,
        'experiment_name': args.experiment_name,
        'timestamp': overall_start_datetime.isoformat()
    }
    
    # Dataset names
    dataset_names = [
        "anna_vs_carla",
        "breaking_down_hydrogen_peroxide",
        "gas_filled_balloons",
        "layers_in_test_tube",
        "model_for_making_water",
        "namis_careful_experiment",
        "natural_sugar"
    ]
    
    # Set initial random seed
    set_random_seeds(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using random seed: {config['seed']}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    all_data = {}
    task_configs = {}
    
    for dataset_name in dataset_names:
        train_file = os.path.join(args.data_dir, f'{dataset_name}_train.csv')
        
        if os.path.exists(train_file):
            print(f"\nLoading {dataset_name}...")
            
            # Load dataset with teacher predictions
            dataset_info = load_dataset_with_teacher_probs(
                train_file, 
                args.teacher_pred_dir if config['kd_weight'] > 0 else None, 
                dataset_name
            )
            
            all_data[dataset_name] = dataset_info
            task_configs[dataset_name] = {
                'num_labels': dataset_info['num_labels'],
                'label_names': dataset_info['label_names']
            }
    
    # Split data into train/val
    print("\nSplitting data into train/val...")
    all_train_data = {}
    all_val_data = {}
    
    for dataset_name, dataset_info in all_data.items():
        sentences = dataset_info['sentences']
        labels = dataset_info['labels']
        teacher_probs = dataset_info['teacher_probs']
        
        labels_str = [''.join(map(str, label)) for label in labels]
        label_counts = Counter(labels_str)
        single_instance_mask = [label_counts[label] == 1 for label in labels_str]
        multi_indices = [i for i, m in enumerate(single_instance_mask) if not m]
        single_indices = [i for i, m in enumerate(single_instance_mask) if m]
        multi_sentences = [sentences[i] for i in multi_indices]
        multi_labels = [labels[i] for i in multi_indices]
        multi_labels_str = [labels_str[i] for i in multi_indices]

        multi_teacher_probs = None
        single_teacher_probs = None
        if teacher_probs is not None:
            multi_teacher_probs = [teacher_probs[i] for i in multi_indices]
            single_teacher_probs = [teacher_probs[i] for i in single_indices]
        
        single_sentences = [sentences[i] for i in single_indices]
        single_labels = [labels[i] for i in single_indices]
        
        # Split multi-instance data with stratification
        if len(multi_sentences) > 10:  
            if multi_teacher_probs:
                train_sentences, val_sentences, train_labels, val_labels, train_teacher_probs, val_teacher_probs = train_test_split(
                    multi_sentences, multi_labels, multi_teacher_probs,
                    test_size=0.1, random_state=config['seed'], stratify=multi_labels_str
                )
            else:
                train_sentences, val_sentences, train_labels, val_labels = train_test_split(
                    multi_sentences, multi_labels,
                    test_size=0.1, random_state=config['seed'], stratify=multi_labels_str
                )
                train_teacher_probs = None
                val_teacher_probs = None
        else:
            train_sentences = multi_sentences
            train_labels = multi_labels
            train_teacher_probs = multi_teacher_probs
            val_sentences = []
            val_labels = []
            val_teacher_probs = None
        
        # Add single-instance samples to training
        train_sentences.extend(single_sentences)
        train_labels.extend(single_labels)
        if train_teacher_probs is not None and single_teacher_probs is not None:
            train_teacher_probs.extend(single_teacher_probs)
        elif single_teacher_probs is not None:
            train_teacher_probs = single_teacher_probs
        
        # Store the split data
        all_train_data[dataset_name] = {
            'sentences': train_sentences,
            'labels': train_labels,
            'teacher_probs': train_teacher_probs
        }
        
        all_val_data[dataset_name] = {
            'sentences': val_sentences,
            'labels': val_labels,
            'teacher_probs': val_teacher_probs
        }
        
        print(f"  {dataset_name}: {len(train_sentences)} train, {len(val_sentences)} val samples")
    
    # Create datasets and dataloaders
    train_dataset = MultiTaskDataset(all_train_data, tokenizer, config['max_length'])
    val_dataset = MultiTaskDataset(all_val_data, tokenizer, config['max_length'])
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn
    )
    
    print(f"\nTotal training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    
    # Train model with specified configuration
    results = train_model_with_config(config, train_loader, val_loader, task_configs, device, args.output_dir)
    
    # Record overall end time
    overall_end_time = time.time()
    overall_end_datetime = datetime.now()
    overall_duration = overall_end_time - overall_start_time
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Overall end time: {overall_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {overall_duration/3600:.2f} hours ({overall_duration/60:.1f} minutes)")
    print(f"Configuration: experts={config['num_experts']}, lb={config['load_balance_weight']}, kd={config['kd_weight']}, seed={config['seed']}")
    
    # Print final summary
    final_metrics = results.get('final_metrics', {})
    print(f"\nFinal Performance Summary:")
    print(f"  - Average Micro F1: {final_metrics.get('avg_micro_f1', 0):.4f}")
    print(f"  - Average Kappa: {final_metrics.get('avg_kappa', 0):.4f}")  
    print(f"  - Average Per-label Accuracy: {final_metrics.get('avg_per_label_acc', 0):.4f}")
    print(f"  - Best Validation Loss: {results['best_val_loss']:.4f}")
    print(f"  - Training completed in {results['epochs_trained']} epochs")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
