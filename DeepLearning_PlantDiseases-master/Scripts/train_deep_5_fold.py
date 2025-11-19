# Deep Transfer Learning with 5-Fold Cross-Validation
# Update entire model, not just last layer

import time
import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import ResNet50_Weights, VGG11_Weights
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import SubsetRandomSampler

# Configuration
batch_size = 128
n_folds = 5
random_seed = 42

use_gpu = torch.cuda.is_available()
print("For mac gpu available: ", torch.backends.mps.is_available())
print("For windows gpu available: ", torch.cuda.is_available())
device = torch.device("cuda" if use_gpu else "mps")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Function to load the model for deep fine-tuning
def load_model_for_deep_finetuning(name, num_classes):
    if name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for p in model.parameters():
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {num_trainable:,} / {num_total:,}  (deep TL)")
        return model

    if name == 'vgg11':
        model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = True
        in_feat = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feat, num_classes)
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {num_trainable:,} / {num_total:,}  (deep TL)")
        return model


# Deep fine-tuning function for one fold
def train_one_fold(model_name, train_loader, val_loader, num_classes, epochs, fold_num):
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_num + 1}/{n_folds}")
    print(f"{'='*60}")

    # Load pre-trained model for deep fine-tuning
    model = load_model_for_deep_finetuning(model_name, num_classes)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # Optimizer to fine-tune the entire model
    if model_name == 'vgg11':
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    else:  # for resnet50
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    losses = []
    best_f1 = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses.append(loss.item())
            print(f'[Epoch {epoch+1}/{epochs}, Batch {i+1}] Loss: {loss.item():.3f}')
        print(f'  Epoch {epoch+1}/{epochs} completed')
        scheduler.step()

        # Validation after each epoch (every 10 epochs or last epoch)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"\nValidation at epoch {epoch+1}:")
            val_stats = evaluate_stats(model, val_loader)
            if val_stats['f1_macro'] > best_f1:
                best_f1 = val_stats['f1_macro']
                best_epoch = epoch + 1

    print(f"\nFold {fold_num + 1} Training Complete!")
    print(f"Best F1-macro: {best_f1:.4f} at epoch {best_epoch}")

    # Final validation
    print(f"\nFinal Validation Results for Fold {fold_num + 1}:")
    final_stats = evaluate_stats(model, val_loader)

    return model, final_stats, losses


# Function to load data and apply data augmentation
def load_data():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.7553, 0.3109, 0.1059], [0.1774, 0.1262, 0.0863]) # these values have to be extracted from training set of what you are using now, check get_mean_std_for_normalisation.py
        ])
    }

    data_dir = 'PlantVillage_2019'

    dsets = {split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
             for split in ['train']}

    dset_loaders = {
        'train': torch.utils.data.DataLoader(dsets['train'], batch_size=batch_size, shuffle=True)
    }

    return dset_loaders['train']


# Function to create data loaders for a specific fold
def get_fold_loaders(dataset, train_indices, val_indices):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler
    )

    return train_loader, val_loader


# Function to evaluate the model's performance
def evaluate_stats(net, dataloader):
    net.eval()
    stats = {}

    total = 0
    correct = 0

    all_preds = []
    all_labels = []

    before = time.time()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # collect for F1
            all_preds.append(predicted.detach().cpu())
            all_labels.append(labels.detach().cpu())

    elapsed = time.time() - before

    preds = torch.cat(all_preds)
    targets = torch.cat(all_labels)

    # Accuracy (Python float)
    accuracy = correct / total if total > 0 else 0.0

    # --- Binary P/R/F1 (positive class assumed label 1 if present, else max label) ---
    classes = targets.unique().tolist()
    pos_cls = 1 if 1 in classes else int(max(classes))

    tp = ((preds == pos_cls) & (targets == pos_cls)).sum().item()
    fp = ((preds == pos_cls) & (targets != pos_cls)).sum().item()
    fn = ((preds != pos_cls) & (targets == pos_cls)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_binary = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # --- Macro-F1 over all classes ---
    f1_per_class = []
    for c in classes:
        tp_c = ((preds == c) & (targets == c)).sum().item()
        fp_c = ((preds == c) & (targets != c)).sum().item()
        fn_c = ((preds != c) & (targets == c)).sum().item()
        prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec_c  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c   = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0
        f1_per_class.append(f1_c)
    f1_macro = float(sum(f1_per_class) / len(f1_per_class)) if f1_per_class else 0.0

    stats.update({
        'accuracy': float(accuracy),
        'precision_pos': float(precision),
        'recall_pos': float(recall),
        'f1_binary': float(f1_binary),
        'f1_macro': float(f1_macro),
        'eval_time': elapsed,
    })

    print(f'Accuracy: {accuracy:.4f} | Precision(+): {precision:.4f} | '
          f'Recall(+): {recall:.4f} | F1(+): {f1_binary:.4f} | F1-macro: {f1_macro:.4f}')
    return stats


# Main cross-validation function
def run_kfold_training(model_name, num_classes, epochs):
    print(f"\n{'#'*60}")
    print(f"Starting {n_folds}-Fold Cross-Validation")
    print(f"Model: {model_name} | Epochs: {epochs}")
    print(f"{'#'*60}\n")

    # Load the training dataset (using original data structure)
    trainloader = load_data()

    # Get the full training dataset for cross-validation
    train_dataset = trainloader.dataset

    # Get labels for stratification
    labels = [label for _, label in train_dataset.samples]

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # Storage for results
    fold_results = []
    all_losses = []
    best_fold_model = None
    best_fold_f1 = 0.0
    best_fold_num = 0

    # Cross-validation loop (only on training data)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # Create data loaders for this fold
        fold_train_loader, fold_val_loader = get_fold_loaders(train_dataset, train_idx, val_idx)

        print(f"\nFold {fold + 1}: Train samples = {len(train_idx)}, Val samples = {len(val_idx)}")

        # Train model for this fold
        model, fold_stats, losses = train_one_fold(
            model_name, fold_train_loader, fold_val_loader, num_classes, epochs, fold
        )

        # Store results
        fold_results.append(fold_stats)
        all_losses.extend(losses)

        # Keep track of best model
        if fold_stats['f1_macro'] > best_fold_f1:
            best_fold_f1 = fold_stats['f1_macro']
            best_fold_model = model
            best_fold_num = fold + 1

    # Aggregate results across all folds
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'='*60}\n")

    metrics = ['accuracy', 'precision_pos', 'recall_pos', 'f1_binary', 'f1_macro']

    for metric in metrics:
        values = [result[metric] for result in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric:15s}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"  Per fold: {[f'{v:.4f}' for v in values]}")

    print(f"\nBest fold: {best_fold_num} with F1-macro: {best_fold_f1:.4f}")

    # Save the best model
    save_folder = 'model_saves'
    os.makedirs(save_folder, exist_ok=True)
    model_save_path = os.path.join(save_folder, f'{model_name}_deepTL_5fold_best.pth')
    torch.save(best_fold_model.state_dict(), model_save_path)
    print(f'\nBest model (fold {best_fold_num}) saved to {model_save_path}')

    return fold_results, all_losses


# Main execution
if __name__ == "__main__":
    # Uses the same data structure as original:
    # PlantVillage_1_2019train_2022test/
    #   ├── train/
    #   │   ├── other (Not alternaria solani)/
    #   │   └── Alternaria solani/
    #   └── test/
    #       ├── other (Not alternaria solani)/
    #       └── Alternaria solani/

    # Run 5-fold cross-validation on training data, then evaluate on test set
    # ResNet50
    fold_results, losses = run_kfold_training(
        'resnet50',
        num_classes=2,
        epochs=100
    )

    # VGG11 (uncomment to use)
    # fold_results, test_stats, losses = run_kfold_training(
    #     'vgg11',
    #     num_classes=2,
    #     epochs=100
    # )


# ResNet50 5-fold deep train/test 2022
# ------------------------------------------------------------------------------------------------------------
# WRONG
# ------------------------------------------------------------------------------------------------------------
# CROSS-VALIDATION RESULTS SUMMARY
# accuracy       : 0.9468 ± 0.0044
# Per fold: ['0.9464', '0.9540', '0.9488', '0.9437', '0.9412']
# precision_pos  : 0.9460 ± 0.0117
# Per fold: ['0.9418', '0.9624', '0.9552', '0.9416', '0.9288']
# recall_pos     : 0.9773 ± 0.0072
# Per fold: ['0.9811', '0.9697', '0.9697', '0.9773', '0.9886']
# f1_binary      : 0.9613 ± 0.0029
# Per fold: ['0.9610', '0.9660', '0.9624', '0.9591', '0.9578']
# f1_macro       : 0.9382 ± 0.0058
# Per fold: ['0.9377', '0.9473', '0.9412', '0.9345', '0.9304']
# Best fold: 2 with F1-macro: 0.9473
# FINAL EVALUATION ON TEST SET
# Accuracy: 0.9517 | Precision(+): 0.9552 | Recall(+): 0.9741 | F1(+): 0.9645 | F1-macro: 0.9444
# Paper reports on ResNet50 deep train/test 2022 :
# F1: 0.91, ours 0.96
# Acc: 0.94, ours 0.95
# Precision: 0.93, ours 0.95
# Recall: 0.89, ours 0.97


# ResNet50 5-fold deep train/test 2022
# ------------------------------------------------------------------------------------------------------------
# CORRECT
# ------------------------------------------------------------------------------------------------------------


# VGG11 5-fold deep train/test 2022
# ------------------------------------------------------------------------------------------------------------
# WRONG
# ------------------------------------------------------------------------------------------------------------
# CROSS-VALIDATION RESULTS SUMMARY
# accuracy       : 0.9473 ± 0.0116
# Per fold: ['0.9643', '0.9488', '0.9437', '0.9514', '0.9284']
# precision_pos  : 0.9600 ± 0.0118
# Per fold: ['0.9771', '0.9586', '0.9618', '0.9623', '0.9403']
# recall_pos     : 0.9621 ± 0.0063
# Per fold: ['0.9697', '0.9659', '0.9545', '0.9659', '0.9545']
# f1_binary      : 0.9611 ± 0.0085
# Per fold: ['0.9734', '0.9623', '0.9582', '0.9641', '0.9474']
# f1_macro       : 0.9399 ± 0.0135
# Per fold: ['0.9596', '0.9414', '0.9361', '0.9445', '0.9177']
# Best fold: 1 with F1-macro: 0.9596
# FINAL EVALUATION ON TEST SET
# Accuracy: 0.9509 | Precision(+): 0.9563 | Recall(+): 0.9716 | F1(+): 0.9639 | F1-macro: 0.9436
# Paper reports on VGG11 deep train/test 2022 :
# F1: 0.92, ours 0.96
# Acc: 0.95, ours 0.95
# Precision: 0.93, ours 0.95
# Recall: 0.92, ours 0.97


# VGG11 5-fold deep train/test 2022
# ------------------------------------------------------------------------------------------------------------
# CORRECT
# ------------------------------------------------------------------------------------------------------------


# ResNet50 5-fold deep train/test 2019
# ------------------------------------------------------------------------------------------------------------
# CORRECT
# ------------------------------------------------------------------------------------------------------------


# VGG11 5-fold deep train/test 2019
# ------------------------------------------------------------------------------------------------------------
# CORRECT
# ------------------------------------------------------------------------------------------------------------