# Shallow Transfer Learning with 5-Fold Cross-Validation
# Freeze everything except the final layer

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

# Function to load the model for fine-tuning
def load_model_for_finetuning(name, num_classes):
    if name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for p in model.parameters(): 
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {num_trainable:,} / {num_total:,}  (shallow TL)")
        return model

    if name == 'vgg11':
        model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        for p in model.parameters(): 
            p.requires_grad = False
        in_feat = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feat, num_classes)
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {num_trainable:,} / {num_total:,}  (shallow TL)")
        return model


# Fine-tuning function for one fold
def train_one_fold(model_name, train_loader, val_loader, num_classes, epochs, fold_num):
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_num + 1}/{n_folds}")
    print(f"{'='*60}")
    
    # Load pre-trained model for fine-tuning
    model = load_model_for_finetuning(model_name, num_classes)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # Optimizer to fine-tune only the last layer
    if model_name == 'vgg11':
        optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.0001, weight_decay=0.0001)
    else:  # for resnet50
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=0.0001)

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
            if i % 30 == 29:
                avg_loss = running_loss / 30
                losses.append(avg_loss)
                print(f'[Epoch {epoch+1}/{epochs}, Step {i+1}] Loss: {avg_loss:.3f}')
                running_loss = 0.0

        scheduler.step()
        
        # Validation after each epoch
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
            # transforms.RandomRotation(20),
            # transforms.GaussianBlur(5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.7074, 0.2772, 0.0759], [0.1713, 0.1298, 0.0812]) # these values have to be extracted from training set of what you are using now, check get_mean_std_for_normalisation.py
        ]),
        'test': transforms.Compose([
            # transforms.RandomRotation(20),
            # transforms.GaussianBlur(5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.7074, 0.2772, 0.0759], [0.1713, 0.1298, 0.0812]) # these values have to be extracted from training set of what you are using now, check get_mean_std_for_normalisation.py
        ]),
    }

    data_dir = 'PlantVillage_2022'

    dsets = {split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
             for split in ['train', 'test']}

    dset_loaders = {
        'train': torch.utils.data.DataLoader(dsets['train'], batch_size=batch_size, shuffle=True),
        'test' : torch.utils.data.DataLoader(dsets['test'],  batch_size=batch_size, shuffle=False),
    }

    return dset_loaders['train'], dset_loaders['test']


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
    trainloader, testloader = load_data()
    
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
    
    # Final evaluation on original test set
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}\n")
    test_stats = evaluate_stats(best_fold_model, testloader)
    
    # Save the best model
    # 'C:\\Users\\Dan Loznean\\Desktop\\proiect_plant_master_y1_s1\\DeepLearning_PlantDiseases-master\\Scripts\\model_saves'
    # '/Users/horiaionescu/Main Folder/project_master_y1_s1/DeepLearning_PlantDiseases-master/Scripts/model_saves'

    save_folder = 'C:\\Users\\Dan Loznean\\Desktop\\proiect_plant_master_y1_s1\\DeepLearning_PlantDiseases-master\\Scripts\\model_saves'
    os.makedirs(save_folder, exist_ok=True)
    model_save_path = os.path.join(save_folder, f'{model_name}_shallowTL_5fold_best_2022.pth')
    torch.save(best_fold_model.state_dict(), model_save_path)
    print(f'\nBest model (fold {best_fold_num}) saved to {model_save_path}')
    
    return fold_results, test_stats, all_losses


# Main execution
if __name__ == "__main__":
    # Uses the same data structure as original:
    # PlantVillage_2019/
    #   ├── train/
    #   │   ├── other (Not alternaria solani)/
    #   │   └── Alternaria solani/
    #   └── test/
    #       ├── other (Not alternaria solani)/
    #       └── Alternaria solani/
    
    # Run 5-fold cross-validation on training data, then evaluate on test set
    # ResNet50
    #fold_results, test_stats, losses = run_kfold_training(
    #    'resnet50',
    ##   epochs=100
    #)
    
    # VGG11 (uncomment to use)
     fold_results, test_stats, losses = run_kfold_training(
         'vgg11',
         num_classes=2,
         epochs=100
     )


# ResNet50 5-fold train 2022
# Stats : ============================================================
# CROSS-VALIDATION RESULTS SUMMARY
# ============================================================
#
# accuracy       : 0.8231 ± 0.0220
#   Per fold: ['0.8648', '0.8056', '0.8082', '0.8107', '0.8261']
# precision_pos  : 0.8323 ± 0.0148
#   Per fold: ['0.8601', '0.8310', '0.8203', '0.8188', '0.8311']
# recall_pos     : 0.9242 ± 0.0198
#   Per fold: ['0.9545', '0.8939', '0.9167', '0.9242', '0.9318']
# f1_binary      : 0.8758 ± 0.0156
#   Per fold: ['0.9048', '0.8613', '0.8658', '0.8683', '0.8786']
# f1_macro       : 0.7842 ± 0.0269
#   Per fold: ['0.8357', '0.7683', '0.7648', '0.7660', '0.7861']
#
# Best fold: 1 with F1-macro: 0.8357
#
# ============================================================
# FINAL EVALUATION ON TEST SET
# ============================================================
#
# Accuracy: 0.8235 | Precision(+): 0.8300 | Recall(+): 0.9284 | F1(+): 0.8765 | F1-macro: 0.7837
#



# VGG11 5-fold train 2022
# Stats :

# ============================================================
# CROSS-VALIDATION RESULTS SUMMARY
# ============================================================
#
# accuracy       : 0.8277 ± 0.0208
#   Per fold: ['0.8622', '0.8005', '0.8235', '0.8159', '0.8363']
# precision_pos  : 0.8538 ± 0.0140
#   Per fold: ['0.8777', '0.8370', '0.8495', '0.8453', '0.8597']
# recall_pos     : 0.8985 ± 0.0163
#   Per fold: ['0.9242', '0.8750', '0.8977', '0.8902', '0.9053']
# f1_binary      : 0.8756 ± 0.0151
#   Per fold: ['0.9004', '0.8556', '0.8729', '0.8672', '0.8819']
# f1_macro       : 0.7977 ± 0.0244
#   Per fold: ['0.8386', '0.7666', '0.7921', '0.7836', '0.8076']
#
# Best fold: 1 with F1-macro: 0.8386
#
# ============================================================
# FINAL EVALUATION ON TEST SET
# ============================================================
#
# Accuracy: 0.8218 | Precision(+): 0.8363 | Recall(+): 0.9148 | F1(+): 0.8738 | F1-macro: 0.7854