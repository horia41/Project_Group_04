# Knowledge Distillation Training Script
# Teacher: ResNetCSWinHybrid (new_model)
# Student: EfficientNet

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import timm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
from cswin_fpn_hybrid.resnet50_cswin.new_model import ResNetCSWinHybrid


# Device configuration
use_gpu = torch.cuda.is_available()
print("For mac gpu available: ", torch.backends.mps.is_available())
print("For windows gpu available: ", torch.cuda.is_available())
device = torch.device("cuda" if use_gpu else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
temperature = 4.0  # Temperature for distillation
alpha = 0.7  # Weight for distillation loss (1-alpha for hard label loss)

# Offline augmentation configuration
USE_AUGMENTED_DATASET = False
ONLINE_AUGMENTATION_WITH_OFFLINE = True


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
    - Soft target loss (KL divergence with teacher)
    - Hard target loss (Cross Entropy with ground truth)
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets: KL divergence between student and teacher
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        # Hard targets: standard cross entropy with ground truth
        student_loss = self.ce_loss(student_logits, labels)

        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        return total_loss, distillation_loss, student_loss


def load_teacher_model(checkpoint_path, num_classes=2):
    """
    Load the pretrained teacher model (ResNetCSWinHybrid)
    """
    print("\n" + "="*60)
    print("Loading Teacher Model: ResNetCSWinHybrid")
    print("="*60)

    model = ResNetCSWinHybrid(
        num_classes=num_classes,
        resnet_pretrained=True,
        cswin_pretrained=True
    )

    # Load pretrained weights if checkpoint exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading teacher weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Teacher weights loaded successfully!")
    else:
        print("WARNING: No checkpoint provided. Using randomly initialized teacher.")
        print("For effective distillation, please provide a trained teacher model.")

    model.eval()  # Set to evaluation mode
    for param in model.parameters():
        param.requires_grad = False  # Freeze teacher

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Teacher parameters: {num_params:,}")
    print("="*60 + "\n")

    return model


def load_student_model(model_name='efficientnet_b0', num_classes=2):
    """
    Load the student model (EfficientNet)
    """
    print("\n" + "="*60)
    print(f"Loading Student Model: {model_name}")
    print("="*60)

    # Create EfficientNet student
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Student total parameters: {num_params:,}")
    print(f"Student trainable parameters: {num_trainable:,}")
    print("="*60 + "\n")

    return model


def train_with_distillation(
    teacher_checkpoint,
    student_name='efficientnet_b0',
    trainloader=None,
    testloader=None,
    num_classes=2,
    epochs=100,
    temperature=4.0,
    alpha=0.7
):
    """
    Train student model using knowledge distillation from teacher
    """
    # Load models
    teacher = load_teacher_model(teacher_checkpoint, num_classes).to(device)
    student = load_student_model(student_name, num_classes).to(device)

    # Setup optimizer with discriminative learning rates
    # Typically use higher LR for classifier head, lower for backbone
    backbone_params = []
    head_params = []

    for name, param in student.named_parameters():
        if 'classifier' in name or 'fc' in name or 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 5e-4}
    ], weight_decay=0.05)

    # Setup loss function
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)

    # Setup learning rate scheduler
    warmup_epochs = 10
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_main = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_main], milestones=[warmup_epochs])

    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Temperature: {temperature}")
    print(f"Alpha (distillation weight): {alpha}")
    print(f"Backbone LR: 1e-4")
    print(f"Head LR: 5e-4")
    print(f"Weight decay: 0.05")
    print("="*60 + "\n")

    losses = []
    best_accuracy = 0.0

    # Training loop
    for epoch in range(epochs):
        student.train()
        teacher.eval()

        running_loss = 0.0
        running_distill_loss = 0.0
        running_student_loss = 0.0
        epoch_loss = 0.0

        # Progress bar for batches
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)

        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass through both models
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)

            # Compute combined loss
            total_loss, distill_loss, student_loss = criterion(student_logits, teacher_logits, labels)

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            # Track losses
            running_loss += total_loss.item()
            running_distill_loss += distill_loss.item()
            running_student_loss += student_loss.item()
            epoch_loss += total_loss.item()

            if i % 20 == 19:
                avg_loss = running_loss / 20
                avg_distill = running_distill_loss / 20
                avg_student = running_student_loss / 20
                losses.append(avg_loss)
                pbar.set_postfix({
                    'Total': f'{avg_loss:.3f}',
                    'Distill': f'{avg_distill:.3f}',
                    'Student': f'{avg_student:.3f}'
                })
                running_loss = 0.0
                running_distill_loss = 0.0
                running_student_loss = 0.0

        # Print epoch summary
        epoch_avg_loss = epoch_loss / len(trainloader)
        print(f'Epoch {epoch+1}/{epochs} - Average Loss: {epoch_avg_loss:.4f}')

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"\nEvaluating student model at epoch {epoch+1}...")
            accuracy = quick_evaluate(student, testloader, device)

            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_folder = 'cswin_fpn_hybrid/model_saves'
                os.makedirs(save_folder, exist_ok=True)
                model_save_path = os.path.join(save_folder, f'student_{student_name}_best.pth')
                torch.save(student.state_dict(), model_save_path)
                print(f"New best model saved! Accuracy: {accuracy:.4f}")

        scheduler.step()

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    print("\nFinding best threshold for student:")
    optimal_t = analyze_test_thresholds(student, testloader, device)
    print(f"\nOptimal Threshold: {optimal_t}")

    print("\nStudent model evaluation at best threshold:")
    evaluate_stats(student, testloader, device, threshold=optimal_t)

    print("\nStudent model evaluation at default threshold (0.5):")
    evaluate_stats(student, testloader, device, threshold=0.5)

    # Compare with teacher
    if teacher_checkpoint and os.path.exists(teacher_checkpoint):
        print("\nTeacher model evaluation for comparison:")
        evaluate_stats(teacher, testloader, device, threshold=0.5)

    # Save final model
    save_folder = 'cswin_fpn_hybrid/model_saves'
    os.makedirs(save_folder, exist_ok=True)
    model_save_path = os.path.join(save_folder, f'student_{student_name}_final_threshold_{optimal_t:.2f}.pth')
    torch.save(student.state_dict(), model_save_path)
    print(f'\nFinal student model saved to {model_save_path}')

    return losses, student


def quick_evaluate(model, testloader, device):
    """Quick accuracy evaluation"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy


def evaluate_stats(net, testloader, device, threshold=0.5):
    """Detailed evaluation with metrics"""
    net.eval()

    all_preds = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            # Apply threshold
            predicted = (probs >= threshold).long()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    eval_time = time.time() - start_time

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_binary = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    stats = {
        'accuracy': float(acc),
        'precision_pos': float(precision),
        'recall_pos': float(recall),
        'f1_binary': float(f1_binary),
        'f1_macro': float(f1_macro),
        'eval_time': eval_time
    }

    print(f'Threshold: {threshold:.2f} | Acc: {acc:.4f} | Prec(+): {precision:.4f} | Rec(+): {recall:.4f} | F1(+): {f1_binary:.4f} | F1-macro: {f1_macro:.4f}')

    return stats


def analyze_test_thresholds(model, test_loader, device):
    """Find optimal threshold for binary classification"""
    model.eval()

    y_true = []
    y_probs = []

    print(f"Test Set ({len(test_loader.dataset)} images)")

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            y_true.extend(targets.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    print("\n")
    print(f"{'Threshold':<10} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 55)

    best_f1 = 0
    best_thresh = 0.5

    for thresh in np.arange(0.01, 0.96, 0.02):
        preds = (y_probs >= thresh).astype(int)

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)

        marker = " (*)" if abs(thresh - 0.5) < 0.01 else ""

        print(f"{thresh:.2f}{marker:<3}    | {acc:.4f}   | {prec:.4f}   | {rec:.4f}   | {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print("-" * 55)
    print(f"Default (0.50) row marked (*)")
    print(f"Optimal (best F1) threshold: {best_thresh:.2f} | F1: {best_f1:.4f}")

    return round(best_thresh, 2)


def load_data():
    """Load and prepare data with augmentation"""
    # Select dataset path based on configuration
    if USE_AUGMENTED_DATASET:
        data_dir = 'DeepLearning_PlantDiseases-master/Scripts/PlantVillage_1_2019train_2022test_augmented'
        print(f"Using AUGMENTED dataset: {data_dir}")
    else:
        data_dir = 'DeepLearning_PlantDiseases-master/Scripts/PlantVillage_1_2019train_2022test'
        print(f"Using ORIGINAL dataset: {data_dir}")

    # Configure training augmentation based on settings
    if USE_AUGMENTED_DATASET and ONLINE_AUGMENTATION_WITH_OFFLINE:
        print("Augmentation mode: Offline + Light Online (RECOMMENDED)")
        train_transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(p=0.3),
            v2.ToTensor(),
            v2.Normalize([0.7553, 0.3109, 0.1059], [0.1774, 0.1262, 0.0863]),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])
    elif USE_AUGMENTED_DATASET and not ONLINE_AUGMENTATION_WITH_OFFLINE:
        print("Augmentation mode: Offline Only")
        train_transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize([0.7553, 0.3109, 0.1059], [0.1774, 0.1262, 0.0863]),
        ])
    else:
        print("Augmentation mode: Original Heavy Online")
        train_transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.ToTensor(),
            v2.Normalize([0.7553, 0.3109, 0.1059], [0.1774, 0.1262, 0.0863]),
            v2.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ])

    data_transforms = {
        'train': train_transform,
        'test': v2.Compose([
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize([0.7553, 0.3109, 0.1059], [0.1774, 0.1262, 0.0863])
        ]),
    }

    dsets = {split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
             for split in ['train', 'test']}

    dset_loaders = {
        'train': torch.utils.data.DataLoader(dsets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'test': torch.utils.data.DataLoader(dsets['test'], batch_size=batch_size, shuffle=False, num_workers=4),
    }

    return dset_loaders['train'], dset_loaders['test']


if __name__ == '__main__':
    # Path to your trained teacher model
    # Update this path to point to your trained ResNetCSWinHybrid checkpoint
    teacher_checkpoint = 'cswin_fpn_hybrid/model_saves/threshold_0.50_hybrid_Tr2019_Te2022.pth'

    # Load data
    trainloader, testloader = load_data()

    # Train student with knowledge distillation
    losses, trained_student = train_with_distillation(
        teacher_checkpoint=teacher_checkpoint,
        student_name='efficientnet_b0',  # Can change to efficientnet_b1, b2, etc.
        trainloader=trainloader,
        testloader=testloader,
        num_classes=2,
        epochs=100,
        temperature=4.0,
        alpha=0.7
    )

    print("\n" + "="*60)
    print("Knowledge Distillation Training Complete!")
    print("="*60)
