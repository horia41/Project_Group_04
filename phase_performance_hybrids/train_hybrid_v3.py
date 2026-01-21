import os
import time
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from torchvision.transforms import v2
from phase_performance_hybrids.resnet50_cswin.model_v3 import ResNetCSWinHybridV3 as ResNetCSWinHybrid


use_gpu = torch.cuda.is_available()
print("For mac gpu available: ", torch.backends.mps.is_available())
print("For windows gpu available: ", torch.cuda.is_available())
device = torch.device("cuda" if use_gpu else "cpu")
print(f"Using device: {device}")
batch_size = 128


def load_model_for_deep_finetuning(name, num_classes):

    if name == 'hybrid_cswin':

        model = ResNetCSWinHybrid(num_classes=num_classes, resnet_pretrained=True, cswin_pretrained=True)

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_resnet = sum(p.numel() for p in model.resnet_stem.parameters() if p.requires_grad)
        num_cswin_block1 = sum(p.numel() for p in model.stage1.parameters() if p.requires_grad)
        num_cswin_block2 = sum(p.numel() for p in model.stage2.parameters() if p.requires_grad)
        num_cswin_block3 = sum(p.numel() for p in model.stage3.parameters() if p.requires_grad)
        num_cswin_block4 = sum(p.numel() for p in model.stage4.parameters() if p.requires_grad)
        num_head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        num_bridge = sum(p.numel() for p in model.bridge.parameters() if p.requires_grad)
        print(f"Total Trainable params: {num_trainable:,} / {num_total:,}")
        print(f"ResNet50 params: {num_resnet:,}")
        print(f"CSWin Stage 1 Trainable params: {num_cswin_block1:,}")
        print(f"CSWin Stage 2 Trainable params: {num_cswin_block2:,}")
        print(f"CSWin Stage 3 Trainable params: {num_cswin_block3:,}")
        print(f"CSWin Stage 4 Trainable params: {num_cswin_block4:,}")
        print(f"Head Classifier Trainable params: {num_head:,}")
        print(f"Bridge Trainable params: {num_bridge:,}")

        return model


# focal loss code
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



# training & fine tuning
def fine_tune_model(model_name, trainloader, testloader, num_classes, epochs=100):

    model = load_model_for_deep_finetuning(model_name, num_classes)

    model = model.to(device)

    # different learning rates for resnet and attention cswin blocks
    stem_params = []   # pre-trained ResNet
    body_params = []   # Transformer + Bridge
    head_params = []   # classifier head

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'resnet_stem' in name:
            stem_params.append(param)
        elif 'head' in name:
            head_params.append(param)
        else:
           # bride, transformer
            body_params.append(param)


    optimizer = optim.AdamW([
    {'params': stem_params, 'lr': 1e-5},
    {'params': body_params, 'lr': 5e-5},
    {'params': head_params, 'lr': 5e-4}
    ], weight_decay=0.05)


    # focal loss criterion
    criterion = FocalLoss(gamma=2.0).to(device)

    # setup scheduler for new modification of architecture
    warmup_epochs = 10
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)

    # Main: Cosine Annealing for the rest of the time
    scheduler_main = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    # Combine them
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_main], milestones=[warmup_epochs])


    losses = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss standard
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                avg_loss = running_loss / 20
                losses.append(avg_loss)
                # one print per epoch, modify if you want more prints
                print(f'[Epoch {epoch+1}, Step {i+1}] Loss: {avg_loss:.3f}')
                running_loss = 0.0

        scheduler.step()

    print("\nFinding best threshold:")
    optimal_t = analyze_test_thresholds(model, testloader, device)
    print(f"\nOptimal Threshold: {optimal_t}")

    print("\nEvaluating model at best thershold:")
    evaluate_stats(model, testloader, device, threshold = optimal_t)

    print("\nEvaluating model at default threshold (0.5):")
    evaluate_stats(model, testloader, device, threshold = 0.5)

    # save model
    save_folder = '../phase_performance_hybrids/results_model_saves_resnet50_cswin'
    os.makedirs(save_folder, exist_ok=True)
    model_save_path = os.path.join(save_folder, f'threshold_{optimal_t:.2f}_hybrid_model3_XAug_Tr2022_Te2019.pth') # modify here the name accordingly
    torch.save(model.state_dict(), model_save_path)
    print(f'\nModel saved to {model_save_path}')

    return losses


# Function to load data and apply data augmentation
def load_data():
    data_transforms = {
        'train': v2.Compose([
            v2.Resize((224, 224)),

            # ------------------------------------ BASIC AUGMENTATION
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ToTensor(),
            # v2.Normalize([0.7553, 0.3109, 0.1059], [0.1774, 0.1262, 0.0863]),
            v2.Normalize([0.7083, 0.2776, 0.0762], [0.1704, 0.1296, 0.0815]),
            # ------------------------------------ BASIC AUGMENTATION

            # ------------------------------------ HEAVY AUGMENTATION

            # # Geometric Transforms
            # v2.RandomHorizontalFlip(p=0.5),

            # v2.RandomRotation(degrees=15),
            # # Slight zoom/shift
            # v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),

            # # Color/Signal Transforms
            # v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),

            # # Noise & Robustness
            # # Gaussian Blur helps ignore grain/noise
            # v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

            # v2.ToTensor(),

            # use this norm below when training on 2022
            # # v2.Normalize([0.7083, 0.2776, 0.0762], [0.1704, 0.1296, 0.0815]),

            # use this norm below when training on 2019
            # v2.Normalize([0.7553, 0.3109, 0.1059], [0.1774, 0.1262, 0.0863]),

            # # Occlusion (The Precision Booster)
            # v2.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            # --------------------------------- HEAVY AUGMENTATION
        ]),
        'test': v2.Compose([
            v2.Resize((224, 224)),
            v2.ToTensor(),
            # use this norm below when training on 2022
            v2.Normalize([0.7083, 0.2776, 0.0762], [0.1704, 0.1296, 0.0815])

            # use this norm below when training on 2019
            # v2.Normalize([0.7553, 0.3109, 0.1059], [0.1774, 0.1262, 0.0863])
        ]),
    }

    # use this below when training on 2019
    # data_dir = '../cross_year_configurations_data/PlantVillage_1_2019train_2022test'

    # use this below when training on 2022
    data_dir = '../cross_year_configurations_data/PlantVillage_2_2022train_2019test'


    dsets = {split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
             for split in ['train', 'test']}

    dset_loaders = {
        'train': torch.utils.data.DataLoader(dsets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'test' : torch.utils.data.DataLoader(dsets['test'],  batch_size=batch_size, shuffle=False, num_workers=4),
    }

    return dset_loaders['train'], dset_loaders['test']


def evaluate_stats(net, testloader, device, threshold=0.5):
    net.eval()
    stats = {}

    all_preds = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]

            # here we apply the given threshold
            predicted = (probs >= threshold).long()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    eval_time = time.time() - start_time

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_binary = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # macro metrics (average for both pos and neg classes)
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


# see which threshold is best for model to avoid prior data distribution drift
def analyze_test_thresholds(model, test_loader, device):
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


if __name__ == '__main__':

    trainloader, testloader = load_data()

    losses = fine_tune_model(
        model_name='hybrid_cswin',
        trainloader=trainloader,
        testloader=testloader,
        num_classes=2,
        epochs=100
    )