import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from hybrid_hpem_cswin import HybridCSWinClassifier


use_gpu = torch.cuda.is_available()
print("For mac gpu available: ", torch.backends.mps.is_available())
print("For windows gpu available: ", torch.cuda.is_available())
device = torch.device("cuda" if use_gpu else "mps")
print(f"Using device: {device}")
batch_size = 128


def load_model_for_deep_finetuning(name, num_classes):

    if name == 'hybrid_cswin':

        # this must match the pre-trained model's structure, see models.py
        cswin_kwargs_tiny = dict(
            img_size=256, # Using 256x256 images
            in_chans=3,
            embed_dim=64,
            depth=[1, 2, 21, 1],
            split_size=[1, 2, 8, 8],
            num_heads=[2, 4, 8, 16],
            mlp_ratio=4.,
        )

        model = HybridCSWinClassifier(num_classes=num_classes, **cswin_kwargs_tiny)

        timm_model = timm.create_model('CSWin_64_12211_tiny_224', pretrained=True)
        pretrained_weights = timm_model.state_dict()

        weights_to_load = {}
        for key, value in pretrained_weights.items():
            if key.startswith('stage1_conv_embed.') or key.startswith('head.'):
                continue  # skip this, we have HPEM now
            weights_to_load[key] = value

        model.load_state_dict(weights_to_load, strict=False)
        print("Successfully loaded pretrained backbone weights for CSWin")

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {num_trainable:,} / {num_total:,}")

        return model


# Deep fine-tuning function
def fine_tune_model(model_name, trainloader, testloader, num_classes, epochs=15):
    model = load_model_for_deep_finetuning(model_name, num_classes)

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)


    if model_name == 'hybrid_cswin':
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

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

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                avg_loss = running_loss / 10
                losses.append(avg_loss)
                print(f'[Epoch {epoch+1}, Step {i+1}] Loss: {avg_loss:.3f}')
                running_loss = 0.0

        scheduler.step()

    print("Finished Fine-Tuning")
    print("\nTesting:")

    print("\nTesting:")
    evaluate_stats(model, testloader)

    # save model
    # save_folder = './model_saves' # Save to a local folder
    # os.makedirs(save_folder, exist_ok=True)

    # Modify the name accordingly
    # model_save_path = os.path.join(save_folder, f'{model_name}_deepTL_{epochs}epochs.pth')
    # torch.save(model.state_dict(), model_save_path)
    # print(f'Model saved to {model_save_path}')

    return losses


# Function to load data and apply data augmentation
def load_data():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)), # Ensure all images are 256x256
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]),
    }

    data_dir = '/Users/horiaionescu/Main Folder/project_master_y1_s1/DeepLearning_PlantDiseases-master/Scripts/PlantVillage_1_2019train_2022test'

    dsets = {split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
             for split in ['train', 'test']}

    dset_loaders = {
        'train': torch.utils.data.DataLoader(dsets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'test' : torch.utils.data.DataLoader(dsets['test'],  batch_size=batch_size, shuffle=False, num_workers=4),
    }

    return dset_loaders['train'], dset_loaders['test']


def evaluate_stats(net, testloader):
    net.eval()
    stats = {}

    total = 0            # Python ints (avoid MPS float64 issues)
    correct = 0

    all_preds = []
    all_labels = []

    before = time.time()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()   # .item() -> Python int

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

    print(f'Accuracy: {accuracy:.4f} | Precision(+): {precision:.4f} | Recall(+): {recall:.4f} | F1(+): {f1_binary:.4f} | F1-macro: {f1_macro:.4f}')
    return stats


if __name__ == '__main__':

    trainloader, testloader = load_data()

    losses = fine_tune_model(
        model_name='hybrid_cswin',
        trainloader=trainloader,
        testloader=testloader,
        num_classes=2,
        epochs=100
    )