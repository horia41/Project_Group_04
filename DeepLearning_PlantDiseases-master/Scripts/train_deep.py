# update entire model, not just last layer like in shallow transfer learning
import time
import os
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from torchvision.models import ResNet50_Weights, VGG11_Weights

# Configuration
torch.manual_seed(0)
batch_size = 20

use_gpu = torch.cuda.is_available()
print("For mac gpu available: ", torch.backends.mps.is_available())
print("For windows gpu available: ", torch.cuda.is_available())
device = torch.device("cuda" if use_gpu else "mps")
print(f"Using device: {device}")

# Model URLs (for ImageNet pre-trained weights)
# not used anymore, imported directly from torchvision, ignore
model_urls = {
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
}

# Function to load the model for deep transfer learning
def load_model_for_deep_finetuning(name, num_classes):
    # Use the correct weights argument
    if name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for p in model.parameters(): p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {num_trainable:,} / {num_total:,}  (deep TL)")
        return model

    if name == 'vgg11':
        model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        for p in model.parameters(): p.requires_grad = True
        in_feat = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feat, num_classes)
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {num_trainable:,} / {num_total:,}  (deep TL)")
        return model

# Deep fine-tuning function
def fine_tune_model(model_name, trainloader, valloader, testloader, num_classes, epochs=15):
    # Load pre-trained model for deep fine-tuning
    model = load_model_for_deep_finetuning(model_name, num_classes)

    # Send model to device (GPU/CPU)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)  # Move loss function to device

    # Optimizer to fine-tune the entire model
    if model_name == 'vgg11':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:  # for resnet50
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_f1 = -1.0
    best_state = None
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
            if i % 30 == 29:
                avg_loss = running_loss / 30
                losses.append(avg_loss)
                print(f'[Epoch {epoch+1}, Step {i+1}] Loss: {avg_loss:.3f}')
                running_loss = 0.0

        # validation part
        model.eval()
        with torch.no_grad():
            val_stats = evaluate_stats(model, valloader)
        f1m = float(val_stats.get('f1_macro', 0.0))
        if f1m > best_f1:
            best_f1 = f1m
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    print("Finished Fine-Tuning")
    print("\nTesting:")

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        print(f"Loaded best checkpoint (val F1-macro={best_f1:.4f}) for final test.")

    print("\nTesting:")
    evaluate_stats(model, testloader)

    # save model
    # here, modify path for your pc
    # keep in mind it has to end up in Scripts/model_saves
    save_folder = '/Users/horiaionescu/Main Folder/project_master_y1_s1/DeepLearning_PlantDiseases-master/Scripts/model_saves'
    model_save_path = os.path.join(save_folder, 'resnet50_deepTL_100epochs_seed0.pth') # modify the name accordingly
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    return losses

# Function to load data and apply data augmentation
# Splits it as follows, 80/20 train/val for 2019, test fully on 2022
def load_data(resize):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.GaussianBlur(5),
            transforms.RandomResizedCrop(resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.GaussianBlur(5),
            transforms.Resize(int(resize / 224 * 256)),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.GaussianBlur(5),
            transforms.Resize(int(resize / 224 * 256)),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # data_dir = "PlantVillage"
    data_dir = '/Users/horiaionescu/Main Folder/project_master_y1_s1/DeepLearning_PlantDiseases-master/Scripts/PlantVillage'

    dsets = {split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
             for split in ['train', 'val', 'test']}

    dset_loaders = {
        'train': torch.utils.data.DataLoader(dsets['train'], batch_size=batch_size, shuffle=True),
        'val'  : torch.utils.data.DataLoader(dsets['val'],   batch_size=batch_size, shuffle=False),
        'test' : torch.utils.data.DataLoader(dsets['test'],  batch_size=batch_size, shuffle=False),
    }

    return dset_loaders['train'], dset_loaders['val'], dset_loaders['test']


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


# Fine-tune model
trainloader, valloader, testloader = load_data(224)  # Set resize to 224 for models like ResNet50

# Fine-tune model for 15 or 100 epochs
# resnet50
losses = fine_tune_model('resnet50', trainloader, valloader, testloader, num_classes=2, epochs=100)

# vgg11
# losses = fine_tune_model('vgg11', trainloader, testloader, num_classes=2, epochs=15)



# Results ResNet50 - learning rate = 0.001
# trained (15 epochs) on 2019, validation also on 2019, tested on 2022
# Stats : Accuracy: 0.7837 | Precision(+): 0.9750 | Recall(+): 0.6972 | F1(+): 0.8130 | F1-macro: 0.7782

# trained (100 epochs) on 2019, validation also on 2019,tested on 2022
# Stats : Accuracy: 0.8622 | Precision(+): 0.9544 | Recall(+): 0.8357 | F1(+): 0.8911 | F1-macro: 0.8518

# trained (15 epochs) on 2019, tested also on 2019
# Accuracy : 0.89623

# trained (100 epochs) on 2019, tested also on 2019
# Accuracy on test images: 0.867925  - most probably overfitted due to small dataset and at the same time having a big number of epochs


# Results VGG11 - learning rate = 0.001
# trained (15 epochs) on 2019, validation also on 2019,tested on 2022
# Stats : Accuracy: 0.8068 | Precision(+): 0.9691 | Recall(+): 0.7371 | F1(+): 0.8373 | F1-macro: 0.7997

# trained (100 epochs) on 2019, validation also on 2019,tested on 2022
# Stats : Accuracy: 0.8153 | Precision(+): 0.9585 | Recall(+): 0.7592 | F1(+): 0.8473 | F1-macro: 0.8069