# cswin_hybrid training

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from cswin_fpn_hybrid.resnet50_cswin import ResNetCSWinHybrid


use_gpu = torch.cuda.is_available()
print("For mac gpu available: ", torch.backends.mps.is_available())
print("For windows gpu available: ", torch.cuda.is_available())
device = torch.device("cuda" if use_gpu else "cpu")
print(f"Using device: {device}")
batch_size = 128


def load_model_for_deep_finetuning(name, num_classes):

    if name == 'hybrid_cswin':

        model = ResNetCSWinHybrid(num_classes=num_classes, resnet_pretrained=True)

        # loading pretrained weights, seems like they are not useful here

        # timm_model = timm.create_model('CSWin_64_12211_tiny_224', pretrained=True)
        # pretrained_weights = timm_model.state_dict()

        # weights_to_load = {}
        # for key, value in pretrained_weights.items():
        #     if key.startswith('stage1_conv_embed.') or key.startswith('head.'):
        #         continue  # skip this, we have HPEM now
        #     weights_to_load[key] = value

        # model.load_state_dict(weights_to_load, strict=False)
        # print("Successfully loaded pretrained backbone weights for CSWin")

        # load_result = model.load_state_dict(weights_to_load, strict=False)
        # print(f"Weight loading result (missing keys): {load_result.missing_keys}")
        # print(f"Weight loading result (unexpected keys): {load_result.unexpected_keys}")

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_resnet = sum(p.numel() for p in model.resnet_stem.parameters() if p.requires_grad)
        num_cswin_block3 = sum(p.numel() for p in model.stage3.parameters() if p.requires_grad)
        num_cswin_block4 = sum(p.numel() for p in model.stage4.parameters() if p.requires_grad)
        num_head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"Total Trainable params: {num_trainable:,} / {num_total:,}")
        print(f"ResNet50 params: {num_resnet:,}")
        print(f"CSWin Stage 3 Trainable params: {num_cswin_block3:,}")
        print(f"CSWin Stage 4 Trainable params: {num_cswin_block4:,}")
        print(f"Head Classifier Trainable params: {num_head:,}")

        return model


# Deep fine-tuning function
def fine_tune_model(model_name, trainloader, testloader, num_classes, epochs=15):
    model = load_model_for_deep_finetuning(model_name, num_classes)

    model = model.to(device)

    # add class weights
    # 0: 1710
    # 1: 2795
    count_0 = 1710.0
    count_1 = 2795.0
    total = count_0 + count_1

    # assign more weight to healthy cases due to imbalance
    weight_for_0 = total / (2.0 * count_0)  # around 1.317
    weight_for_1 = total / (2.0 * count_1)  # around 0.806

    class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)

    criterion = nn.CrossEntropyLoss(class_weights).to(device)

    if model_name == 'hybrid_cswin':
        print("Using AdamW with differential LR for hybrid model")

        backbone_params = []
        new_params = []

        for name, param in model.named_parameters():
            if name.startswith('resnet_stem.'):
                backbone_params.append(param)
            else:
                # All other parts are new (bridge, stage3, stage4, head)
                new_params.append(param)

        # Give both groups a high LR since we are training from scratch/overwriting
        optimizer = optim.AdamW(
            [
                {'params': backbone_params, 'lr': 1e-5}, #1e-4
                {'params': new_params, 'lr': 1e-4}
            ],
            lr=1e-4,
            weight_decay=0.01
        )

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
            if i % 20 == 19:
                avg_loss = running_loss / 20
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
            transforms.Resize((224, 224)), # thus also doing it here is necessary
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.7553, 0.3109, 0.1059], [0.1774, 0.1262, 0.0863])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)), # thus also doing it here is necessary
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.7553, 0.3109, 0.1059], [0.1774, 0.1262, 0.0863])
        ]),
    }

    data_dir = 'PlantVillage_1_2019train_2022test'

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


# Train 2019 -> Test 2022 first run (100 epochs), only resnet pretrained
# Stats Accuracy: 0.8160 | Precision(+): 0.9826 | Recall(+): 0.7404 | F1(+): 0.8444 | F1-macro: 0.8096

# Train 2019 -> Test 2022 second run, only resnet pretrained, added class weights, modified lr for backbone from 1e-4 to 1e-5 (100 epochs)
# Stats Accuracy: 0.7868 | Precision(+): 0.9734 | Recall(+): 0.7033 | F1(+): 0.8166 | F1-macro: 0.7811

# Train 2019 -> Test 2022 3rd run (100 epochs), both resnet stages and cswin blocks pretrained, cosine annealing for optimizer, 1e-4
# Stats Accuracy: 0.8198 | Precision(+): 0.9815 | Recall(+): 0.7469 | F1(+): 0.8483 | F1-macro: 0.8132

# Train 2019 -> Test 2022 4th run (100 epochs), both resnet stages and cswin blocks pretrained, focal loss with cosine annealing, 1e-4
# Stats Accuracy: 0.8508 | Precision(+): 0.9809 | Recall(+): 0.7944 | F1(+): 0.8778 | F1-macro: 0.8431
