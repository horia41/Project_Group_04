# Configuration file for Knowledge Distillation experiments

# Available EfficientNet variants (smaller to larger)
EFFICIENTNET_VARIANTS = [
    'efficientnet_b0',  # 5.3M params, baseline
    'efficientnet_b1',  # 7.8M params
    'efficientnet_b2',  # 9.2M params
    'efficientnet_b3',  # 12M params
    'efficientnet_b4',  # 19M params
]

# Distillation hyperparameters
DISTILLATION_CONFIGS = {
    'default': {
        'temperature': 4.0,
        'alpha': 0.7,  # 70% distillation loss, 30% hard label loss
        'epochs': 100,
        'batch_size': 128,
    },
    'soft_distillation': {
        'temperature': 6.0,
        'alpha': 0.9,  # More emphasis on teacher knowledge
        'epochs': 100,
        'batch_size': 128,
    },
    'hard_distillation': {
        'temperature': 2.0,
        'alpha': 0.5,  # More emphasis on ground truth
        'epochs': 100,
        'batch_size': 128,
    },
    'balanced': {
        'temperature': 4.0,
        'alpha': 0.5,  # Equal weight for distillation and hard labels
        'epochs': 100,
        'batch_size': 128,
    },
    'high_temp': {
        'temperature': 8.0,
        'alpha': 0.8,
        'epochs': 100,
        'batch_size': 128,
    },
}

# Learning rate configurations
LR_CONFIGS = {
    'default': {
        'backbone_lr': 1e-4,
        'head_lr': 5e-4,
        'weight_decay': 0.05,
        'warmup_epochs': 10,
    },
    'aggressive': {
        'backbone_lr': 5e-4,
        'head_lr': 1e-3,
        'weight_decay': 0.01,
        'warmup_epochs': 5,
    },
    'conservative': {
        'backbone_lr': 5e-5,
        'head_lr': 2e-4,
        'weight_decay': 0.1,
        'warmup_epochs': 15,
    },
}

# Teacher model paths
TEACHER_MODELS = {
    'default': 'cswin_fpn_hybrid/model_saves/threshold_0.50_hybrid_Tr2019_Te2022.pth',
    # Add more teacher checkpoints here if you have them
}

# Data configurations
DATA_CONFIGS = {
    'original': {
        'use_augmented_dataset': False,
        'online_augmentation': True,
    },
    'augmented': {
        'use_augmented_dataset': True,
        'online_augmentation': True,
    },
    'augmented_only': {
        'use_augmented_dataset': True,
        'online_augmentation': False,
    },
}
