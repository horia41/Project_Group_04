# Run Knowledge Distillation Experiments with different configurations

import os
import sys
import torch
from train_knowledge_distillation import train_with_distillation, load_data
from distillation_config import (
    EFFICIENTNET_VARIANTS,
    DISTILLATION_CONFIGS,
    LR_CONFIGS,
    TEACHER_MODELS,
    DATA_CONFIGS
)


def run_single_experiment(
    experiment_name,
    teacher_checkpoint,
    student_name,
    trainloader,
    testloader,
    distillation_config,
    lr_config
):
    """
    Run a single distillation experiment with given configuration
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*80)
    print(f"Student Model: {student_name}")
    print(f"Teacher: {teacher_checkpoint}")
    print(f"Temperature: {distillation_config['temperature']}")
    print(f"Alpha: {distillation_config['alpha']}")
    print(f"Epochs: {distillation_config['epochs']}")
    print(f"Backbone LR: {lr_config['backbone_lr']}")
    print(f"Head LR: {lr_config['head_lr']}")
    print("="*80 + "\n")

    try:
        losses, trained_student = train_with_distillation(
            teacher_checkpoint=teacher_checkpoint,
            student_name=student_name,
            trainloader=trainloader,
            testloader=testloader,
            num_classes=2,
            epochs=distillation_config['epochs'],
            temperature=distillation_config['temperature'],
            alpha=distillation_config['alpha']
        )

        # Save experiment-specific model
        save_folder = 'cswin_fpn_hybrid/model_saves/experiments'
        os.makedirs(save_folder, exist_ok=True)
        model_save_path = os.path.join(save_folder, f'{experiment_name}_{student_name}.pth')
        torch.save(trained_student.state_dict(), model_save_path)
        print(f"\nExperiment model saved: {model_save_path}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Experiment {experiment_name} failed: {str(e)}")
        return False


def run_student_comparison():
    """
    Compare different EfficientNet variants as students
    """
    print("\n" + "="*80)
    print("EXPERIMENT SET: Student Model Comparison")
    print("="*80)

    trainloader, testloader = load_data()
    teacher_checkpoint = TEACHER_MODELS['default']
    distillation_config = DISTILLATION_CONFIGS['default']
    lr_config = LR_CONFIGS['default']

    for student_name in EFFICIENTNET_VARIANTS[:3]:  # Test first 3 variants
        experiment_name = f"student_comparison_{student_name}"
        run_single_experiment(
            experiment_name=experiment_name,
            teacher_checkpoint=teacher_checkpoint,
            student_name=student_name,
            trainloader=trainloader,
            testloader=testloader,
            distillation_config=distillation_config,
            lr_config=lr_config
        )


def run_temperature_ablation():
    """
    Ablation study on temperature parameter
    """
    print("\n" + "="*80)
    print("EXPERIMENT SET: Temperature Ablation Study")
    print("="*80)

    trainloader, testloader = load_data()
    teacher_checkpoint = TEACHER_MODELS['default']
    student_name = 'efficientnet_b0'
    lr_config = LR_CONFIGS['default']

    temperatures = [2.0, 4.0, 6.0, 8.0]
    alpha = 0.7

    for temp in temperatures:
        experiment_name = f"temp_ablation_T{temp}"
        distillation_config = {
            'temperature': temp,
            'alpha': alpha,
            'epochs': 50,  # Shorter for ablation
            'batch_size': 128,
        }

        run_single_experiment(
            experiment_name=experiment_name,
            teacher_checkpoint=teacher_checkpoint,
            student_name=student_name,
            trainloader=trainloader,
            testloader=testloader,
            distillation_config=distillation_config,
            lr_config=lr_config
        )


def run_alpha_ablation():
    """
    Ablation study on alpha parameter (distillation vs hard label weight)
    """
    print("\n" + "="*80)
    print("EXPERIMENT SET: Alpha Ablation Study")
    print("="*80)

    trainloader, testloader = load_data()
    teacher_checkpoint = TEACHER_MODELS['default']
    student_name = 'efficientnet_b0'
    lr_config = LR_CONFIGS['default']

    alphas = [0.3, 0.5, 0.7, 0.9]
    temperature = 4.0

    for alpha in alphas:
        experiment_name = f"alpha_ablation_A{alpha}"
        distillation_config = {
            'temperature': temperature,
            'alpha': alpha,
            'epochs': 50,  # Shorter for ablation
            'batch_size': 128,
        }

        run_single_experiment(
            experiment_name=experiment_name,
            teacher_checkpoint=teacher_checkpoint,
            student_name=student_name,
            trainloader=trainloader,
            testloader=testloader,
            distillation_config=distillation_config,
            lr_config=lr_config
        )


def run_preset_configs():
    """
    Run all preset configurations from config file
    """
    print("\n" + "="*80)
    print("EXPERIMENT SET: Preset Configurations")
    print("="*80)

    trainloader, testloader = load_data()
    teacher_checkpoint = TEACHER_MODELS['default']
    student_name = 'efficientnet_b0'
    lr_config = LR_CONFIGS['default']

    for config_name, distillation_config in DISTILLATION_CONFIGS.items():
        experiment_name = f"preset_{config_name}"

        run_single_experiment(
            experiment_name=experiment_name,
            teacher_checkpoint=teacher_checkpoint,
            student_name=student_name,
            trainloader=trainloader,
            testloader=testloader,
            distillation_config=distillation_config,
            lr_config=lr_config
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Knowledge Distillation Experiments')
    parser.add_argument(
        '--experiment',
        type=str,
        default='student_comparison',
        choices=['student_comparison', 'temperature', 'alpha', 'presets', 'all'],
        help='Which experiment set to run'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Knowledge Distillation Experiment Suite")
    print("Teacher: ResNetCSWinHybrid (new_model)")
    print("Student: EfficientNet variants")
    print("="*80)

    if args.experiment == 'student_comparison':
        run_student_comparison()
    elif args.experiment == 'temperature':
        run_temperature_ablation()
    elif args.experiment == 'alpha':
        run_alpha_ablation()
    elif args.experiment == 'presets':
        run_preset_configs()
    elif args.experiment == 'all':
        run_student_comparison()
        run_temperature_ablation()
        run_alpha_ablation()
        run_preset_configs()

    print("\n" + "="*80)
    print("All Experiments Complete!")
    print("="*80)
