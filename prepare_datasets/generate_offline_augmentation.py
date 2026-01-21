"""
Offline Data Augmentation Pipeline for Plant Disease Classification

This script generates pre-augmented copies of training images to expand the dataset
from ~3-4K images to 10K+ images for improved transformer model performance.

Usage:
    python generate_offline_augmentation.py --source PlantVillage_1_2019train_2022test --multiplier 2.33
    python generate_offline_augmentation.py --source PlantVillage_2_2022train_2019test --multiplier 3.17
"""

import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


class AugmentationVariants:
    """Defines augmentation transform variants for offline generation."""

    def __init__(self, normalization_mean, normalization_std):
        self.norm_mean = normalization_mean
        self.norm_std = normalization_std

    def get_variant(self, variant_id: str) -> transforms.Compose:
        """Get augmentation transforms for a specific variant."""

        variants = {
            'v1': self._variant_geometric_light(),
            'v2': self._variant_geometric_moderate(),
            'v3': self._variant_photometric_brightness(),
            'v4': self._variant_photometric_contrast(),
            'v5': self._variant_mixed(),
            'v6': self._variant_advanced_blur(),
            'v7': self._variant_perspective(),
            'v8': self._variant_occlusion(),
        }

        return variants.get(variant_id, self._variant_geometric_light())

    def _variant_geometric_light(self) -> transforms.Compose:
        """Variant 1: Geometric Light - RandomHorizontalFlip + RandomRotation(10°)"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=10),
        ])

    def _variant_geometric_moderate(self) -> transforms.Compose:
        """Variant 2: Geometric Moderate - RandomRotation(20°) + RandomAffine + RandomVerticalFlip"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def _variant_photometric_brightness(self) -> transforms.Compose:
        """Variant 3: Photometric Brightness - ColorJitter brightness-focused"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    def _variant_photometric_contrast(self) -> transforms.Compose:
        """Variant 4: Photometric Contrast - ColorJitter contrast/saturation + GaussianBlur"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ])

    def _variant_mixed(self) -> transforms.Compose:
        """Variant 5: Mixed - Rotation + Affine + ColorJitter"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ])

    def _variant_advanced_blur(self) -> transforms.Compose:
        """Variant 6: Advanced Blur - Heavy rotation + GaussianBlur"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=25),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    def _variant_perspective(self) -> transforms.Compose:
        """Variant 7: Perspective - RandomPerspective + RandomAffine + shear"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        ])

    def _variant_occlusion(self) -> transforms.Compose:
        """Variant 8: Occlusion - ColorJitter + RandomErasing (applied post-tensor)"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def get_variant_description(self, variant_id: str) -> Dict:
        """Get metadata description for a variant."""
        descriptions = {
            'v1': {'name': 'Geometric Light', 'transforms': ['RandomHorizontalFlip(p=1.0)', 'RandomRotation(10)']},
            'v2': {'name': 'Geometric Moderate', 'transforms': ['RandomRotation(20)', 'RandomAffine', 'RandomVerticalFlip']},
            'v3': {'name': 'Photometric Brightness', 'transforms': ['ColorJitter(brightness=0.4)', 'RandomHorizontalFlip']},
            'v4': {'name': 'Photometric Contrast', 'transforms': ['ColorJitter(contrast=0.4, saturation=0.4)', 'GaussianBlur']},
            'v5': {'name': 'Mixed', 'transforms': ['RandomRotation(15)', 'RandomAffine', 'ColorJitter']},
            'v6': {'name': 'Advanced Blur', 'transforms': ['RandomRotation(25)', 'GaussianBlur', 'RandomHorizontalFlip']},
            'v7': {'name': 'Perspective', 'transforms': ['RandomPerspective', 'RandomAffine', 'Shear']},
            'v8': {'name': 'Occlusion', 'transforms': ['ColorJitter', 'RandomHorizontalFlip', 'RandomErasing(post-save)']},
        }
        return descriptions.get(variant_id, {'name': 'Unknown', 'transforms': []})


class ClassBalancer:
    """Calculate target counts for balanced augmentation."""

    @staticmethod
    def calculate_target_counts(original_counts: Dict[str, int], target_total: int) -> Dict[str, int]:
        """
        Calculate target image counts per class to maintain original distribution.

        Args:
            original_counts: Dict mapping class names to original image counts
            target_total: Total target number of images after augmentation

        Returns:
            Dict mapping class names to target image counts
        """
        total_original = sum(original_counts.values())
        target_counts = {}

        for class_name, count in original_counts.items():
            ratio = count / total_original
            target_counts[class_name] = int(target_total * ratio)

        return target_counts

    @staticmethod
    def calculate_variants_per_image(original_count: int, target_count: int) -> Tuple[int, float]:
        """
        Calculate how many variants to generate per image.

        Returns:
            (base_variants, probability_for_extra): Each image gets base_variants,
            plus one more with given probability
        """
        multiplier = target_count / original_count
        base_variants = int(multiplier)
        probability_for_extra = multiplier - base_variants

        return base_variants, probability_for_extra


class AugmentationPipeline:
    """Main pipeline for generating offline augmented dataset."""

    def __init__(self, source_dir: Path, output_dir: Path, target_multiplier: float,
                 normalization_mean: List[float], normalization_std: List[float]):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.target_multiplier = target_multiplier
        self.norm_mean = normalization_mean
        self.norm_std = normalization_std

        self.variants = AugmentationVariants(normalization_mean, normalization_std)
        self.metadata = {
            'dataset': output_dir.name,
            'generation_date': datetime.now().isoformat(),
            'source_dir': str(source_dir),
            'multiplier': target_multiplier,
            'normalization': {
                'mean': normalization_mean,
                'std': normalization_std
            },
            'augmentation_variants': [],
            'original_count': {},
            'augmented_count': {},
            'images': []
        }

    def generate_augmented_dataset(self):
        """Main execution method to generate augmented dataset."""
        print(f"\n{'='*60}")
        print(f"Offline Data Augmentation Pipeline")
        print(f"{'='*60}")
        print(f"Source: {self.source_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Target Multiplier: {self.target_multiplier}x")
        print(f"{'='*60}\n")

        # Step 1: Analyze source dataset
        print("Step 1: Analyzing source dataset...")
        original_counts = self._analyze_source_dataset()
        total_original = sum(original_counts.values())
        target_total = int(total_original * self.target_multiplier)

        print(f"  Original total: {total_original} images")
        print(f"  Target total: {target_total} images")
        for class_name, count in original_counts.items():
            print(f"    - {class_name}: {count} images")

        # Step 2: Calculate target counts per class
        print("\nStep 2: Calculating target distribution...")
        target_counts = ClassBalancer.calculate_target_counts(original_counts, target_total)
        for class_name, count in target_counts.items():
            print(f"    - {class_name}: {count} images (target)")

        # Step 3: Create output directory structure
        print("\nStep 3: Creating output directories...")
        self._create_output_directories()

        # Step 4: Copy test set without augmentation
        print("\nStep 4: Copying test set (no augmentation)...")
        self._copy_test_set()

        # Step 5: Generate augmented training images
        print("\nStep 5: Generating augmented training images...")
        self._generate_augmented_training(original_counts, target_counts)

        # Step 6: Validate generated dataset
        print("\nStep 6: Validating generated dataset...")
        self._validate_generated_dataset(target_counts)

        # Step 7: Save metadata
        print("\nStep 7: Saving metadata...")
        self._save_metadata()

        print(f"\n{'='*60}")
        print(f"Augmentation completed successfully!")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

    def _analyze_source_dataset(self) -> Dict[str, int]:
        """Analyze source dataset and count images per class."""
        train_dir = self.source_dir / 'train'
        class_counts = {}

        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                images = self._list_images(class_dir)
                class_counts[class_dir.name] = len(images)
                self.metadata['original_count'][class_dir.name] = len(images)

        return class_counts

    def _create_output_directories(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_dir = self.output_dir / 'train'
        test_dir = self.output_dir / 'test'

        for split_dir in [train_dir, test_dir]:
            split_dir.mkdir(exist_ok=True)
            for class_name in self.metadata['original_count'].keys():
                (split_dir / class_name).mkdir(exist_ok=True)

    def _copy_test_set(self):
        """Copy test set without any augmentation."""
        source_test = self.source_dir / 'test'
        output_test = self.output_dir / 'test'

        for class_dir in source_test.iterdir():
            if class_dir.is_dir():
                source_class_dir = source_test / class_dir.name
                output_class_dir = output_test / class_dir.name

                images = self._list_images(source_class_dir)
                for img_path in tqdm(images, desc=f"  Copying {class_dir.name}"):
                    shutil.copy2(img_path, output_class_dir / img_path.name)

    def _generate_augmented_training(self, original_counts: Dict[str, int],
                                    target_counts: Dict[str, int]):
        """Generate augmented training images for all classes."""
        train_dir = self.source_dir / 'train'
        output_train = self.output_dir / 'train'

        # Select variants based on multiplier
        selected_variants = self._select_variants_for_multiplier(self.target_multiplier)
        print(f"  Selected variants: {', '.join(selected_variants)}")

        # Add variant metadata
        for variant_id in selected_variants:
            variant_info = self.variants.get_variant_description(variant_id)
            variant_info['id'] = variant_id
            self.metadata['augmentation_variants'].append(variant_info)

        for class_name, target_count in target_counts.items():
            original_count = original_counts[class_name]
            print(f"\n  Processing class: {class_name}")
            print(f"    Original: {original_count} → Target: {target_count}")

            source_class_dir = train_dir / class_name
            output_class_dir = output_train / class_name

            images = self._list_images(source_class_dir)

            # Calculate how many variants per image
            variants_needed = target_count - original_count  # Exclude originals
            base_variants, extra_prob = ClassBalancer.calculate_variants_per_image(
                original_count, target_count
            )

            print(f"    Strategy: {base_variants - 1} variants per image + {extra_prob:.2%} get 1 more")

            generated_count = 0

            for img_path in tqdm(images, desc=f"    Augmenting {class_name}"):
                # Extract image ID from filename
                img_id = img_path.stem

                # Copy original image
                output_original = output_class_dir / f"{img_id}_original{img_path.suffix}"
                shutil.copy2(img_path, output_original)
                generated_count += 1

                # Determine how many variants to generate for this image
                num_variants = base_variants - 1  # -1 because we already copied original
                if np.random.random() < extra_prob:
                    num_variants += 1

                # Generate variants
                num_variants = min(num_variants, len(selected_variants))  # Don't exceed available variants

                for i, variant_id in enumerate(selected_variants[:num_variants]):
                    augmented_img = self._apply_augmentation(img_path, variant_id)
                    output_path = output_class_dir / f"{img_id}_{variant_id}{img_path.suffix}"
                    augmented_img.save(output_path, quality=95)
                    generated_count += 1

                    # Track in metadata (sample, not all to keep JSON manageable)
                    if np.random.random() < 0.01:  # Sample 1%
                        self.metadata['images'].append({
                            'original_path': str(img_path.relative_to(self.source_dir)),
                            'augmented_path': str(output_path.relative_to(self.output_dir)),
                            'variant_id': variant_id,
                            'class': class_name
                        })

            self.metadata['augmented_count'][class_name] = generated_count
            print(f"    Generated: {generated_count} images")

    def _select_variants_for_multiplier(self, multiplier: float) -> List[str]:
        """Select which augmentation variants to use based on multiplier."""
        if multiplier <= 1.5:
            return ['v1']
        elif multiplier <= 2.0:
            return ['v1', 'v2']
        elif multiplier <= 2.5:
            return ['v1', 'v2', 'v3']
        elif multiplier <= 3.0:
            return ['v1', 'v2', 'v3', 'v5']
        else:
            return ['v1', 'v2', 'v3', 'v4', 'v5']

    def _apply_augmentation(self, img_path: Path, variant_id: str) -> Image.Image:
        """Apply augmentation variant to an image."""
        img = Image.open(img_path).convert('RGB')
        transform = self.variants.get_variant(variant_id)

        # Set seed for reproducibility (based on image path and variant)
        seed = hash(str(img_path) + variant_id) % (2**32)
        torch.manual_seed(seed)
        np.random.seed(seed)

        augmented = transform(img)
        return augmented

    def _validate_generated_dataset(self, target_counts: Dict[str, int]):
        """Validate the generated dataset."""
        output_train = self.output_dir / 'train'

        print("  Validation results:")
        all_valid = True

        for class_name, target_count in target_counts.items():
            class_dir = output_train / class_name
            images = self._list_images(class_dir)
            actual_count = len(images)

            # Allow 1% tolerance
            tolerance = max(10, int(target_count * 0.01))
            is_valid = abs(actual_count - target_count) <= tolerance

            status = "✓" if is_valid else "✗"
            print(f"    {status} {class_name}: {actual_count} images (target: {target_count})")

            if not is_valid:
                all_valid = False

        if all_valid:
            print("  All classes validated successfully!")
        else:
            print("  WARNING: Some classes have count mismatches")

    def _save_metadata(self):
        """Save metadata JSON file."""
        metadata_path = self.output_dir / 'augmentation_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"  Metadata saved to: {metadata_path}")

    @staticmethod
    def _list_images(folder: Path) -> List[Path]:
        """List all image files in a folder."""
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        return sorted([p for p in folder.iterdir()
                      if p.suffix.lower() in exts and p.is_file()])


def main():
    parser = argparse.ArgumentParser(
        description='Generate offline augmented dataset for plant disease classification'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source dataset name (e.g., PlantVillage_1_2019train_2022test)'
    )
    parser.add_argument(
        '--multiplier',
        type=float,
        required=True,
        help='Target multiplier for dataset size (e.g., 2.33 for 2019 data, 3.17 for 2022 data)'
    )
    parser.add_argument(
        '--norm-mean',
        type=float,
        nargs=3,
        default=[0.7553, 0.3109, 0.1059],
        help='Normalization mean values (default: 2019 train values)'
    )
    parser.add_argument(
        '--norm-std',
        type=float,
        nargs=3,
        default=[0.1774, 0.1262, 0.0863],
        help='Normalization std values (default: 2019 train values)'
    )

    args = parser.parse_args()

    # Construct paths
    base_dir = Path(__file__).parent.parent / 'DeepLearning_PlantDiseases-master' / 'Scripts'
    source_dir = base_dir / args.source
    output_dir = base_dir / f"{args.source}_augmented"

    # Validate source directory exists
    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}")
        return

    # Create pipeline and run
    pipeline = AugmentationPipeline(
        source_dir=source_dir,
        output_dir=output_dir,
        target_multiplier=args.multiplier,
        normalization_mean=args.norm_mean,
        normalization_std=args.norm_std
    )

    pipeline.generate_augmented_dataset()


if __name__ == '__main__':
    main()
