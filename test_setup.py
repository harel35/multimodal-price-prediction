"""
Quick test script to verify dataset and dataloader setup.
This script performs basic checks without loading the full dataset.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src import config


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.data import MultimodalPriceDataset, create_dataloaders
        from src.utils import TextPreprocessor, PriceTransformer
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_paths():
    """Test that required paths exist."""
    print("\nTesting paths...")
    
    checks = {
        "CSV file": config.CSV_PATH,
        "Images directory": config.IMAGES_DIR,
        "Project root": config.PROJECT_ROOT,
    }
    
    all_exist = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False
    
    return all_exist


def test_config():
    """Test configuration values."""
    print("\nTesting configuration...")
    
    configs = {
        "Batch size": config.BATCH_SIZE,
        "Image size": config.IMAGE_SIZE,
        "Train split": config.TRAIN_SPLIT,
        "Val split": config.VAL_SPLIT,
        "Test split": config.TEST_SPLIT,
        "Random seed": config.RANDOM_SEED,
    }
    
    for name, value in configs.items():
        print(f"  ✓ {name}: {value}")
    
    # Check splits sum to 1
    total = config.TRAIN_SPLIT + config.VAL_SPLIT + config.TEST_SPLIT
    if abs(total - 1.0) < 1e-6:
        print(f"  ✓ Splits sum to 1.0")
        return True
    else:
        print(f"  ✗ Splits sum to {total}, should be 1.0")
        return False


def test_dataset_creation():
    """Test creating a dataset instance."""
    print("\nTesting dataset creation...")
    
    try:
        from src.data import MultimodalPriceDataset
        from src.data.dataloader import get_transforms
        
        transform = get_transforms('test', config.IMAGE_SIZE)
        
        dataset = MultimodalPriceDataset(
            csv_path=str(config.CSV_PATH),
            images_dir=str(config.IMAGES_DIR),
            transform=transform,
            mode='test'
        )
        
        print(f"  ✓ Dataset created with {len(dataset)} samples")
        
        # Get statistics
        stats = dataset.get_statistics()
        print(f"  ✓ Price range: ${stats['price_min']:.2f} - ${stats['price_max']:.2f}")
        print(f"  ✓ Price mean: ${stats['price_mean']:.2f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Dataset creation failed: {e}")
        return False


def test_text_preprocessor():
    """Test text preprocessor."""
    print("\nTesting text preprocessor...")
    
    try:
        from src.utils import TextPreprocessor
        
        preprocessor = TextPreprocessor(
            model_name='bert-base-uncased',
            max_length=128
        )
        
        test_text = "Item Name: Test Product 16oz (Pack of 4)"
        encoded = preprocessor.encode_text(test_text)
        
        print(f"  ✓ Text preprocessor initialized")
        print(f"  ✓ Encoded shape: {encoded['input_ids'].shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Text preprocessor failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Dataset & DataLoader Setup Test")
    print("=" * 80)
    
    tests = [
        test_imports,
        test_paths,
        test_config,
        test_dataset_creation,
        test_text_preprocessor,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Setup is ready to use.")
        print("\nNext: Run 'python example_usage.py' for a complete demo")
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
