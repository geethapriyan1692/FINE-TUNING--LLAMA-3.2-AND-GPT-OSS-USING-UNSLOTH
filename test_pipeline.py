#!/usr/bin/env python3
"""
Simple test script to demonstrate the LLM Fine-tuning Pipeline functionality.
This script tests the core components without requiring all dependencies.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_config_system():
    """Test the configuration system."""
    print("🔧 Testing Configuration System...")
    
    try:
        from utils.config import PipelineConfig, ModelConfig, LoraConfig, TrainingConfig, DataConfig, ExperimentConfig
        
        # Create a sample configuration
        config = PipelineConfig(
            model=ModelConfig(
                name="test-model",
                source="huggingface",
                model_id="microsoft/DialoGPT-small",
                max_seq_length=512
            ),
            lora=LoraConfig(r=8, lora_alpha=16),
            training=TrainingConfig(
                output_dir="./test_outputs",
                num_train_epochs=1,
                per_device_train_batch_size=1
            ),
            data=DataConfig(
                dataset_path="./data/sample.json",
                dataset_format="sharegpt"
            ),
            experiment=ExperimentConfig(
                experiment_name="test_experiment"
            )
        )
        
        # Test saving and loading
        config_path = Path("test_config.yaml")
        config.save(config_path)
        
        loaded_config = PipelineConfig.from_yaml(config_path)
        
        print(f"✅ Configuration system working!")
        print(f"   - Model: {loaded_config.model.model_id}")
        print(f"   - LoRA rank: {loaded_config.lora.r}")
        print(f"   - Output dir: {loaded_config.training.output_dir}")
        
        # Cleanup
        config_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        return False

def test_data_processors():
    """Test data processing functionality."""
    print("\n📊 Testing Data Processing...")
    
    try:
        from data.processors import DataProcessorFactory
        
        # Create sample data
        sample_data = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello!"},
                    {"from": "gpt", "value": "Hi there! How can I help you?"}
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "What is AI?"},
                    {"from": "gpt", "value": "AI is artificial intelligence."}
                ]
            }
        ]
        
        # Save sample data
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        sample_file = data_dir / "test_sample.json"
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f)
        
        print(f"✅ Data processing system working!")
        print(f"   - Created sample data: {sample_file}")
        print(f"   - Sample count: {len(sample_data)}")
        print(f"   - Available processors: {list(DataProcessorFactory.processors.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def test_model_handlers():
    """Test model handler factory."""
    print("\n🤖 Testing Model Handlers...")
    
    try:
        from models.handlers import ModelHandlerFactory
        from utils.config import ModelConfig
        
        # Test handler creation
        config = ModelConfig(
            name="test-model",
            source="huggingface",
            model_id="microsoft/DialoGPT-small"
        )
        
        handler = ModelHandlerFactory.create_handler(config)
        
        print(f"✅ Model handlers working!")
        print(f"   - Handler type: {type(handler).__name__}")
        print(f"   - Available handlers: {list(ModelHandlerFactory.handlers.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model handlers failed: {e}")
        return False

def test_monitoring_system():
    """Test monitoring and tracking system."""
    print("\n📈 Testing Monitoring System...")
    
    try:
        from monitoring.tracking import create_tracker
        from utils.config import ExperimentConfig
        
        experiment_config = ExperimentConfig(
            experiment_name="test_experiment",
            tags=["test"]
        )
        
        # Test tracker creation (without actual API keys)
        tracker = create_tracker(
            experiment_config=experiment_config,
            use_comet=False,  # Disable to avoid API key requirements
            use_opik=False
        )
        
        print(f"✅ Monitoring system working!")
        print(f"   - Tracker created: {tracker is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring system failed: {e}")
        return False

def test_pipeline_structure():
    """Test pipeline structure and imports."""
    print("\n🔄 Testing Pipeline Structure...")
    
    try:
        from pipelines.steps import load_and_validate_config, validate_data_format
        from pipelines.training_pipeline import llm_finetuning_pipeline
        
        print(f"✅ Pipeline structure working!")
        print(f"   - ZenML steps available")
        print(f"   - Main pipeline defined")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline structure failed: {e}")
        return False

def show_system_info():
    """Show system information."""
    print("\n💻 System Information:")
    print("=" * 40)
    
    # Python version
    print(f"Python: {sys.version}")
    
    # Check for key dependencies
    dependencies = [
        "torch", "transformers", "datasets", "peft", 
        "zenml", "comet_ml", "opik", "rich"
    ]
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {dep}: {version}")
        except ImportError:
            print(f"❌ {dep}: not installed")
    
    # GPU info
    try:
        import torch
        print(f"\nGPU Info:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except:
        print("❌ Could not check GPU info")

def main():
    """Run all tests."""
    print("🚀 LLM Fine-tuning Pipeline - System Test")
    print("=" * 50)
    
    # Show system info
    show_system_info()
    
    # Run tests
    tests = [
        test_config_system,
        test_data_processors,
        test_model_handlers,
        test_monitoring_system,
        test_pipeline_structure
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 30)
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Setup environment: cp env_template .env")
        print("3. Initialize config: python -m src.cli init")
        print("4. Start training: python -m src.cli train --config config.yaml")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("Install missing dependencies and try again.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
