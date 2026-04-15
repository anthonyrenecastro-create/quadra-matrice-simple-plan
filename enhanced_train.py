#!/usr/bin/env python3
"""
CognitionSim - Enhanced Training Configuration
Trains longer, uses multiple datasets, and scales up field size for better results
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def enhanced_training():
    """Run enhanced training with scaled parameters"""
    from train_atlantean_hrm import CognitionSimTrainer
    
    print("\n" + "="*80)
    print("🚀 QUADRA MATRIX - ENHANCED TRAINING SESSION")
    print("="*80)
    print("\n📈 Improvements:")
    print("  • Training longer: 200 batches (vs 100)")
    print("  • Larger field size: 200 (vs 100) - more complex patterns")
    print("  • Bigger batches: 15 samples (vs 10)")
    print("  • Multiple dataset support ready")
    print("\n⏱️  Estimated time: ~20-25 minutes")
    print("🎯 Expected results: Even higher speedup and better convergence!\n")
    print("="*80 + "\n")
    
    # Phase 1: Train on WikiText-2 with enhanced parameters
    print("\n📚 PHASE 1: Enhanced WikiText-2 Training")
    print("-"*80)
    print("Field size: 200 | Batches: 200 | Batch size: 15\n")
    
    trainer = CognitionSimTrainer(field_size=200, device='cpu')
    
    metrics = await trainer.train(
        dataset_name='wikitext',
        dataset_config='wikitext-2-raw-v1',
        num_batches=200,
        batch_size=15
    )
    
    print("\n✅ Phase 1 Complete!")
    print(f"   Peak Speedup: {max(metrics['speedup_factors']):.2f}x")
    print(f"   Final Reward: {metrics['rewards'][-1]:.4f}")
    print(f"   Final Loss: {metrics['rewards'][-1]:.4f}")
    
    # Save enhanced model
    trainer.oscillator.save_weights("atlantean_hrm_enhanced.pth")
    print("\n💾 Enhanced model saved to: atlantean_hrm_enhanced.pth")
    
    return metrics


async def multi_dataset_training():
    """Train on multiple datasets for robustness"""
    from train_atlantean_hrm import CognitionSimTrainer
    
    print("\n" + "="*80)
    print("🌐 MULTI-DATASET TRAINING")
    print("="*80 + "\n")
    
    datasets = [
        ('wikitext', 'wikitext-2-raw-v1', 'WikiText-2'),
        ('tiny_shakespeare', None, 'Shakespeare'),
    ]
    
    all_results = {}
    
    for idx, (dataset_name, config, display_name) in enumerate(datasets, 1):
        print(f"\n📖 Dataset {idx}/{len(datasets)}: {display_name}")
        print("-"*80)
        
        trainer = CognitionSimTrainer(field_size=150, device='cpu')
        
        try:
            if config:
                metrics = await trainer.train(
                    dataset_name=dataset_name,
                    dataset_config=config,
                    num_batches=100,
                    batch_size=10
                )
            else:
                metrics = await trainer.train(
                    dataset_name=dataset_name,
                    num_batches=100,
                    batch_size=10
                )
            
            all_results[display_name] = metrics
            
            print(f"\n✅ {display_name} Complete!")
            print(f"   Peak Speedup: {max(metrics['speedup_factors']):.2f}x")
            print(f"   Final Reward: {metrics['rewards'][-1]:.4f}")
            
            # Save model for this dataset
            trainer.oscillator.save_weights(f"atlantean_hrm_{dataset_name}.pth")
            print(f"   💾 Model saved: atlantean_hrm_{dataset_name}.pth")
            
        except Exception as e:
            print(f"\n⚠️  {display_name} failed: {e}")
            print(f"   Continuing with next dataset...")
            continue
    
    print("\n" + "="*80)
    print("🎯 MULTI-DATASET TRAINING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    for name, metrics in all_results.items():
        peak_speedup = max(metrics['speedup_factors'])
        final_reward = metrics['rewards'][-1]
        print(f"  {name:20s} | Speedup: {peak_speedup:5.2f}x | Reward: {final_reward:7.4f}")
    
    return all_results


async def scaled_field_comparison():
    """Compare different field sizes to show scaling benefits"""
    from train_atlantean_hrm import CognitionSimTrainer
    
    print("\n" + "="*80)
    print("🔬 FIELD SIZE SCALING EXPERIMENT")
    print("="*80)
    print("\nTesting how field size affects pattern complexity handling\n")
    
    field_sizes = [100, 150, 200, 250]
    results = {}
    
    for field_size in field_sizes:
        print(f"\n📏 Testing Field Size: {field_size}")
        print("-"*80)
        
        trainer = CognitionSimTrainer(field_size=field_size, device='cpu')
        
        metrics = await trainer.train(
            dataset_name='wikitext',
            dataset_config='wikitext-2-raw-v1',
            num_batches=50,  # Shorter for comparison
            batch_size=10
        )
        
        results[field_size] = {
            'peak_speedup': max(metrics['speedup_factors']),
            'final_reward': metrics['rewards'][-1],
            'final_loss': metrics['training_time'][-1],
            'avg_k_clusters': sum(metrics['k_values']) / len(metrics['k_values'])
        }
        
        print(f"\n✅ Field {field_size} Complete!")
        print(f"   Peak Speedup: {results[field_size]['peak_speedup']:.2f}x")
        print(f"   Final Reward: {results[field_size]['final_reward']:.4f}")
        print(f"   Avg K-Clusters: {results[field_size]['avg_k_clusters']:.1f}")
    
    print("\n" + "="*80)
    print("📊 FIELD SIZE COMPARISON")
    print("="*80 + "\n")
    print(f"{'Field Size':<12} | {'Speedup':<8} | {'Reward':<10} | {'Avg K':<8}")
    print("-"*50)
    
    for field_size, data in results.items():
        print(f"{field_size:<12} | {data['peak_speedup']:<8.2f} | "
              f"{data['final_reward']:<10.4f} | {data['avg_k_clusters']:<8.1f}")
    
    # Find best configuration
    best_field = max(results.items(), key=lambda x: x[1]['final_reward'])
    print(f"\n🏆 Best Configuration: Field Size {best_field[0]}")
    print(f"   Achieved {best_field[1]['final_reward']:.4f} reward")
    
    return results


async def main():
    """Main enhanced training launcher"""
    
    print("\n" + "="*80)
    print("🌟 QUADRA MATRIX A.I. - ENHANCED TRAINING LAUNCHER")
    print("="*80)
    print("\nSelect training mode:\n")
    print("  1. 🚀 Enhanced Single Run (200 batches, field=200)")
    print("  2. 🌐 Multi-Dataset Training (WikiText + Shakespeare)")
    print("  3. 🔬 Field Size Scaling Experiment (100-250)")
    print("  4. 🎯 Full Suite (All of the above)")
    print("  5. ❌ Cancel")
    print()
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == '1':
        await enhanced_training()
    elif choice == '2':
        await multi_dataset_training()
    elif choice == '3':
        await scaled_field_comparison()
    elif choice == '4':
        print("\n🎯 Running Full Enhanced Training Suite...")
        print("This will take approximately 45-60 minutes.\n")
        
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            print("\n" + "="*80)
            print("FULL SUITE STARTING")
            print("="*80)
            
            # Run all experiments
            print("\n\n" + "="*80)
            print("EXPERIMENT 1/3: Enhanced Single Run")
            print("="*80)
            await enhanced_training()
            
            print("\n\n" + "="*80)
            print("EXPERIMENT 2/3: Multi-Dataset Training")
            print("="*80)
            await multi_dataset_training()
            
            print("\n\n" + "="*80)
            print("EXPERIMENT 3/3: Field Size Scaling")
            print("="*80)
            await scaled_field_comparison()
            
            print("\n\n" + "="*80)
            print("🎉 FULL SUITE COMPLETE!")
            print("="*80)
            print("\n✅ All experiments finished successfully!")
            print("\n📁 Generated files:")
            print("   • atlantean_hrm_enhanced.pth")
            print("   • atlantean_hrm_wikitext.pth")
            print("   • atlantean_hrm_tiny_shakespeare.pth")
            print("   • training_metrics.json")
            print("   • training_metrics.png")
        else:
            print("\n❌ Cancelled.")
    else:
        print("\n❌ Cancelled or invalid choice.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        print("💾 Partial results may have been saved.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
