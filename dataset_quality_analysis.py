import os
import torch
import json
import numpy as np
from pathlib import Path

def analyze_dataset(data_dir="./large_quality_data"):
    """
    Comprehensive analysis of downloaded audio dataset
    """
    data_dir = Path(data_dir)
    
    # Find all dataset files
    dataset_files = list(data_dir.glob("quality_dataset_*.pt"))
    checkpoint_files = list(data_dir.glob("checkpoints/checkpoint_*.pt"))
    
    all_files = dataset_files + checkpoint_files
    
    if not all_files:
        print("❌ No dataset files found!")
        return
    
    print("🔍 Analyzing Dataset Files...")
    print("=" * 50)
    
    # Initialize statistics
    total_samples = 0
    dataset_breakdown = {}
    duration_stats = []
    mining_scores = []
    alignment_scores = []
    total_duration = 0
    
    # Analyze each file
    for file_path in all_files:
        try:
            print(f"📁 Loading: {file_path.name}")
            samples = torch.load(file_path)
            
            if not isinstance(samples, list):
                print(f"⚠️ Skipping {file_path.name} - unexpected format")
                continue
            
            file_samples = len(samples)
            total_samples += file_samples
            
            print(f"   ✅ Samples: {file_samples}")
            
            # Analyze each sample
            for sample in samples:
                dataset_name = sample.get('dataset', 'unknown')
                
                # Update dataset breakdown
                if dataset_name not in dataset_breakdown:
                    dataset_breakdown[dataset_name] = 0
                dataset_breakdown[dataset_name] += 1
                
                # Collect metrics
                duration = sample.get('duration', 0)
                if duration > 0:
                    duration_stats.append(duration)
                    total_duration += duration
                
                mining_score = sample.get('mining_score', 0)
                if mining_score > 0:
                    mining_scores.append(mining_score)
                
                alignment_score = sample.get('alignment_score', 0)
                if alignment_score > 0:
                    alignment_scores.append(alignment_score)
                    
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")
            continue
    
    # Calculate file sizes
    total_size = sum(f.stat().st_size for f in all_files if f.exists())
    
    # Print comprehensive report
    print("\n" + "=" * 50)
    print("📊 DATASET ANALYSIS REPORT")
    print("=" * 50)
    
    print(f"📈 TOTAL SAMPLES: {total_samples}")
    print(f"💾 TOTAL SIZE: {total_size / (1024**3):.2f} GB")
    print(f"⏱️ TOTAL DURATION: {total_duration / 3600:.2f} hours")
    
    print(f"\n📁 FILES ANALYZED: {len(all_files)}")
    for file_path in all_files:
        size_mb = file_path.stat().st_size / (1024**2)
        print(f"   • {file_path.name}: {size_mb:.1f} MB")
    
    print(f"\n🏷️ DATASET BREAKDOWN:")
    for dataset, count in dataset_breakdown.items():
        percentage = (count / total_samples) * 100
        print(f"   • {dataset}: {count} samples ({percentage:.1f}%)")
    
    if duration_stats:
        print(f"\n⏱️ DURATION STATISTICS:")
        print(f"   • Average: {np.mean(duration_stats):.2f} seconds")
        print(f"   • Minimum: {np.min(duration_stats):.2f} seconds") 
        print(f"   • Maximum: {np.max(duration_stats):.2f} seconds")
        print(f"   • Total: {total_duration / 3600:.2f} hours")
    
    if mining_scores:
        print(f"\n🎯 MINING SCORE STATISTICS:")
        print(f"   • Average: {np.mean(mining_scores):.3f}")
        print(f"   • Minimum: {np.min(mining_scores):.3f}")
        print(f"   • Maximum: {np.max(mining_scores):.3f}")
    
    if alignment_scores:
        print(f"\n📐 ALIGNMENT SCORE STATISTICS:")
        print(f"   • Average: {np.mean(alignment_scores):.3f}")
        print(f"   • Minimum: {np.min(alignment_scores):.3f}")
        print(f"   • Maximum: {np.max(alignment_scores):.3f}")
    
    # Save analysis report
    report = {
        "total_samples": total_samples,
        "total_size_gb": total_size / (1024**3),
        "total_duration_hours": total_duration / 3600,
        "dataset_breakdown": dataset_breakdown,
        "duration_stats": {
            "average": np.mean(duration_stats) if duration_stats else 0,
            "min": np.min(duration_stats) if duration_stats else 0,
            "max": np.max(duration_stats) if duration_stats else 0
        },
        "quality_metrics": {
            "mining_score_avg": np.mean(mining_scores) if mining_scores else 0,
            "alignment_score_avg": np.mean(alignment_scores) if alignment_scores else 0
        },
        "files_analyzed": [str(f.name) for f in all_files]
    }
    
    report_path = data_dir / "dataset_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Analysis report saved to: {report_path}")

def check_memory_usage():
    """Check current memory usage"""
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"🧠 Current memory usage: {memory_mb:.1f} MB")

if __name__ == "__main__":
    check_memory_usage()
    analyze_dataset()
    check_memory_usage()