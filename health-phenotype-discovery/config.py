"""
Project Configuration - Embedded Paths and Utilities

Health Phenotype Discovery Project
"""

import os
import sys
import joblib

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# PROJECT CONFIGURATION - EMBEDDED PATHS AND UTILITIES
# =============================================================================

print("=" * 70)
print("PROJECT CONFIGURATION")
print("=" * 70)

# Define project root directory (current working directory)
PROJECT_ROOT = os.path.abspath(os.path.dirname('__file__'))

# Define main directory paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_v2')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Define phase-specific subdirectories
PHASE_DIRS = {
    'data': os.path.join(DATA_DIR, 'raw'),
    'processed': os.path.join(DATA_DIR, 'processed'),
    'reports': os.path.join(OUTPUT_DIR, 'reports'),
    'logs': os.path.join(OUTPUT_DIR, 'logs'),
    'plots': os.path.join(FIGURES_DIR, 'plots')
}

# Define model subdirectories
MODEL_SUBDIRS = {
    'gmm_clustering': os.path.join(MODELS_DIR, 'gmm_clustering'),
    'baseline': os.path.join(MODELS_DIR, 'baseline'),
    'tuned': os.path.join(MODELS_DIR, 'tuned'),
    'final': os.path.join(MODELS_DIR, 'final'),
    'comparison': os.path.join(MODELS_DIR, 'comparison')
}

# Define output subdirectories
OUTPUT_SUBDIRS = {
    'metrics': os.path.join(OUTPUT_DIR, 'metrics'),
    'predictions': os.path.join(OUTPUT_DIR, 'predictions'),
    'thresholds': os.path.join(OUTPUT_DIR, 'thresholds'),
    'fairness': os.path.join(OUTPUT_DIR, 'fairness'),
    'validation': os.path.join(OUTPUT_DIR, 'validation'),
    'cluster_profiles': os.path.join(OUTPUT_DIR, 'cluster_profiles')
}

# Create all directories if they don't exist
all_dirs = [
    PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR,
    *PHASE_DIRS.values(), *MODEL_SUBDIRS.values(), *OUTPUT_SUBDIRS.values()
]

created_count = 0
for dir_path in all_dirs:
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        created_count += 1

print(f"\n[INFO] Directory Structure:")
print(f"  Project Root: {PROJECT_ROOT}")
print(f"  Data Directory: {DATA_DIR}")
print(f"  Output Directory: {OUTPUT_DIR}")
print(f"  Models Directory: {MODELS_DIR}")
print(f"  Figures Directory: {FIGURES_DIR}")
print(f"\n  Created {created_count} directory(ies)")

# Define utility functions
def save_fig(figure, filename, subdir=None, formats=['png', 'pdf', 'svg']):
    """Save a matplotlib figure in multiple formats."""
    save_dir = FIGURES_DIR
    if subdir:
        save_dir = os.path.join(FIGURES_DIR, subdir)
        os.makedirs(save_dir, exist_ok=True)
    
    saved_files = []
    for fmt in formats:
        filepath = os.path.join(save_dir, f"{filename}.{fmt}")
        figure.savefig(filepath, dpi=300, bbox_inches='tight', format=fmt)
        saved_files.append(filepath)
    
    return saved_files

def save_model(model, filename, subdir=None):
    """Save a trained model using joblib."""
    if subdir and subdir in MODEL_SUBDIRS:
        save_dir = MODEL_SUBDIRS[subdir]
    else:
        save_dir = MODELS_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{filename}.joblib")
    joblib.dump(model, filepath)
    return filepath

def save_data(data, filename, subdir=None, fmt='csv'):
    """Save data (DataFrame or array) to file."""
    if subdir and subdir in OUTPUT_SUBDIRS:
        save_dir = OUTPUT_SUBDIRS[subdir]
    else:
        save_dir = OUTPUT_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{filename}.{fmt}")
    
    if fmt == 'csv':
        if hasattr(data, 'to_csv'):
            data.to_csv(filepath, index=False)
        else:
            pd.DataFrame(data).to_csv(filepath, index=False)
    
    return filepath

print("\n[OK] Utility functions defined successfully!")
print("=" * 70)
