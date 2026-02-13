"""
Cleanup utility for temporary and generated scripts

⚠️ AI-GENERATED CODE
This module was generated using AI assistance.
Review and test thoroughly before using in production.

Author: GitHub Copilot (AI Assistant)
Created: February 2026
"""

import os
import glob
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def cleanup_patterns(patterns: list = None) -> dict:
    """
    Clean up temporary files matching patterns
    
    Args:
        patterns: List of file patterns to remove (glob patterns)
        
    Returns:
        Dictionary with cleanup statistics
    """
    if patterns is None:
        patterns = [
            'gen_nb_*.py',           # Generated notebooks
            '__pycache__',           # Python cache
            '.pytest_cache',         # Pytest cache
            '*.pyc',                 # Compiled Python
            '*.pyo',                 # Optimized Python
        ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats = {'removed': 0, 'errors': 0, 'dirs': 0}
    
    logger.info(f"🧹 Cleaning up temporary files in {script_dir}...\n")
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(script_dir, pattern))
        
        for match_path in matches:
            try:
                if os.path.isdir(match_path):
                    # Skip directory removal, just log it
                    logger.info(f"  ⚠️  Directory (skip): {os.path.basename(match_path)}")
                    stats['dirs'] += 1
                else:
                    os.remove(match_path)
                    logger.info(f"  ✅ Removed: {os.path.basename(match_path)}")
                    stats['removed'] += 1
            except Exception as e:
                logger.error(f"  ❌ Error with {os.path.basename(match_path)}: {e}")
                stats['errors'] += 1
    
    return stats


def cleanup_notebooks_cache():
    """Remove Jupyter notebook cache files"""
    notebook_dir = os.path.join(
        os.path.dirname(__file__), '..', 'notebooks'
    )
    
    logger.info(f"\n🧹 Cleaning Jupyter cache in {notebook_dir}...\n")
    
    stats = {'removed': 0, 'errors': 0}
    checkpoint_pattern = os.path.join(notebook_dir, '**/.ipynb_checkpoints')
    
    for checkpoint_dir in glob.glob(checkpoint_pattern, recursive=True):
        try:
            import shutil
            shutil.rmtree(checkpoint_dir)
            logger.info(f"  ✅ Removed: {checkpoint_dir}")
            stats['removed'] += 1
        except Exception as e:
            logger.error(f"  ❌ Error removing {checkpoint_dir}: {e}")
            stats['errors'] += 1
    
    return stats


def main():
    """Main cleanup function"""
    logger.info("=" * 60)
    logger.info("Repository Cleanup Utility (AI-Generated)")
    logger.info("=" * 60 + "\n")
    
    # Cleanup scripts
    script_stats = cleanup_patterns()
    
    # Cleanup notebooks cache
    nb_stats = cleanup_notebooks_cache()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Cleanup Summary")
    logger.info("=" * 60)
    logger.info(f"Scripts removed: {script_stats['removed']}")
    logger.info(f"Directories found: {script_stats['dirs']}")
    logger.info(f"Script errors: {script_stats['errors']}")
    logger.info(f"\nNotebook cache removed: {nb_stats['removed']}")
    logger.info(f"Notebook cache errors: {nb_stats['errors']}")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
