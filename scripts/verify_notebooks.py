"""
Notebook verification and validation utility

⚠️ AI-GENERATED CODE
This module was generated using AI assistance.
Review and test thoroughly before using in production.

Author: GitHub Copilot (AI Assistant)
Created: February 2026
"""

import glob
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Try to import notebook execution tools (optional)
try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    HAS_NBCONVERT = True
except ImportError:
    HAS_NBCONVERT = False
    logger.warning("⚠️  nbformat/nbconvert not installed. Limited functionality.")


class NotebookValidator:
    """Validate and verify Jupyter notebooks"""
    
    def __init__(self, timeout: int = 600):
        """
        Initialize validator
        
        Args:
            timeout: Execution timeout in seconds
        """
        self.timeout = timeout
        self.results = {'passed': [], 'failed': []}
    
    def validate_structure(self, notebook_path: str) -> bool:
        """
        Validate notebook JSON structure
        
        Args:
            notebook_path: Path to notebook file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            assert 'cells' in nb, "Missing 'cells' key"
            assert 'metadata' in nb, "Missing 'metadata' key"
            assert isinstance(nb['cells'], list), "'cells' must be a list"
            
            logger.info(f"  ✅ Structure valid: {Path(notebook_path).name}")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"  ❌ Invalid JSON: {Path(notebook_path).name} - {e}")
            return False
        except AssertionError as e:
            logger.error(f"  ❌ Invalid structure: {Path(notebook_path).name} - {e}")
            return False
        except Exception as e:
            logger.error(f"  ❌ Error validating {Path(notebook_path).name}: {e}")
            return False
    
    def execute_notebook(self, notebook_path: str) -> bool:
        """
        Execute notebook and check for errors
        
        Args:
            notebook_path: Path to notebook file
            
        Returns:
            True if execution successful, False otherwise
        """
        if not HAS_NBCONVERT:
            logger.warning(f"  ⚠️  Skipping execution (nbconvert not installed): {Path(notebook_path).name}")
            return True
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            ep = ExecutePreprocessor(timeout=self.timeout, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
            
            logger.info(f"  ✅ Execution passed: {Path(notebook_path).name}")
            return True
            
        except TimeoutError:
            logger.error(f"  ❌ Execution timeout: {Path(notebook_path).name}")
            return False
        except Exception as e:
            logger.error(f"  ❌ Execution failed: {Path(notebook_path).name}")
            logger.error(f"     Error: {str(e)[:100]}")
            return False
    
    def verify_notebook(self, notebook_path: str, execute: bool = False) -> bool:
        """
        Complete notebook verification
        
        Args:
            notebook_path: Path to notebook file
            execute: Whether to execute the notebook
            
        Returns:
            True if all checks pass, False otherwise
        """
        logger.info(f"\n📋 Verifying: {Path(notebook_path).name}")
        
        # Validate structure
        if not self.validate_structure(notebook_path):
            self.results['failed'].append(notebook_path)
            return False
        
        # Execute if requested
        if execute:
            if not self.execute_notebook(notebook_path):
                self.results['failed'].append(notebook_path)
                return False
        
        self.results['passed'].append(notebook_path)
        return True
    
    def verify_directory(self, directory: str, pattern: str = '*.ipynb', execute: bool = False) -> bool:
        """
        Verify all notebooks in a directory
        
        Args:
            directory: Directory path
            pattern: Glob pattern for notebooks
            execute: Whether to execute notebooks
            
        Returns:
            True if all notebooks pass, False if any fail
        """
        notebook_files = glob.glob(os.path.join(directory, pattern))
        
        if not notebook_files:
            logger.warning(f"No notebooks found in {directory}")
            return True
        
        logger.info(f"Found {len(notebook_files)} notebook(s)")
        
        for notebook_path in notebook_files:
            if not self.verify_notebook(notebook_path, execute=execute):
                return False
        
        return True
    
    def report(self) -> None:
        """Print verification report"""
        total = len(self.results['passed']) + len(self.results['failed'])
        
        logger.info("\n" + "=" * 60)
        logger.info("Verification Report")
        logger.info("=" * 60)
        logger.info(f"Total notebooks: {total}")
        logger.info(f"✅ Passed: {len(self.results['passed'])}")
        logger.info(f"❌ Failed: {len(self.results['failed'])}")
        
        if self.results['failed']:
            logger.info("\nFailed notebooks:")
            for nb in self.results['failed']:
                logger.info(f"  - {Path(nb).name}")
        
        logger.info("=" * 60 + "\n")


def main():
    """Main verification function"""
    logger.info("=" * 60)
    logger.info("Notebook Verification Utility (AI-Generated)")
    logger.info("=" * 60)
    
    # Find notebooks directory
    notebook_dir = os.path.join(
        os.path.dirname(__file__), '..', 'notebooks', '00_math_foundations'
    )
    
    if not os.path.exists(notebook_dir):
        logger.error(f"Notebooks directory not found: {notebook_dir}")
        sys.exit(1)
    
    # Verify notebooks (without execution by default)
    validator = NotebookValidator()
    
    # Check if we should execute (optional CLI flag)
    execute = '--execute' in sys.argv or '-x' in sys.argv
    
    if execute:
        logger.info("Mode: Structure validation + Execution\n")
    else:
        logger.info("Mode: Structure validation only")
        logger.info("(Use --execute or -x flag to also execute notebooks)\n")
    
    success = validator.verify_directory(notebook_dir, execute=execute)
    validator.report()
    
    if success:
        logger.info("✅ All notebooks verified successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some notebooks failed verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
