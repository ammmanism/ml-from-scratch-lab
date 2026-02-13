"""
Unified Notebook Generator & Manager
Handles creation, analysis, and enhancement of Jupyter notebooks

⚠️ AI-GENERATED CODE
This module was generated using AI assistance.
Review and test thoroughly before using in production.

Author: GitHub Copilot (AI Assistant)
Created: February 2026
Last Updated: February 2026
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class NotebookGenerator:
    """Unified notebook generation and management utility"""
    
    def __init__(self):
        """Initialize with standard Jupyter notebook metadata"""
        self.metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        }
    
    @staticmethod
    def create_code_cell(source: str) -> Dict[str, Any]:
        """
        Create a code cell
        
        Args:
            source: Python code as string
            
        Returns:
            Dictionary representing a code cell
        """
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        }
    
    @staticmethod
    def create_markdown_cell(source: str) -> Dict[str, Any]:
        """
        Create a markdown cell
        
        Args:
            source: Markdown content as string
            
        Returns:
            Dictionary representing a markdown cell
        """
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        }
    
    def create_notebook(self, cells: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Package cells into a complete notebook
        
        Args:
            cells: List of cell dictionaries
            
        Returns:
            Complete notebook dictionary
        """
        return {
            "cells": cells,
            "metadata": self.metadata,
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    def save_notebook(self, notebook: Dict[str, Any], filepath: str) -> bool:
        """
        Save notebook to file
        
        Args:
            notebook: Notebook dictionary
            filepath: Target file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=1)
            logger.info(f"✅ Saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving {filepath}: {e}")
            return False
    
    def load_notebook(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load notebook from file
        
        Args:
            filepath: Path to notebook file
            
        Returns:
            Notebook dictionary or None if error
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading {filepath}: {e}")
            return None
    
    def analyze_notebook(self, filepath: str) -> None:
        """
        Analyze and display notebook structure
        
        Args:
            filepath: Path to notebook file
        """
        nb = self.load_notebook(filepath)
        if not nb:
            return
        
        print(f"\n{'='*60}")
        print(f"📓 Notebook: {Path(filepath).name}")
        print(f"{'='*60}")
        print(f"Total Cells: {len(nb['cells'])}\n")
        
        for i, cell in enumerate(nb['cells'], 1):
            cell_type = cell['cell_type'].upper()
            source = cell['source']
            first_line = source[0].strip() if source else "<empty>"
            
            if len(first_line) > 70:
                first_line = first_line[:67] + "..."
            
            print(f"Cell {i:2d} [{cell_type:8s}]: {first_line}")
        
        print(f"{'='*60}\n")
    
    def validate_notebook(self, filepath: str) -> bool:
        """
        Validate notebook structure
        
        Args:
            filepath: Path to notebook file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            nb = self.load_notebook(filepath)
            if not nb:
                return False
            
            assert "cells" in nb, "Missing 'cells' key"
            assert "metadata" in nb, "Missing 'metadata' key"
            assert isinstance(nb["cells"], list), "'cells' must be a list"
            
            logger.info(f"✅ Valid notebook: {Path(filepath).name}")
            return True
        except AssertionError as e:
            logger.error(f"❌ Invalid notebook structure: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error validating {filepath}: {e}")
            return False
    
    def get_notebook_stats(self, filepath: str) -> Dict[str, Any]:
        """
        Get statistics about a notebook
        
        Args:
            filepath: Path to notebook file
            
        Returns:
            Dictionary with notebook statistics
        """
        nb = self.load_notebook(filepath)
        if not nb:
            return {}
        
        stats = {
            "total_cells": len(nb['cells']),
            "code_cells": 0,
            "markdown_cells": 0,
            "empty_cells": 0,
        }
        
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                stats['code_cells'] += 1
            elif cell['cell_type'] == 'markdown':
                stats['markdown_cells'] += 1
            
            if not cell.get('source'):
                stats['empty_cells'] += 1
        
        return stats


def demo_usage():
    """Example of how to use the NotebookGenerator"""
    gen = NotebookGenerator()
    
    # Example: Create a simple notebook
    cells = [
        gen.create_markdown_cell("# My Notebook\nThis is a test"),
        gen.create_code_cell("import numpy as np\nprint('Hello, World!')"),
    ]
    
    notebook = gen.create_notebook(cells)
    
    # Save to file
    output_path = "example_notebook.ipynb"
    gen.save_notebook(notebook, output_path)
    
    # Analyze it
    gen.analyze_notebook(output_path)
    
    # Validate it
    gen.validate_notebook(output_path)
    
    # Get stats
    stats = gen.get_notebook_stats(output_path)
    print(f"Notebook Stats: {stats}")


if __name__ == "__main__":
    print("=" * 60)
    print("Notebook Generator Module (AI-Generated)")
    print("=" * 60)
    print("Import this module and use NotebookGenerator class\n")
    print("Example:")
    print("  from notebook_generator import NotebookGenerator")
    print("  gen = NotebookGenerator()")
    print("  cells = [gen.create_code_cell('print(1+1)')]")
    print("  nb = gen.create_notebook(cells)")
    print("  gen.save_notebook(nb, 'output.ipynb')")
    print("=" * 60)
    
    # Uncomment to run demo
    # demo_usage()
