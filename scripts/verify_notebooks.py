import glob
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import sys

def verify_notebook(notebook_path):
    print(f"Verifying {notebook_path}...")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        try:
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
            print(f"PASS: {notebook_path} executed successfully.")
            return True
        except Exception as e:
            print(f"FAIL: {notebook_path} execution failed.")
            print(e)
            return False
            
    except Exception as e:
        print(f"FAIL: {notebook_path} could not be read or parsed.")
        print(e)
        return False

def main():
    notebook_dir = os.path.join(os.path.dirname(__file__), '..', 'notebooks', '00_math_foundations')
    notebooks = glob.glob(os.path.join(notebook_dir, '*.ipynb'))
    
    if not notebooks:
        print(f"No notebooks found in {notebook_dir}")
        sys.exit(1)
        
    success = True
    for nb in notebooks:
        if not verify_notebook(nb):
            success = False
            
    if success:
        print("\nAll notebooks verified successfully!")
        sys.exit(0)
    else:
        print("\nSome notebooks failed verification.")
        sys.exit(1)

if __name__ == "__main__":
    main()
