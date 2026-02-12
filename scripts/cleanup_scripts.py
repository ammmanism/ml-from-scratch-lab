import os
import glob

def cleanup():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    patterns = ['gen_nb_1.py', 'gen_nb_2.py', 'gen_vectors_nb_part1.py']
    
    print(f"Cleaning up temporary scripts in {script_dir}...")
    
    for pattern in patterns:
        files = glob.glob(os.path.join(script_dir, pattern))
        for f in files:
            try:
                os.remove(f)
                print(f"Removed: {f}")
            except Exception as e:
                print(f"Error removing {f}: {e}")

if __name__ == "__main__":
    cleanup()
