import json

nb_path = r"c:\Users\amman\.gemini\antigravity\scratch\ml_from_scratch_lib\notebooks\00_math_foundations\probability_distributions.ipynb"

try:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Total Cells: {len(nb['cells'])}")
    for i, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        source = cell['source']
        first_line = source[0].strip() if source else "<empty>"
        print(f"Cell {i} [{cell_type}]: {first_line}")

except Exception as e:
    print(f"Error: {e}")
