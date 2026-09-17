"""Execute a notebook's code cells in order, in one namespace, and print their output.

Why not nbconvert: it is not installed in validation/venv, and installing it to run
four notebooks unattended is a heavier dependency than the twenty lines below. This
runs the notebook's OWN cell sources, in order, so what it proves is what the
notebook does - it just does not write the outputs back into the .ipynb.

    cd validation
    venv/bin/python3 run_notebook.py phase5_rul_v3_914.ipynb
    RUL_PREFIX=rul_v4 venv/bin/python3 run_notebook.py phase5_rul_v3_914.ipynb

Exits non-zero on the first failing cell, naming it - so a driver loop stops rather
than training three more engines against a broken one.
"""
import json
import sys
import traceback


def run(path):
    nb = json.load(open(path))
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if not src.strip():
            continue
        print(f"\n===== {path} cell {i} =====", flush=True)
        try:
            exec(compile(src, f"<{path} cell {i}>", "exec"), ns)
        except Exception:
            traceback.print_exc()
            print(f"CELL {i} FAILED", flush=True)
            return 1
    print(f"\n{path}: NOTEBOOK OK", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(run(sys.argv[1]))
