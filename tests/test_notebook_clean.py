import json
from pathlib import Path


def test_analiza_notebook_has_no_execution_state():
    nb_path = Path(__file__).resolve().parents[1] / "analiza.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
