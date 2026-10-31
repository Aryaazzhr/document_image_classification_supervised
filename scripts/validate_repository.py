"""Check documentation, notebook structure, docstrings, and files intended for Git."""

import ast
import json
from pathlib import Path

from scripts.preprocess import ROOT


def validate_repository():
    """Reject missing docstrings, invalid notebooks, and oversized project artifacts."""
    import nbformat
    checked, executed_notebooks = [], []
    for folder in ("scripts", "tests", "configs", "notebooks", "reports"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.stat().st_size > 100_000_000:
                raise ValueError(f"Artifact exceeds 100 MB: {path}")
            if path.suffix == ".py":
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if not ast.get_docstring(node):
                            raise ValueError(f"Missing docstring: {path}:{node.lineno}")
            if path.suffix == ".ipynb":
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                if all(item.execution_count is not None for item in notebook.cells if item.cell_type == "code"):
                    executed_notebooks.append(path.name)
                for item in notebook.cells:
                    if item.cell_type == "code":
                        compile(item.source, str(path), "exec")
                        if any(output.output_type == "error" for output in item.outputs):
                            raise ValueError(f"Notebook contains an execution error: {path}")
            checked.append(str(path.relative_to(ROOT)))
    ignores = (ROOT / ".gitignore").read_text().splitlines()
    for required in ("data/", "checkpoint/", "train/", "test/", "*.pt", "*.tif"):
        if required not in ignores:
            raise ValueError(f"Required .gitignore pattern missing: {required}")
    if not (ROOT / "README.md").exists():
        raise ValueError("README.md is missing")
    result = {"checked_artifacts": len(checked), "docstrings": "pass", "notebooks": "pass",
              "artifact_size_limit_mb": 100, "required_ignores": "pass",
              "executed_notebooks": sorted(executed_notebooks)}
    (ROOT / "reports/repository_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    validate_repository()
