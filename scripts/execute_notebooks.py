"""Execute notebooks with the current interpreter and save authentic cell outputs."""

import argparse
import os
import sys

from scripts.preprocess import ROOT


def execute_notebooks(notebook_name=None):
    """Run each workflow notebook and persist successful execution results."""
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    cache = ROOT / ".cache/jupyter"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(cache)
    os.environ["IPYTHONDIR"] = str(cache / "ipython")
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    paths = [ROOT / "notebooks" / notebook_name] if notebook_name else sorted((ROOT / "notebooks").glob("*.ipynb"))
    for path in paths:
        print(f"Starting: {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3", connection_file=str(cache / f"kernel-{path.stem}.json"))
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, km=manager, timeout=7200, startup_timeout=90,
                                on_cell_start=log_cell,
                                resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(notebook, path)
        print(f"Executed: {path.name}", flush=True)


def log_cell(cell, cell_index, **kwargs):
    """Expose notebook progress so slow kernels and cells can be diagnosed."""
    if cell.cell_type == "code":
        print(f"  Executing cell {cell_index + 1}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=[path.name for path in (ROOT / "notebooks").glob("*.ipynb")])
    execute_notebooks(parser.parse_args().notebook)
