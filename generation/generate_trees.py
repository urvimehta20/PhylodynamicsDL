import numpy as np
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

n_trees = 20000
vals = np.sort(np.random.uniform(0.20, 0.8, size=(n_trees, 2)).round(3))

# setup
venv_bin = Path(
    "/Users/mmcanear/Projects/PhD_Courses/STAT700/PhylodynamicsDL/.venv/bin"
)
generate_bd = venv_bin / "generate_bd"

max_workers = 10


def generate_tree(i, la, psi, out_dir=Path("./output_trees"), timeout_seconds=5):
    out_dir.mkdir(exist_ok=True, parents=True)
    tree_path = out_dir / f"tree_{i}.nwk"
    log_path = out_dir / f"params_{i}.csv"

    if tree_path.exists():
        return None

    try:
        subprocess.run(
            [
                str(generate_bd),
                "--min_tips",
                "100",
                "--max_tips",
                "500",
                "--la",
                str(la),
                "--psi",
                str(psi),
                "--p",
                str(0.5),
                "--nwk",
                tree_path,
                "--log",
                log_path,
            ],
            check=True,
            timeout=timeout_seconds,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return tree_path, log_path
    except subprocess.TimeoutExpired:
        # print(f"Timeout encountered! Skipping tree {i}.")
        return None


output_paths = []
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = []
    for i, row in enumerate(vals):
        # la, psi = row
        psi, la = row
        futures.append(ex.submit(generate_tree, i, la, psi))
    for fut in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Simulating Trees.",
        unit="trees",
    ):
        pass
