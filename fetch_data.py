import os
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

if __name__ == "__main__":
    data_dir = Path(os.environ["FLAKE_ROOT"], "data")
    if data_dir.exists:
        print("Dataset already donwloaded.")
    else:
        print("Fetching Cora dataset...")
        Planetoid(root=data_dir, name="Cora", transform=NormalizeFeatures())
        print("Done.")
