# CoSiR

## Installation:

```bash
# Create conda environment
conda create -n CoSiR python=3.10 -y
conda activate CoSiR
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.8.*" "dask-cudf-cu12==25.8.*" "cuml-cu12==25.8.*" \
    "cugraph-cu12==25.8.*" "nx-cugraph-cu12==25.8.*" "cuxfilter-cu12==25.8.*" \
    "cucim-cu12==25.8.*" "pylibraft-cu12==25.8.*" "raft-dask-cu12==25.8.*" \
    "cuvs-cu12==25.8.*" "nx-cugraph-cu12==25.8.*"
pip install -e .
```