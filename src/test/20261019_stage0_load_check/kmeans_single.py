"""kmeans_repeat.py with one BLAS/OpenMP thread for k-means: are the centers then bit-identical across runs?"""
import runpy
import sys
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1):
    sys.argv = ['kmeans_repeat.py', sys.argv[1], sys.argv[2]]
    runpy.run_path('kmeans_repeat.py', run_name='__main__')
