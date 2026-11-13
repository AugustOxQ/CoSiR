"""Check spawn-based AMI workers after a multithreaded Torch matrix product."""
import r29rd_c1_core as C
import numpy as np
import torch


def main():
    rng=np.random.default_rng(7)
    x=torch.from_numpy(rng.normal(size=(128,64)).astype(np.float32))
    (x@x.T).sum().item()
    y=np.array([0,0,-1,1,1,0])
    candidate=rng.integers(0,2,size=(5,3,len(y)))
    comparator=rng.integers(0,2,size=(3,len(y)))
    serial=C.ami_interval(candidate,comparator,y,draws=13,workers=1)
    parallel=C.ami_interval(candidate,comparator,y,draws=13,workers=2)
    np.testing.assert_allclose(list(serial.values()),list(parallel.values()),rtol=0,atol=1e-14)
    print('Spawn workers after Torch matmul: serial/parallel match within 1e-14.')


if __name__=='__main__':
    main()
