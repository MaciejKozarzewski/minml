import os
import sys

from cuda_builder import CudaBuilder
from cpu_builder import CpuBuilder
from utils import run_commands

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('expected single argument')
        exit(-1)

    if not os.path.exists('../bin/objects'):
        os.mkdir('../bin/objects')

    build_target = sys.argv[1]

    use_cuda = True
    if len(sys.argv) == 3:
        use_cuda = not (sys.argv[2] == 'no_cuda')

    #cb = CpuBuilder(build_target, 'gcc')
    #run_commands(cb.create_lib())
    #cb.clear_objects()
    if use_cuda:
        cb = CudaBuilder(build_target, 'gcc', 61)
        run_commands(cb.create_lib())
        cb.clear_objects()
