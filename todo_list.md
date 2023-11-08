# TODO List
All notable ideas will be documented in this file.

## New Features

### Implicit gemm convolution
- Implement packing of input tensor.
- Implement custom epilogue that fuses bias and activation.

### Dense layer
- Implement fused gemm+bias+act for CPU.

### Pooling
- Implement global average- and max-pooling.

### SE-block
- Add Squeeze-and-excitation block.


## Optimizations
- Improve 12x8 avx2 gemm compute kernel.
- Implement 5x5 input and output Winograd transforms in assembly.
- Maybe optimize im2row? (unless implicit gemm conv is done)
- Try optimize training on CUDA.


## Refactorings
- Remove vector classes as they are used only in a few places and probably suboptimal.
- Remove vector classes on CUDA and implement special cases manually for certain algorithms.