# TODO List
All notable ideas will be documented in this file.

## New Features

### Implicit gemm convolution
- Implement packing of input tensor.
- Implement custom epilogue that fuses bias and activation.

## Optimizations
- Maybe optimize im2row? (unless implicit gemm conv is done)
- Try optimize training on CUDA.

## Refactorings
- Remove vector classes on CUDA and implement special cases manually for certain algorithms.