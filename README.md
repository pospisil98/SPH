# Smoothed Particle Hydrodynamics

Implementation of SPH in C++ with possibility to use CPU or GPU (CUDA) for particles updates.

It is pretty trivial implementation with uniform spatial grid used as acceleration structure for faster neighbor interactions. GPU part is definitely not optimised as the grid is being built on CPU and then copied onto GPU, and there is no usage of shared memory or other things besides global memory.

