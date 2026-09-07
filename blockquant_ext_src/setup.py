from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6;8.9;9.0;10.0;10.3;12.0')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6;8.9;9.0')

# Set parallel compile jobs
os.environ.setdefault('MAX_JOBS', '8')

extra_compile_args = {
    'cxx': ['-O3', '-std=c++17', '-fPIC', '-DNDEBUG'],
    'nvcc': ['-O3', '--expt-relaxed-constexpr', '-lineinfo', '--threads=8', '--use_fast_math'],
}

sources = ["blockquant_ext.cpp", "gemm.cu"]

ext_modules = [
    CUDAExtension(
        name='blockquant_ext',
        sources=sources,
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name='blockquant_ext',
    version='0.0.1',
    description='BlockQuant GEMM extension (fp16 inputs, fp16/fp32 accumulators)',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.8',
)
