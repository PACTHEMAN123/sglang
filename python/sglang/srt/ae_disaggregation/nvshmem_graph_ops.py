"""Experimental CUDA-graph-friendly NVSHMEM device-side signaling ops.

This module is intentionally isolated from the production AE communication path.
It JIT-builds a small CUDA extension that keeps the per-batch iteration counter
on GPU and calls device-side NVSHMEM signal/wait APIs from CUDA kernels.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, _get_build_directory


_MODULE = None


def _nvshmem_root() -> Path:
    spec = importlib.util.find_spec("nvidia.nvshmem")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("nvidia.nvshmem package is not importable")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    root = _nvshmem_root()
    include_dir = root / "include"
    lib_dir = root / "lib"
    device_lib = lib_dir / "libnvshmem_device.a"
    host_lib = lib_dir / "libnvshmem_host.so.3"
    if not include_dir.exists() or not device_lib.exists() or not host_lib.exists():
        raise RuntimeError(
            "NVSHMEM wheel does not expose the expected include/lib layout: "
            f"include={include_dir}, device_lib={device_lib}, host_lib={host_lib}"
        )

    cpp_src = r"""
#include <torch/extension.h>
#include <cstdint>

void ae_nvshmem_send_signal_launch(
    std::uintptr_t dst,
    std::uintptr_t src,
    std::size_t bytes,
    std::uintptr_t sig_addr,
    torch::Tensor counter,
    std::uint64_t layer,
    std::uint64_t tokens,
    int pe,
    torch::Tensor debug,
    std::uintptr_t stream);

void ae_nvshmem_wait_signal_launch(
    std::uintptr_t sig_addr,
    torch::Tensor counter,
    torch::Tensor metadata,
    std::uintptr_t stream);

void ae_nvshmem_exchange_signal_launch(
    std::uintptr_t dst,
    std::uintptr_t src,
    std::size_t bytes,
    std::uintptr_t sig_addr,
    torch::Tensor counter,
    std::uint64_t layer,
    std::uint64_t tokens,
    int peer_pe,
    int role,
    torch::Tensor metadata,
    torch::Tensor debug,
    std::uintptr_t stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("send_signal", &ae_nvshmem_send_signal_launch);
    m.def("wait_signal", &ae_nvshmem_wait_signal_launch);
    m.def("exchange_signal", &ae_nvshmem_exchange_signal_launch);
}
"""

    cuda_src = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/types.h>

#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#include <cuda_fp16.h>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace {

constexpr std::uint64_t ITER_SHIFT = 40;
constexpr std::uint64_t LAYER_SHIFT = 20;
constexpr std::uint64_t FIELD_MASK = (1ULL << 20) - 1ULL;

__device__ __forceinline__ std::uint64_t pack_signal(
    std::uint64_t iteration,
    std::uint64_t layer,
    std::uint64_t tokens) {
    return (iteration << ITER_SHIFT) | (layer << LAYER_SHIFT) | tokens;
}

__global__ void send_signal_kernel(
    void* dst,
    const void* src,
    std::size_t bytes,
    std::uint64_t* sig_addr,
    std::uint64_t* counter,
    std::uint64_t layer,
    std::uint64_t tokens,
    int pe,
    std::uint64_t* debug) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        std::uint64_t iteration =
            atomicAdd(reinterpret_cast<unsigned long long*>(counter), 1ULL) + 1ULL;
        std::uint64_t signal = pack_signal(iteration, layer, tokens);
        nvshmem_putmem_signal(
            dst,
            src,
            bytes,
            sig_addr,
            signal,
            NVSHMEM_SIGNAL_SET,
            pe);
        if (debug != nullptr) {
            debug[0] = iteration;
            debug[1] = signal;
        }
    }
}

__global__ void wait_signal_kernel(
    std::uint64_t* sig_addr,
    std::uint64_t* counter,
    std::uint64_t* metadata) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        std::uint64_t iteration =
            atomicAdd(reinterpret_cast<unsigned long long*>(counter), 1ULL) + 1ULL;
        std::uint64_t threshold = iteration << ITER_SHIFT;
        std::uint64_t value =
            nvshmem_signal_wait_until(sig_addr, NVSHMEM_CMP_GE, threshold);
        metadata[0] = iteration;
        metadata[1] = (value >> LAYER_SHIFT) & FIELD_MASK;
        metadata[2] = value & FIELD_MASK;
        metadata[3] = value;
    }
}

__global__ void exchange_signal_kernel(
    void* dst,
    const void* src,
    std::size_t bytes,
    std::uint64_t* sig_addr,
    std::uint64_t* counter,
    std::uint64_t layer,
    std::uint64_t tokens,
    int peer_pe,
    int role,
    std::uint64_t* metadata,
    std::uint64_t* debug) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        std::uint64_t iteration =
            atomicAdd(reinterpret_cast<unsigned long long*>(counter), 1ULL) + 1ULL;
        if (role == 0) {
            std::uint64_t signal = pack_signal(iteration, layer, tokens);
            nvshmem_putmem_signal(
                dst,
                src,
                bytes,
                sig_addr,
                signal,
                NVSHMEM_SIGNAL_SET,
                peer_pe);
            if (debug != nullptr) {
                debug[0] = iteration;
                debug[1] = signal;
            }
        } else {
            std::uint64_t threshold = iteration << ITER_SHIFT;
            std::uint64_t value =
                nvshmem_signal_wait_until(sig_addr, NVSHMEM_CMP_GE, threshold);
            if (metadata != nullptr) {
                metadata[0] = iteration;
                metadata[1] = (value >> LAYER_SHIFT) & FIELD_MASK;
                metadata[2] = value & FIELD_MASK;
                metadata[3] = value;
            }
        }
    }
}

cudaStream_t resolve_stream(std::uintptr_t stream) {
    if (stream != 0) {
        return reinterpret_cast<cudaStream_t>(stream);
    }
    return at::cuda::getCurrentCUDAStream().stream();
}

void check_nvshmem_status(int status, const char* op) {
    TORCH_CHECK(status == 0, op, " failed with status=", status);
}

void ensure_nvshmem_module_initialized() {
    static bool initialized = false;
    if (initialized) {
        return;
    }

    cudaFunction_t function = nullptr;
    cudaError_t cuda_status = cudaGetFuncBySymbol(
        &function, reinterpret_cast<const void*>(exchange_signal_kernel));
    TORCH_CHECK(
        cuda_status == cudaSuccess,
        "cudaGetFuncBySymbol(exchange_signal_kernel) failed: ",
        cudaGetErrorString(cuda_status));

    CUmodule module = nullptr;
    CUresult driver_status = cuFuncGetModule(
        &module, reinterpret_cast<CUfunction>(function));
    TORCH_CHECK(
        driver_status == CUDA_SUCCESS,
        "cuFuncGetModule(exchange_signal_kernel) failed with status=",
        static_cast<int>(driver_status));

    check_nvshmem_status(
        nvshmemx_cumodule_init(module),
        "nvshmemx_cumodule_init(exchange_signal_kernel module)");
    initialized = true;
}

}  // namespace

void ae_nvshmem_send_signal_launch(
    std::uintptr_t dst,
    std::uintptr_t src,
    std::size_t bytes,
    std::uintptr_t sig_addr,
    torch::Tensor counter,
    std::uint64_t layer,
    std::uint64_t tokens,
    int pe,
    torch::Tensor debug,
    std::uintptr_t stream) {
    TORCH_CHECK(counter.is_cuda(), "counter must be a CUDA tensor");
    TORCH_CHECK(counter.scalar_type() == torch::kUInt64 || counter.scalar_type() == torch::kInt64,
                "counter must be int64/uint64");
    auto debug_ptr = debug.defined() ? reinterpret_cast<std::uint64_t*>(debug.data_ptr()) : nullptr;
    send_signal_kernel<<<1, 32, 0, resolve_stream(stream)>>>(
        reinterpret_cast<void*>(dst),
        reinterpret_cast<const void*>(src),
        bytes,
        reinterpret_cast<std::uint64_t*>(sig_addr),
        reinterpret_cast<std::uint64_t*>(counter.data_ptr()),
        layer,
        tokens,
        pe,
        debug_ptr);
}

void ae_nvshmem_wait_signal_launch(
    std::uintptr_t sig_addr,
    torch::Tensor counter,
    torch::Tensor metadata,
    std::uintptr_t stream) {
    TORCH_CHECK(counter.is_cuda(), "counter must be a CUDA tensor");
    TORCH_CHECK(metadata.is_cuda(), "metadata must be a CUDA tensor");
    TORCH_CHECK(counter.scalar_type() == torch::kUInt64 || counter.scalar_type() == torch::kInt64,
                "counter must be int64/uint64");
    TORCH_CHECK(metadata.scalar_type() == torch::kUInt64 || metadata.scalar_type() == torch::kInt64,
                "metadata must be int64/uint64");
    wait_signal_kernel<<<1, 32, 0, resolve_stream(stream)>>>(
        reinterpret_cast<std::uint64_t*>(sig_addr),
        reinterpret_cast<std::uint64_t*>(counter.data_ptr()),
        reinterpret_cast<std::uint64_t*>(metadata.data_ptr()));
}

void ae_nvshmem_exchange_signal_launch(
    std::uintptr_t dst,
    std::uintptr_t src,
    std::size_t bytes,
    std::uintptr_t sig_addr,
    torch::Tensor counter,
    std::uint64_t layer,
    std::uint64_t tokens,
    int peer_pe,
    int role,
    torch::Tensor metadata,
    torch::Tensor debug,
    std::uintptr_t stream) {
    TORCH_CHECK(counter.is_cuda(), "counter must be a CUDA tensor");
    TORCH_CHECK(counter.scalar_type() == torch::kUInt64 || counter.scalar_type() == torch::kInt64,
                "counter must be int64/uint64");
    auto metadata_ptr =
        metadata.defined() && metadata.numel() > 0
            ? reinterpret_cast<std::uint64_t*>(metadata.data_ptr())
            : nullptr;
    auto debug_ptr =
        debug.defined() && debug.numel() > 0
            ? reinterpret_cast<std::uint64_t*>(debug.data_ptr())
            : nullptr;
    void* dst_ptr = reinterpret_cast<void*>(dst);
    void* src_ptr = reinterpret_cast<void*>(src);
    std::uint64_t* sig_ptr = reinterpret_cast<std::uint64_t*>(sig_addr);
    std::uint64_t* counter_ptr = reinterpret_cast<std::uint64_t*>(counter.data_ptr());
    cudaStream_t cuda_stream = resolve_stream(stream);
    ensure_nvshmem_module_initialized();

    void* args[] = {
        &dst_ptr,
        &src_ptr,
        &bytes,
        &sig_ptr,
        &counter_ptr,
        &layer,
        &tokens,
        &peer_pe,
        &role,
        &metadata_ptr,
        &debug_ptr,
    };
    int status = nvshmemx_collective_launch(
        reinterpret_cast<const void*>(exchange_signal_kernel),
        dim3(1),
        dim3(32),
        args,
        0,
        cuda_stream);
    check_nvshmem_status(status, "nvshmemx_collective_launch(exchange_signal_kernel)");
}
"""

    name = "sglang_ae_nvshmem_graph_ops"
    build_dir = Path(_get_build_directory(name, verbose=False))
    build_dir.mkdir(parents=True, exist_ok=True)
    cpp_path = build_dir / "main.cpp"
    cuda_path = build_dir / "cuda.cu"
    cpp_path.write_text(cpp_src)
    cuda_path.write_text(cuda_src)

    from setuptools import setup

    old_argv = sys.argv[:]
    try:
        sys.argv = ["setup.py", "build_ext", "--inplace"]
        setup(
            name=name,
            ext_modules=[
                CUDAExtension(
                    name=name,
                    sources=[str(cpp_path), str(cuda_path)],
                    include_dirs=[str(include_dir)],
                    library_dirs=[str(lib_dir), "/usr/local/cuda/lib64/stubs"],
                    dlink=True,
                    dlink_libraries=["nvshmem_device"],
                    extra_objects=[str(device_lib), str(host_lib)],
                    extra_compile_args={
                        "cxx": ["-std=c++17"],
                        "nvcc": [
                            "-std=c++17",
                            "-rdc=true",
                            "-U__CUDA_NO_HALF_OPERATORS__",
                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                            "-U__CUDA_NO_HALF2_OPERATORS__",
                            f"-I{include_dir}",
                        ],
                    },
                    extra_link_args=["-lcuda", f"-Wl,-rpath,{lib_dir}"],
                )
            ],
            cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
            script_args=["build_ext", "--inplace"],
            options={"build_ext": {"build_lib": str(build_dir)}},
            verbose=os.environ.get("SGLANG_AE_NVSHMEM_BUILD_VERBOSE", "0") == "1",
        )
    finally:
        sys.argv = old_argv

    candidates = sorted(build_dir.glob(f"{name}*.so"))
    if not candidates:
        raise RuntimeError(f"failed to build {name} in {build_dir}")
    spec = importlib.util.spec_from_file_location(name, candidates[-1])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load extension spec from {candidates[-1]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE = module
    return _MODULE


def send_signal(
    dst: int,
    src: int,
    nbytes: int,
    sig_addr: int,
    counter: torch.Tensor,
    layer: int,
    tokens: int,
    pe: int,
    debug: torch.Tensor | None = None,
    stream: int | None = None,
):
    module = _load_module()
    if debug is None:
        debug = torch.empty(0, dtype=torch.int64, device=counter.device)
    module.send_signal(
        int(dst),
        int(src),
        int(nbytes),
        int(sig_addr),
        counter,
        int(layer),
        int(tokens),
        int(pe),
        debug,
        int(stream or torch.cuda.current_stream().cuda_stream),
    )


def wait_signal(
    sig_addr: int,
    counter: torch.Tensor,
    metadata: torch.Tensor,
    stream: int | None = None,
):
    module = _load_module()
    module.wait_signal(
        int(sig_addr),
        counter,
        metadata,
        int(stream or torch.cuda.current_stream().cuda_stream),
    )


def exchange_signal(
    dst: int,
    src: int,
    nbytes: int,
    sig_addr: int,
    counter: torch.Tensor,
    layer: int,
    tokens: int,
    peer_pe: int,
    role: int,
    metadata: torch.Tensor | None = None,
    debug: torch.Tensor | None = None,
    stream: int | None = None,
):
    module = _load_module()
    if metadata is None:
        metadata = torch.empty(0, dtype=torch.int64, device=counter.device)
    if debug is None:
        debug = torch.empty(0, dtype=torch.int64, device=counter.device)
    module.exchange_signal(
        int(dst),
        int(src),
        int(nbytes),
        int(sig_addr),
        counter,
        int(layer),
        int(tokens),
        int(peer_pe),
        int(role),
        metadata,
        debug,
        int(stream or torch.cuda.current_stream().cuda_stream),
    )
