# llama.cpp

## SenseNova-Loop on AMD RDNA2 (ROCm/HIP): fused mHC ops

This fork of [OpenSenseNova/llama.cpp](https://github.com/OpenSenseNova/llama.cpp) (base commit `24ea7bb`)
speeds up token generation of `sensenova_loop` models on the HIP backend. The weights are not touched: the
same GGUF file is used, only the compute graph and the kernels change.

Measured on an RX 6900 XT (gfx1030, ROCm 10.1) with `SenseNova-Loop-9B-Q4_K_M.gguf`, `-ngl 99 -fa on`:

| | tg128 | pp128 |
|---|---|---|
| base `24ea7bb` | 6.0 t/s | 556 t/s |
| this fork | 40.5 t/s | 758 t/s |

One decoded token costs 22.3 ms of kernel time, 83 % of it in the quantized GEMV (`mul_mat_vec_q`), which
reads 8.2 GB of weights per token at 443 GB/s, i.e. 87 % of the card's theoretical memory bandwidth.

### What changed

| Change | Switch (default) | Numerics |
|---|---|---|
| 2x2 Sinkhorn loop as one op `GGML_OP_MHC_SINKHORN` | `LLAMA_MHC_SINKHORN_LEGACY=0` | bit-identical |
| Routing chain as one op `GGML_OP_MHC_ROUTE` | `LLAMA_MHC_ROUTE_FUSED=1` | rounding only |
| One-softplus Sinkhorn normalize inside the route kernel | `LLAMA_MHC_ROUTE_EXACT=0` | rounding only |
| Residual merge as one op `GGML_OP_MHC_MERGE` | `LLAMA_MHC_MERGE_FUSED=1` | rounding only |
| Router `MUL_MAT` no longer requests `GGML_PREC_F32` for src1 (HIP rejected it and ran 96 GEMVs per token on the CPU) | always | none |
| Gated-delta-net state read and written in place in the recurrent cache | `LLAMA_GDN_INPLACE=1` | bit-identical |
| Zero-sized recurrent-state nodes skipped | `LLAMA_RS_SKIP_ZERO=1` | bit-identical |

Further switches: `LLAMA_MHC_ROUTE_GENERIC=1` (scalar route kernel), `LLAMA_MHC_SINKHORN_NOFUSED=1`
(soft_max/log graph chain), `LLAMA_MHC_ITER_CLAMP=N` (fewer Sinkhorn iterations, diagnostics only).

The in-place state path is only taken when the state buffer lives on the CUDA/HIP device that runs the
layer (or on the host with `--no-op-offload`), for a single sequence without rollback snapshots. Everything
else uses the original gather/copy path.

### Measured KL divergence

`llama-perplexity --kl-divergence`, 16 chunks of 512 tokens (`-b 512 -ub 64`), compared against the
unfused reference graph of the same build (`LLAMA_MHC_ROUTE_FUSED=0 LLAMA_MHC_MERGE_FUSED=0
LLAMA_MHC_SINKHORN_LEGACY=1 LLAMA_GDN_INPLACE=0 LLAMA_RS_SKIP_ZERO=0`, PPL 7.073 ± 0.302).

| Variant | Mean KLD | Max KLD | Same top token | PPL |
|---|---|---|---|---|
| reference, second run | 0.000000 | 0.00006 | 100.00 % | 7.078 |
| fused Sinkhorn op only | 0.000000 | 0.00006 | 100.00 % | 7.078 |
| reference with `-fa off` (noise floor of the model) | 0.001405 | 0.346 | 98.36 % | 7.071 |
| fused merge only | 0.001085 | 0.120 | 98.58 % | 7.070 |
| **all optimizations (default)** | **0.001153** | **0.069** | **98.28 %** | **7.063** |
| default, `LLAMA_GDN_INPLACE=0` | 0.001153 | 0.069 | 98.28 % | 7.063 |
| default, `LLAMA_MHC_ROUTE_EXACT=1` | 0.001238 | 0.181 | 98.60 % | 7.066 |
| default, `LLAMA_MHC_ROUTE_GENERIC=1` | 0.001152 | 0.062 | 98.28 % | 7.075 |
| default, two sequences per batch (`-b 1024`) | 0.001177 | – | 98.63 % | 7.075 |
| default, `-ctk q8_0 -ctv q8_0` | 0.001279 | 0.043 | 98.36 % | 7.070 |
| default, `-ngl 20` (partial offload) | 0.001302 | – | 98.36 % | – |
| default, `-ngl 0` | 0.001256 | – | 98.31 % | 7.077 |
| CPU only (`-ngl 0 --no-op-offload`), default | 0.012536 | – | 95.66 % | – |
| CPU only, unfused reference graph | 0.012985 | – | 95.54 % | – |
| `LLAMA_MHC_SINKHORN_NOFUSED=1` (with `LLAMA_MHC_ROUTE_FUSED=0`) | 0.040057 | – | 92.43 % | 7.079 |
| `LLAMA_MHC_ITER_CLAMP=10` | 0.070800 | 6.655 | 90.59 % | 7.619 |
| `LLAMA_MHC_ITER_CLAMP=5` | 0.517135 | 11.202 | 74.14 % | 10.686 |

How to read this:

- The default configuration differs from the reference by the same amount as the reference differs from
  itself with another attention kernel (`-fa off`). The model amplifies rounding differences strongly, so
  any change of kernel or summation order lands at a mean KLD of about 0.001; perplexity does not move.
- CPU and GPU differ by about 0.013 in the unmodified graph already; the fused ops add nothing to that.
- `LLAMA_MHC_SINKHORN_NOFUSED` and `LLAMA_MHC_ITER_CLAMP` do lose accuracy and are kept for diagnostics only.

### Build (ROCm 10.1, gfx1030)

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm cmake -B build-rocm -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF -DHIP_ROOT=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_FLAGS="-isystem /opt/rocm/include -L/opt/rocm/lib" -DCMAKE_BUILD_RPATH=/opt/rocm/lib
cmake --build build-rocm -j --target llama-bench llama-server llama-cli llama-perplexity
```

### Limits

- Only the CUDA/HIP backend and the CPU backend implement the new ops. Other GPU backends fall back to
  the CPU for them.
- The route kernel assumes 32-lane wavefronts (RDNA); it is untested on gfx8/gfx9.
- Tested on one card and one model file only.

---


![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
