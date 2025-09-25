# vLLM benchmarks on AMD Instinct™ MI300X quickstart

Documentation for running benchmarks on AMD Instinct™ MI300X accelerators with vLLM.

## Overview

This guide is summarized from [https://github.com/ROCm/vllm/blob/documentation/docs/dev-docker/README.md](https://github.com/ROCm/vllm/blob/documentation/docs/dev-docker/README.md)

## Pull latest Docker Image

Pull the most recent validated docker image with `docker pull rocm/vllm-dev:main`

## Reproducing Benchmarked Results

### Preparation - Obtaining access to models

The `vllm-dev` docker image should work with any model supported by vLLM.  If needed, the vLLM benchmark scripts will automatically download models and then store them in a Hugging Face cache directory for reuse in future tests. Alternatively, you can choose to download the model to the cache (or to another directory on the system) in advance.

Many HuggingFace models, including Llama-3.1, have gated access.  You will need to set up an account at (https://huggingface.co), search for the model of interest, and request access if necessary. You will also need to create a token for accessing these models from vLLM: open your user profile (https://huggingface.co/settings/profile), select "Access Tokens", press "+ Create New Token", and create a new Read token.

### System validation

Before running performance tests you should ensure the system is validated according to the [Instinct Documentation](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/).  In particular, it is important to ensure that NUMA auto-balancing is disabled.

*Note: Check that NUMA balancing is properly set by inspecting the output of the command below, which should have a value of 0, with, `cat /proc/sys/kernel/numa_balancing`*

### Launch AMD vLLM Docker

Download and launch the docker.  The HF_TOKEN is required to be set (either here or after launching the container) if you want to allow vLLM to download gated models automatically; use your HuggingFace token in place of `<token>` in the command below:

```bash
docker run -it --rm --runtime amd \
  --ipc=host --network=host \
  --security-opt seccomp=unconfined \
  -e AMD_VISIBLE_DEVICES=all \
  -e HF_HOME=/data \
  -e HF_TOKEN=<token> \
  -v /data:/data \
  rocm/vllm-dev:main
```

Note: The instructions in this document use `/data` to store the models.  If you choose a different directory, you will also need to make that change to the host volume mount when launching the docker container.  For example, `-v /home/username/models:/data` in place of `-v /data:/data` would store the models in /home/username/models on the host.  Some models can be quite large; please ensure that you have sufficient disk space prior to downloading the model.  Since the model download may take a long time, you can use `tmux` or `screen` to avoid getting disconnected.

### Downloading models with huggingface-cli

If you would like want to download models directly (instead of allowing vLLM to download them automatically), you can use the huggingface-cli inside the running docker container. (remove an extra white space) Login using the token that you created earlier. (Note, it is not necessary to save it as a git credential.)

```bash
huggingface-cli login
```

You can download a model to the huggingface-cache directory using a command similar to the following (substituting the name of the model you wish to download):

```bash
sudo mkdir -p /data/huggingface-cache
sudo chmod -R a+w /data/huggingface-cache
HF_HOME=/data/huggingface-cache huggingface-cli download meta-llama/Llama-3.1-405B-Instruct --exclude "original/*"
```

In the benchmark commands provided later in this document, replace the model name (e.g. `amd/Llama-3.1-405B-Instruct-FP8-KV`) with the path to the model (e.g. `/data/llama-3.1/Llama-3.1-405B-Instruct`)

### Use pre-quantized models

AMD has provided [FP8-quantized versions](https://huggingface.co/collections/amd/quark-quantized-ocp-fp8-models-66db7936d18fcbaf95d4405c) of several models in order to make them easier to run on MI300X / MI325X, including:

- <https://huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV>
- <https://huggingface.co/amd/Llama-3.1-70B-Instruct-FP8-KV>
- <https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV>

Some models may be private to those who are members of <https://huggingface.co/amd>.

## Performance testing with AMD vLLM Docker

vLLM provides a number of engine options which can be changed to improve performance.  Refer to the [vLLM Engine Args](https://docs.vllm.ai/en/stable/usage/engine_args.html) documentation for the complete list of vLLM engine options.

Below is a list of a few of the key vLLM engine arguments for performance; these can be passed to the vLLM benchmark scripts:

- **--max-model-len** : Maximum context length supported by the model instance. Can be set to a lower value than model configuration value to improve performance and gpu memory utilization.
- **--max-num-batched-tokens** : The maximum prefill size, i.e., how many prompt tokens can be packed together in a single prefill. Set to a higher value to improve prefill performance at the cost of higher gpu memory utilization. 65536 works well for LLama models.
- **--max-num-seqs** : The maximum decode batch size (default 256). Using larger values will allow more prompts to be processed concurrently, resulting in increased throughput (possibly at the expense of higher latency).  If the value is too large, there may not be enough GPU memory for the KV cache, resulting in requests getting preempted.  The optimal value will depend on the GPU memory, model size, and maximum context length.
- **--max-seq-len-to-capture** : Maximum sequence length for which Hip-graphs are captured and utilized. It's recommended to use Hip-graphs for the best decode performance. The default value of this parameter is 8K, which is lower than the large context lengths supported by recent models such as LLama. Set this parameter to max-model-len or maximum context length supported by the model for best performance.
- **--gpu-memory-utilization** : The ratio of GPU memory reserved by a vLLM instance. Default value is 0.9.  Increasing the value (potentially as high as 0.99) will increase the amount of memory available for KV cache.  When running in graph mode (i.e. not using `--enforce-eager`), it may be necessary to use a slightly smaller value of 0.92 - 0.95 to ensure adequate memory is available for the HIP graph.

### Latency Benchmark

vLLM's `benchmark_latency.py` script measures end-to-end latency for a specified model, input/output length, and batch size.

You can run latency tests for FP8 models with:

```bash
MODEL=amd/Llama-3.1-405B-Instruct-FP8-KV
BS=1
IN=128
OUT=2048
TP=8

python3 /app/vllm/benchmarks/benchmark_latency.py \
    --distributed-executor-backend mp \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --batch-size $BS \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-iters-warmup 3 \
    --num-iters 5

```

When measuring models with long context lengths, performance may improve by setting `--max-model-len` to a smaller value.  It is important, however, to ensure that the `--max-model-len` is at least as large as the IN + OUT token counts.

To estimate Time To First Token (TTFT) with the benchmark_latency.py script, set the OUT to 1 token.  It is also recommended to use `--enforce-eager` to get a more accurate measurement of the time that it actually takes to generate the first token.  (For a more comprehensive measurement of TTFT, use the Online Serving Benchmark.)

For additional information about the available parameters run:

```bash
/app/vllm/benchmarks/benchmark_latency.py -h
```

### Throughput Benchmark

vLLM's benchmark_throughput.py script measures offline throughput.  It can either use an input dataset or random prompts with fixed input/output lengths.

You can run throughput tests for FP8 models with:

```bash
MODEL=amd/Llama-3.1-405B-Instruct-FP8-KV
IN=128
OUT=2048
TP=8
PROMPTS=1500
MAX_NUM_SEQS=1500

python3 /app/vllm/benchmarks/benchmark_throughput.py \
    --distributed-executor-backend mp \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --disable-detokenize \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --max-model-len 8192 \
    --max-num-batched-tokens 131072 \
    --max-seq-len-to-capture 131072 \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-prompts $PROMPTS \
    --max-num-seqs $MAX_NUM_SEQS
```

For FP16 models, remove `--kv-cache-dtype fp8`.

When measuring models with long context lengths, performance may improve by setting `--max-model-len` to a smaller value (8192 in this example).  It is important, however, to ensure that the `--max-model-len` is at least as large as the IN + OUT token counts.

It is important to tune vLLM’s `--max-num-seqs` value to an appropriate value depending on the model and input/output lengths.  Larger values will allow vLLM to leverage more of the GPU memory for KV Cache and process more prompts concurrently.  But if the value is too large, the KV cache will reach its capacity and vLLM will have to cancel and re-process some prompts.  Suggested values for various models and configurations are listed below.

For models that fit on a single GPU, it is usually best to run with `--tensor-parallel-size 1`.  Requests can be distributed across multiple copies of vLLM running on different GPUs.  This will be more efficient than running a single copy of the model with `--tensor-parallel-size 8`.  (Note: the benchmark_throughput.py script does not include direct support for using multiple copies of vLLM)

For optimal performance, the PROMPTS value should be a multiple of the MAX_NUM_SEQS value -- for example, if MAX_NUM_SEQS=1500 then the PROMPTS value could be 1500, 3000, etc.  If PROMPTS is smaller than MAX_NUM_SEQS then there won’t be enough prompts for vLLM to maximize concurrency.

For additional information about the available parameters run:

```bash
python3 /app/vllm/benchmarks/benchmark_throughput.py -h
```

### Online Serving Benchmark

Benchmark Llama-3.1-70B with input 4096 tokens, output 512 tokens and tensor parallelism 8 as an example,

```bash
vllm serve amd/Llama-3.1-70B-Instruct-FP8-KV \
    --swap-space 16 \
    --disable-log-requests \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --max-model-len 8192 \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.99 \
    --num_scheduler-steps 10
```

For FP16 models, remove `--kv-cache-dtype fp8`. Change port (for example --port 8005) if port=8000 is currently being used by other processes.

Run client in a separate terminal. Use port_id from previous step else port-id=8000.

```bash
python /app/vllm/benchmarks/benchmark_serving.py \
    --port 8000 \
    --model amd/Llama-3.1-70B-Instruct-FP8-KV \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 512 \
    --request-rate 1 \
    --ignore-eos \
    --num-prompts 500 \
    --percentile-metrics ttft,tpot,itl,e2el
```

Once all prompts are processed, terminate the server gracefully (ctrl+c).

### Running DeepSeek-V3 and DeepSeek-R1

We have experimental support for running both DeepSeek-V3 and DeepSeek-R1 models.
*Note there are currently limitations and `--max-model-len` cannot be greater than 32768*

```bash
docker run -it --rm --ipc=host --network=host --group-add render \
    --privileged --security-opt seccomp=unconfined \
    --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
    --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    -e VLLM_USE_TRITON_FLASH_ATTN=1 \
    -e VLLM_ROCM_USE_AITER=1 \
    -e  VLLM_MLA_DISABLE=0 \
    rocm/vllm-dev:main

# Online serving
vllm serve deepseek-ai/DeepSeek-V3 \
    --disable-log-requests \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 131072 \
    --block-size=1 

python3 /app/vllm/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model deepseek-ai/DeepSeek-V3 \
    --max-concurrency 256\
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 1000

# Offline throughput 
python3 /app/vllm/benchmarks/benchmark_throughput.py --model deepseek-ai/DeepSeek-V3 \
    --input-len <> --output-len <> --tensor-parallel-size 8 \
    --quantization fp8 --kv-cache-dtype fp8 --dtype float16 \
    --max-model-len 32768 --block-size=1 --trust-remote-code

# Offline Latency
python /app/vllm/benchmarks/benchmark_latency.py --model deepseek-ai/DeepSeek-V3 \
--tensor-parallel-size 8 --trust-remote-code --max-model-len 32768 --block-size=1 \
--batch-size <> --input-len <> --output-len <>
```

### CPX mode

Currently only CPX-NPS1 mode is supported. So ONLY tp=1 is supported in CPX mode.
But multiple instances can be started simultaneously (if needed) in CPX-NPS1 mode.

Set GPUs in CPX mode with:

```bash
amd-smi set -C cpx
```

Example of running Llama3.1-8B on 1 CPX-NPS1 GPU with input 4096 and output 512. As mentioned above, tp=1.

```bash
HIP_VISIBLE_DEVICES=0 \
python3 /app/vllm/benchmarks/benchmark_throughput.py \
    --max-model-len 4608 \
    --num-scheduler-steps 10 \
    --num-prompts 100 \
    --model amd/Llama-3.1-8B-Instruct-FP8-KV \
    --input-len 4096 \
    --output-len 512 \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --output-json <path/to/output.json> \
    --quantization fp8 \
    --gpu-memory-utilization 0.99
```

Set GPU to SPX mode.

```bash
amd-smi set -C spx
```

### Speculative Decoding

Speculative decoding is one of the key features in vLLM. It has been supported on MI300. Here below is an example of the performance benchmark w/wo speculative decoding for Llama 3.1 405B with Llama 3.1 8B as the draft model.

Without Speculative Decoding -

```bash
python /app/vllm/benchmarks/benchmark_latency.py --model amd/Llama-3.1-405B-Instruct-FP8-KV --max-model-len 26720 -tp 8 --batch-size 1 --input-len 1024 --output-len 128
```

With Speculative Decoding -

```bash
python /app/vllm/benchmarks/benchmark_latency.py --model amd/Llama-3.1-405B-Instruct-FP8-KV --max-model-len 26720 -tp 8 --batch-size 1 --input-len 1024 --output-len 128 --speculative-model amd/Llama-3.1-8B-Instruct-FP8-KV --num-speculative-tokens 5
```

You should see some performance improvement about the e2e latency.

### AITER use cases

`rocm/vllm-dev:main` image has experimental [AITER](https://github.com/ROCm/aiter) support, and can yield siginficant performance increase for some model/input/output/batch size configurations. To enable the feature make sure the following environment is set: `VLLM_ROCM_USE_AITER=1`, the default value is `0`. When building your own image follow the [Docker build steps](#Docker-manifest) using the [aiter_integration_final](https://github.com/ROCm/vllm/tree/aiter_integration_final) branch.

Some use cases include:

- amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV
- amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0
python3 /app/vllm/benchmarks/benchmark_latency.py --model amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV -tp 8 --batch-size 256 --input-len 128 --output-len 2048
```

## Performance Results

The data in the following tables is a reference point to help users validate observed performance. It should not be considered as the peak performance that can be delivered by AMD Instinct™ MI300X accelerator with vLLM. See the MLPerf section in this document for information about MLPerf 4.1 inference results. The performance numbers above were collected using the steps below.
*Note Benchmarks were run with benchmark scripts from [v0.8.5](https://github.com/vllm-project/vllm/tree/v0.8.5/benchmarks)*

### Throughput Measurements

The table below shows performance data where a local inference client is fed requests at an infinite rate and shows the throughput client-server scenario under maximum load.

| Model | Precision | TP Size | Input | Output | Num Prompts | Max Num Seqs | Throughput (tokens/s) |
|-------|-----------|---------|-------|--------|-------------|--------------|-----------------------|
| Llama 3.1 70B (amd/Llama-3.1-70B-Instruct-FP8-KV) | FP8 | 8 | 128 | 2048 | 3200 | 3200 | 13818.7  |
|       |           |         | 128   | 4096   | 1500        | 1500         | 11612.0               |
|       |           |         | 500   | 2000   | 2000        | 2000         | 11408.7               |
|       |           |         | 2048  | 2048   | 1500        | 1500         | 7800.5                |
| Llama 3.1 405B (amd/Llama-3.1-405B-Instruct-FP8-KV) | FP8 | 8 | 128 | 2048 | 1500 | 1500 | 4134.0 |
|       |           |         | 128   | 4096   | 1500        | 1500         | 3177.6                |
|       |           |         | 500   | 2000   | 2000        | 2000         | 3034.1                |
|       |           |         | 2048  | 2048   | 500         | 500          | 2214.2                |

### Latency Measurements

The table below shows latency measurement, which typically involves assessing the time from when the system receives an input to when the model produces a result.

| Model | Precision | TP Size | Batch Size | Input | Output | MI300X Latency (sec) |
|-------|-----------|----------|------------|--------|---------|-------------------|
| Llama 3.1 70B (amd/Llama-3.1-70B-Instruct-FP8-KV) | FP8 | 8 | 1 | 128 | 2048 | 15.254 |
| | | | 2 | 128 | 2048 | 18.157 |
| | | | 4 | 128 | 2048 | 18.549 |
| | | | 8 | 128 | 2048 | 20.547 |
| | | | 16 | 128 | 2048 | 22.164 |
| | | | 32 | 128 | 2048 | 25.426 |
| | | | 64 | 128 | 2048 | 33.297 |
| | | | 128 | 128 | 2048 | 45.792 |
| | | | 1 | 2048 | 2048 | 15.299 |
| | | | 2 | 2048 | 2048 | 18.194 |
| | | | 4 | 2048 | 2048 | 18.942 |
| | | | 8 | 2048 | 2048 | 20.526 |
| | | | 16 | 2048 | 2048 | 23.211 |
| | | | 32 | 2048 | 2048 | 26.516 |
| | | | 64 | 2048 | 2048 | 34.824 |
| | | | 128 | 2048 | 2048 | 52.211 |
| Llama 3.1 405B (amd/Llama-3.1-405B-Instruct-FP8-KV) | FP8 | 8 | 1 | 128 | 2048 | 47.150 |
| | | | 2 | 128 | 2048 | 50.933 |
| | | | 4 | 128 | 2048 | 52.521 |
| | | | 8 | 128 | 2048 | 55.233 |
| | | | 16 | 128 | 2048 | 59.065 |
| | | | 32 | 128 | 2048 | 68.786 |
| | | | 64 | 128 | 2048 | 88.094 |
| | | | 128 | 128 | 2048 | 118.512 |
| | | | 1 | 2048 | 2048 | 47.675 |
| | | | 2 | 2048 | 2048 | 50.788 |
| | | | 4 | 2048 | 2048 | 52.405 |
| | | | 8 | 2048 | 2048 | 55.459 |
| | | | 16 | 2048 | 2048 | 59.923 |
| | | | 32 | 2048 | 2048 | 70.388 |
| | | | 64 | 2048 | 2048 | 91.218 |
| | | | 128 | 2048 | 2048 | 127.004 |
