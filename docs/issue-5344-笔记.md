# 我在复现 LMCache #5344

issue：<https://github.com/LMCache/LMCache/issues/5344>

这篇就是跟手记。做完哪一步就往对应小节里补几句，命令也留在那一节。长日志别整段贴，关键几行够了。

我想先看清一件事：TP=2、L1 故意弄得很小时，同一轮里有的 rank 打 `Stored`，有的打 `Failed to batched allocate`。然后再看 worker0 还会不会把这条缺分片的前缀发出去。

## 进度

| 小节 | 状态 |
| --- | --- |
| 1. 问题是什么 | 已写 |
| 2. 别人做到哪 | 已写 |
| 3. 机器和原有套件 | 已看，数字是装环境之前的 |
| 4.1 装 LMCache | 虚拟环境已建好，import 输出还没补 |
| 4.2 装 vLLM | 已装上。vLLM 0.31.0，torch 换成 2.13.0+cu130 |
| 5. 把复现条件定下来 | 模型已落地。L1 必须关掉懒分配 |
| 6. 跑出撕裂的存储日志 | 已跑出。一边 Stored 3840，另一边 15 块整批分配失败 |
| 7. 看缺分片之后读起来会怎样 | 已看。第二次没有 Retrieved，缺的那张卡又失败，还是差 10 块 |
| 8. 和 issue 原文对一下 | 已对。存储侧对得上，乱码那半截这次没出现 |
| 9. 跨卡提交怎么改 | 已改一版。有卡失败时不再留下 `Stored`，2026-10-13 在这台机器上试过 |
| 10. 想跟 maintainer 确认的事 | 已写。方案落点、和 #5349 怎么叠、几处边角 |

## 1. 问题是什么

多卡张量并行时，每张卡只存自己那份 KV，大家共用一个 L1。

某一轮要一次申请很多块，空间不够时 `batched_allocate` 整批失败。失败是各卡自己判断的：有的卡申请成功，日志是 `Stored ... tokens`；有的卡失败，日志是 `Failed to batched allocate ...`。

更麻烦的是提交。worker0 发 `MP_TOKENS`，意思是这个前缀各卡都写好了，但它并没有等每一张卡都 `finish_write`。索引里已经有这条前缀，实际上却缺至少一张卡的分片。后面再读，引擎可能当成命中。

原报告用的是 vLLM 0.28.0、LMCache 0.5.4rc5、TP=4、chunk 1600、L1 32GiB。我这边只有两张卡，先把「一部分 Stored、一部分分配失败」这句日志跑出来。

## 2. 别人做到哪

- PR [#5349](https://github.com/LMCache/LMCache/pull/5349) 还开着，没合并。它只修了单卡：这一卡自己分配失败时，把已经暂存的对象回滚。没有跨卡协商。我这次先不把这个 PR 合进来，看没修过的 `dev`。
- 2026-10-09 有人在 issue 里对过代码，觉得「有的卡写入、有的卡失败」在当时的 main 上还在。他还觉得查找会拒绝缺分片的 chunk，所以后面的乱码未必是这次撕裂直接读出来的。第 7 节的结果跟这个判断一致：撕裂留在 L1 里，第二次请求没有 `Retrieved`。
- 2026-10-12 在 Slack 里，ppilgrim 说单卡修复已经做完，跨卡方案 maintainer 还没定。他补充了一句现状：`reserve_write` 会把 OOM 以外的信息过滤掉，他加了追踪把这些信息返回来，但这一步仍然只覆盖单个 TP rank。

## 3. 机器和原有套件

2026-10-12 看的，当时还没装 LMCache 和 vLLM。

口头说的是 2× RTX 6000D 84GB、内存大约 220GB。实际是：

| 项目 | 实际 |
| --- | --- |
| GPU | 2 张 NVIDIA RTX PRO 6000 Blackwell Server Edition |
| 单卡显存 | 97887 MiB，当时占用 0 MiB，没有计算进程 |
| 驱动 / CUDA | 580.95.05 / 13.0，计算能力 (12, 0) |
| 内存 | 1.0 TiB，当时可用大约 866 GiB |
| 系统盘 `/` | overlay 30G，当时可写层大约用了 54M |
| 数据盘 `/root/autodl-tmp` | 50G，当时大约用了 439M |
| 系统 | Ubuntu 22.04.5，AutoDL 基础镜像，区域 west-D |
| Python | miniconda 3.12.3，conda 24.4.0，`/root/miniconda3` |
| 编译器 | gcc、g++、cmake、git 都有。`nvcc` 在 `/usr/local/cuda/bin/nvcc` |

当时显卡利用率显示 100%，显存是 0，也没有进程。不当成有人在占卡。

镜像自带的套件：

| 套件 | 版本 |
| --- | --- |
| CUDA Toolkit | 13.0，`nvcc` V13.0.88，`/usr/local/cuda` |
| 系统 cuDNN / NCCL | 9.14.0.64 / 2.28.3 |
| torch / torchvision / triton | 2.12.1+cu130 / 0.27.1+cu130 / 3.7.1 |
| pip 里的 cuDNN / NCCL / cuBLAS | 9.20.0.48 / 2.29.7 / 13.1.1.3 |
| modelscope / tensorboard / jupyterlab / numpy | 1.40.2 / 2.20.0 / 4.5.9 / 2.4.6 |
| Nsight Compute | 2025.3.1 |
| TensorRT、vLLM、LMCache、transformers、flash-attn、SGLang | 当时都没有 |

系统里一套 CUDA，Python 里又有一套随 torch 装进来的库，版本不完全一样。跑 LMCache 时，Python 加载的是 pip 那一套。

磁盘后面要紧：系统盘可写层只有 30G，虚拟环境和数据放 `/root/autodl-tmp`。torch 已经在系统里，虚拟环境用 `--system-site-packages` 接着用。

裸跑 `df -h` 看不到数据盘。`/dev/md0` 同时挂在 `/init` 和 `/root/autodl-tmp` 上，`df` 只打印第一个挂载点，那一行的 150M 是 `/init` 的配额，不是数据盘。`/usr/bin/nvidia-smi` 那条 879G 是宿主机驱动盘，只读挂了几个 NVIDIA 文件，不能拿来装包。看数据盘用 `df -h /root/autodl-tmp`。2026-10-12 15:39 系统盘 overlay 用了 124M，还没被安装撑满。

## 4. 环境

源码在 `/root/autodl-tmp/src/LMCache`，分支 `dev`，提交 `160b6de`。这是还没合入 #5349 的版本。

### 4.1 装 LMCache

这次 `pip` 安装使用华为云。

```bash
mkdir -p /root/autodl-tmp/{tmp,pip-cache,logs}
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export TORCH_CUDA_ARCH_LIST=12.0
unset PIP_INDEX_URL PIP_TRUSTED_HOST PIP_EXTRA_INDEX_URL

/root/miniconda3/bin/python -m venv /root/autodl-tmp/lmcache-venv --system-site-packages
source /root/autodl-tmp/lmcache-venv/bin/activate
python -m pip install -U "pip" "setuptools>=77,<81" wheel ninja packaging "setuptools_scm>=8" \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  --trusted-host repo.huaweicloud.com

cd /root/autodl-tmp/src/LMCache
pip install -e . --no-build-isolation \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  --trusted-host repo.huaweicloud.com
python -c "import torch, lmcache; print('torch', torch.__version__); print('lmcache', lmcache.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
lmcache server --help | grep -n -i -E "l1|chunk|evict"
df -h / /root/autodl-tmp
```

虚拟环境在 `/root/autodl-tmp/lmcache-venv`，LMCache 是可编辑安装。版本字符串是 `0.1.dev1`，因为这次检出没有发行标签，提交仍是 `160b6de`。

还没补上的输出：上面那三行 import、`l1` / `chunk` / `evict` 的帮助、装完后的 `df -h`。

### 4.2 装 vLLM

同一个虚拟环境。MP 模式要用 vLLM 0.20 及以上，并且用 `kv_connector_module_path` 指向 LMCache 自带的 connector。这次 `pip install` 使用华为云。安装如果要换掉 torch 2.12.1，先停，把冲突记在这里。

```bash
source /root/autodl-tmp/lmcache-venv/bin/activate
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export TORCH_CUDA_ARCH_LIST=12.0
unset PIP_INDEX_URL PIP_TRUSTED_HOST PIP_EXTRA_INDEX_URL

pip install vllm \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  --trusted-host repo.huaweicloud.com
python -c "import torch, lmcache, vllm; print('torch', torch.__version__); print('lmcache', lmcache.__version__); print('vllm', vllm.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
df -h /root/autodl-tmp
```

结果：

安装完成。虚拟环境里现在是：

| 包 | 版本 |
| --- | --- |
| vllm | 0.31.0 |
| torch | 2.13.0+cu130（装 vLLM 时从原来的 2.12.1+cu130 换上来的，CUDA 仍是 13.0） |
| torchvision | 0.28.0 |
| transformers | 5.17.0 |
| flashinfer-python | 0.7.0.post1 |

系统里原来的 torchvision 卸不掉（`Can't uninstall 'torchvision'. No files were found to uninstall.`），新的 0.28.0 装进了虚拟环境，安装没有因此停住。`df -h /root/autodl-tmp` 还没补。

## 5. 把复现条件定下来

先读代码和 `lmcache server --help`，把这几件事写死，再起服务：

- L1 容量、水位、一次申请多少块，分别由哪个参数控制
- 一条日志里的 `Stored` 和 `Failed to batched allocate` 各从哪段代码打出来
- `MP_TOKENS` 由谁发送，发送前有没有看过其他 rank 的写入结果
- 两张卡的分片是不是进同一个 L1。要复现「一个成功、一个失败」，池子得是共享的，而且装得下一边、装不下两边

模型用 `Qwen/Qwen2.5-0.5B-Instruct`。它有 14 个 attention head、2 个 KV head，TP=2 能整除。

`--l1-use-lazy` 默认开着，`--l1-init-size-gb` 默认 20。只把 `--l1-size-gb` 写很小没用，池子一开始就有 20GB。复现时要加 `--no-l1-use-lazy`，让 `--l1-size-gb` 就是整个池子。

模型从 Hugging Face 下。下模型的这个终端先打开学术加速，再登录。令牌不要写进笔记。这个终端先不要跑 `pip`。

```bash
source /etc/network_turbo
source /root/autodl-tmp/lmcache-venv/bin/activate
hf auth login
hf auth whoami
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /root/autodl-tmp/models/Qwen2.5-0.5B-Instruct
df -h /root/autodl-tmp
```

模型已经在 `/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct`。`model.safetensors` 988097824 字节（943M），`config.json` 是 Qwen2，24 层，2 个 KV head。数据盘 `/root/autodl-tmp` 用了 8.6G，还剩 42G。

启动命令在下一节。

## 6. 跑出撕裂的存储日志

就想看一件事：同一次写入里，一个 rank 是 `Stored`，另一个是 `Failed to batched allocate`。

这个模型每个 token、每个 rank 的 KV 大约 6KB。chunk 用 256，一条 4096 token 的请求大约是 16 个 chunk，一边大约 24MB，两边一起大约 48MB。L1 设成 0.03GB（大约 31MB），并且关掉懒分配，池子只够一边整批申请成功。

两个终端都要先进入虚拟环境，并且不要开学术加速。

终端 A，LMCache server：

```bash
source /root/autodl-tmp/lmcache-venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
mkdir -p /root/autodl-tmp/logs
lmcache server \
  --host localhost --port 5555 \
  --l1-size-gb 0.03 \
  --no-l1-use-lazy \
  --eviction-policy LRU \
  --chunk-size 256 \
  2>&1 | tee /root/autodl-tmp/logs/lmcache-server.log
```

终端 B，等 server 起来后再开 vLLM：

```bash
source /root/autodl-tmp/lmcache-venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
vllm serve /root/autodl-tmp/models/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen25-0.5b \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.2 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}'
```

终端 C，等 vLLM 打出启动完成后再发请求。提示词重复很多次，凑够大约 4096 个 token：

```bash
python3 - << 'PY' > /tmp/prompt.txt
print("The quick brown fox jumps over the lazy dog. " * 400)
PY
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "$(python3 - << 'PY'
import json
print(json.dumps({
  "model": "qwen25-0.5b",
  "prompt": open("/tmp/prompt.txt").read(),
  "max_tokens": 16,
  "temperature": 0
}))
PY
)"
```

然后在 server 日志里找这两类行：`Stored` 和 `Failed to batched allocate`。两边都失败就把 `--l1-size-gb` 调到 0.05 再发一次；两边都成功就调到 0.02。

vLLM 在 2026-10-12 16:49:41 启动完成，地址 `http://0.0.0.0:8000`。进程 16:45:13 起来，引擎初始化这 235.71 秒从 16:45:43 算到 16:49:39。16:46:33 打出 `JIT kernel warmup finished in 0.00s`，同时 KV cache 已经注册。所以这 235.71 秒分成两段：前大约 50 秒是测显存、建 KV cache，torch 编译的 10.51 秒在这段里面；后大约 186 秒是这条 warmup 之后到初始化结束，当时 CPU 在用 `nvcc` 给 Blackwell（`sm_120f`）编 FlashInfer 采样内核，GPU 利用率为 0。LMCache 连接器已接上，版本 `0.1.dev1`（`g160b6dec`），vLLM 的 block size 是 16。

2026-10-12 16:52:19 发了上面那条请求。接口返回正常：`prompt_tokens` 4001，补了 16 个 token，`finish_reason` 是 `length`。server 日志里同一毫秒内只有这两行：

```text
Stored 3840 tokens in 0.105 seconds
Failed to batched allocate 15 memory blocks of size 1572864 because no enough memory is available (short by 10 blocks)
```

4001 个 token 按 chunk 256 切，完整 chunk 是 15 个，也就是 3840 个 token，剩下 161 个不满一块，本来就不会存。一块 1572864 字节，大约 1.5MB，15 块大约 22.5MB。L1 是 0.03GB，大约 31MB，第一张卡放进去之后只剩大约 5 块，第二张卡再要 15 块，正好差 10 块。每个 token、每个 rank 实际是 6144 字节，和前面 6KB 的估计一致。

日志没有写是哪张卡成功。两张卡的任务被排到同一条线程上，先打出 `Stored`，1 毫秒后另一张卡分配失败。失败那一侧没有再打异常栈：分配函数把 `RuntimeError` 吃掉，返回空，所以这一侧没有 `finish_write`，也没有第二条 `Stored`。成功那一侧的 3840 个 token 已经提交。server 和 vLLM 都还开着，先不要重启。

## 7. 看缺分片之后读起来会怎样

存储侧的撕裂如果出现了，接着看两件事：

- worker0 有没有仍然发出这条前缀的 `MP_TOKENS`
- 再发一条能命中这条前缀的请求。查找是把它当成命中，还是因为缺了一张卡的分片而当成未命中

第 2 节里那条代码分析认为查找会拒绝缺分片的 chunk。这里用这次跑出来的日志核对。

2026-10-12 17:00:04 用同一条提示词又发了一次。接口还是 4001 个 prompt token、补 16 个 token，生成的文字和第一次一样。temperature 是 0，同样的提示词本来就会生成同样的字，这不能当成命中。

server 日志这一次只有一行：

```text
Failed to batched allocate 15 memory blocks of size 1572864 because no enough memory is available (short by 10 blocks)
```

没有 `Retrieved`，也没有新的 `Stored`。还是差 10 块，说明第一次成功的那 15 块还占着，池子没有被清掉。已经写进去的那张卡这次没有再申请：它的 key 已经在 L1 里。没写进去的那张卡又要 15 块，还是失败，所以仍然没有提交。

`MP_TOKENS` 这次日志里看不到。现在的代码是 worker0 在真正分配之前就发这个事件，而且只有有人订阅才发。这次没有开订阅，日志不会打印它。查找并不靠这个事件，它看的是每一张卡的对象在不在。缺任何一张卡，这个 chunk 就不算命中，所以不会去读。这次没有 `Retrieved`，和这个行为一致。

## 8. 和 issue 原文对一下

存储侧这边对得上。同一轮里一张卡 `Stored 3840 tokens`，另一张卡 `Failed to batched allocate`，失败的那张卡也没有把对象提交。

原文后半段我们对不上。原文是大约九分钟后再读，四个 rank 都打了 `Retrieved`，输出变成乱码。我们隔了大约八分钟再发同一条，没有 `Retrieved`，输出也没变。这次看到的是：撕裂的前缀留在 L1 里，但查找因为它缺了一张卡的分片，没有把它读回去。乱码那件事这次没有复现。

乱码我不在这台机器上继续追。原文那次 `Retrieved 24000 tokens` 配的是 `60/64 retained keys`：24000 / 1600 = 15 个 chunk，16×4 = 64 个 key 里留了 60 个，对应的是 15 个 chunk 在四张卡上都在。那不是九分钟前两张卡有、两张卡没有的那 71 个 chunk。0xRyanC 把乱码指到 #4984：hybrid Mamba/GDN 加推测解码，0.5.4rc5 上对一条完整前缀命中会输出垃圾，#5004 修了，0.5.6rc3 才带上。我们是 Qwen2.5-0.5B、TP=2、当前 `dev`，没有这条路径。这台机器也只有两张卡，做不了原文的 TP=4。

## 9. 跨卡提交怎么改

要修的其实就一件事：这一轮里有的卡把 chunk 写进 L1 了，有的卡没写进去。写进去的那份又读不回来，就一直占着空间。乱码是 #4984，这边不管。#5349 只丢掉失败那张卡自己的暂存，成功的那张卡它碰不到。

我打算怎么改：一个 chunk 先别急着提交，每张卡先只暂存。大家都准备好了，再一起 `finish_write`。只要有一张卡这轮没准备好，所有卡都把这个 chunk 新占的缓冲区放开。以前已经写进 L1 的旧数据，这次请求不回头删，等驱逐回收就行。

确认放在 LMCache server 上。这些卡本来就连着同一个 server。要是放到 worker 之间做集合通信，某张卡没走进 `store()`，别的卡就得在那儿干等。

每张卡的 `store()` 还是先申请、再把 KV 从 GPU 拷到暂存。拷贝没完之前别释放这块内存，不然一边还在写、一边缓冲区已经清了。拷贝完了再对结果。默认只有一条 GPU 工作线程，两张卡的 `store()` 会前后执行，所以不能在 `store()` 里面干等另一张卡，否则后一张永远进不来。

失败也得分开看。内存不够，算这张卡这个 chunk 没准备好。key 本来就在，算准备好，不用再占一块新的。这个 chunk 是空的、故意不存，也算准备好。

`MP_TOKENS` 只给大家都同意的 chunk 发，而且要发在 `finish_write` 之前。缓存事件得先看到 token 内容，再看到写完成。某张卡 5 秒还没来，就把已经到了的缓冲区放开，调用方照常返回。水位不够就提前拒绝，只能让这种失败少发生，这次先不做。

代码还没提交。单卡（`world_size == 1`）还走原来的路径，不进这张表。

动到的文件大概是这些：

- `lmcache/v1/multiprocess/cross_rank_store.py`：新建的。按 `(request_id, start, end)` 把每张卡的结果收齐，都到了再决定提交还是放开。超时、晚到的卡也走放开。
- `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`：`store()` 里两张卡以上改走这张表。申请时把每种 key 的错误码留下来。拷贝完用 `cross_rank_store_settle` 回调回来。worker0 只发布同意了的 chunk。
- `lmcache/v1/distributed/storage_manager.py`：`reserve_write_with_status` 把 `L1Error` 留下来，好区分内存不够和 key 已在。`abort_write_by_owner` 丢掉暂存，不让它变成可读对象。
- `tests/v1/multiprocess/test_cross_rank_store.py`：不启动 GPU，只测上面那张表。

多卡时 `store()` 返回给调用方的 chunk 掩码先是空的。这次用的是 default engine，不走 CacheBlend，这样 CacheBlend 不会把后来又被丢掉的 chunk 登记成指纹。

测试这边，不碰 GPU 的五个例子已经过了：两边都准备好就一起提交；一边没准备好就丢掉另一边的缓冲区；拷贝没结束不能提交；只来了一张卡、超时后缓冲区被丢掉；key 已经在、没有新缓冲区，也算这张卡准备好。

这台机器上，2026-10-13 11:46:12，用原来那条 4001 token 的请求跑过。L1 还是 0.03GB，TP=2。接口正常返回。server 日志是：

```text
Failed to batched allocate 15 memory blocks of size 1572864 (short by 10 blocks)
Cross-rank store released request cmpl-903a93e7574b3f03-0-81adf67d tokens [0, 3840): a rank was not ready
```

没有 `Stored`。申请失败还会出现，失败之后另一张卡这次占的缓冲区被放开了。紧接着用同一条提示词又发了一次，LMCache 没有新的存储日志，第二次走的是 GPU 上刚算出来的前缀。

2026-10-13 11:58 又跑了两边都放得下的情况。server 和 vLLM 先停掉，再用 `--l1-size-gb 0.08` 开起来，同一条 4001 token 的请求。日志是：

```text
Cross-rank store committing request cmpl-9001a913d1cabfe4-0-8771cea3 tokens [0, 3840): 15/15 chunks agreed
Stored 3840 tokens in 0.161 seconds
Stored 3840 tokens in 0.053 seconds
```

没有分配失败。15 个 chunk 两边都同意了，两张卡都打了 `Stored`。

为了不让 GPU 前缀把读请求接走，只重启了 vLLM，LMCache server 接着开着。11:59:55 再发同一条提示词：

```text
Prefetch request completed (L1+L2): 30/30 retained keys (30 L1, 0 L2) in 2.4 ms
Retrieved 3840 tokens in 0.043 seconds
Retrieved 3840 tokens in 0.013 seconds
```

30 个 key 就是 15 个 chunk × 两张卡，两边都 `Retrieved 3840`。刚才写进去的那段能读回来。

超时不在这台机器上停卡来造。单测里已经覆盖：只来一张卡，到点放开它占的缓冲区。

## 10. 想跟 maintainer 确认的事

本地复现和试改已经告一段落。失败路径、成功路径、读回来，在 TP=2 上都跑过了。代码还在本地，没提 PR。下面几件事想先跟你们对齐，再决定怎么往上游合。

**1. 跨卡确认放哪**

我这版放在 LMCache server 上：按 `(request_id, start, end)` 收齐每张卡的结果，拷贝完再决定提交还是放开。不在 worker 之间加集合通信，也不在 `store()` 里干等另一张卡——默认只有一条 GPU 工作线程，干等会把自己堵死。

Slack 里说过跨卡方案还没定落点。我想确认：这个放法你们能不能接受？如果希望换地方，我按你们定的位置再改一版。

**2. 和 #5349 怎么叠**

#5349 做的是单卡：失败时丢掉自己的暂存，并把 OOM 以外的错误码留下来。跨卡同意是它外面的一层。我这版里也写了 `reserve_write_with_status` 和 `abort_write_by_owner`，是为了本地能跑通，不是想另起一套。

开 PR 之前想跟 ppilgrim / 你们对一下：是先合 #5349，我再叠跨卡表，还是合成一个 PR 一起看。我不想弄出两份重复实现。

**3. 几个边角，合入前想听一下意见**

- `KEY_NOT_WRITABLE` 现在既表示「key 本来就在」，也可能是「别人正占着写锁」。我这版两种都当成「这张卡准备好了、没有新缓冲区」。对 OOM 撕裂够用，要不要再拆开？
- 多卡时 `store()` 返回的 chunk 掩码先是空的，CacheBlend 不会把后来又丢掉的 chunk 登记成指纹。这次测的是 default engine。blend 要不要另补？
- 超时（某张卡一直不来）只在单测里验过，GPU 上没故意停卡。5 秒这个数、超时后放开已到缓冲区，这个行为你们觉得合适吗？
- 水位不够就提前拒绝，只能让这类失败少发生。我这版没做，要不要跟跨卡同意拆开，后面单独做？

**4. issue 后半段的乱码**

我这边没复现乱码。查找会拒绝缺分片的 chunk，撕裂前缀读不回去。0xRyanC 指到 #4984 那条路径。我这台机器是 Qwen2.5-0.5B、TP=2，追不了原文的 hybrid + TP=4。想确认：#5344 这边是不是先把跨卡提交修干净，乱码仍归 #4984？

本地验证我可以继续补；落点和 PR 形状想先听你们的。
