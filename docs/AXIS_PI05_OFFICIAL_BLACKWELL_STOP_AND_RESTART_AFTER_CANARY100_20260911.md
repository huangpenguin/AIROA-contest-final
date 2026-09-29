# 官方 Blackwell：canary100 后 Stop / 下次快速启动

> 日期：2026-09-11  
> 当前：本机 GPU3 正在跑 T3 N5 48-case；官方机空闲。**Stop 不停止本机评测。**  
> 训练步数仍为 0。未授权 250/300/500。

## 0. Stop 前 30 秒核对（必须全 True）

在官方 Jupyter Shift+Enter：

```python
import os, subprocess
from pathlib import Path

print("train.py", subprocess.run(["pgrep", "-af", "scripts/train.py"], capture_output=True, text=True).stdout.strip() or "(none)")
print(subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv"], text=True))
persist = Path.home() / "data/parc20260911"
for rel in ("t1_p20_canary100/100", "t2_p30_canary100/100", "t3_four_gold_canary100/100", "ckpts/r6/params"):
    p = persist / rel
    ok = (p / "params").is_dir() if rel.endswith("/100") else p.is_dir()
    if rel.endswith("/100"):
        ok = (p / "params").is_dir() and (p / "train_state").is_dir() and (p / "_CHECKPOINT_METADATA").is_file()
    print(rel, ok, p)
print("hf_token", (Path.home() / ".cache/huggingface/token").is_file())
print("netrc", (Path.home() / ".netrc").is_file())
```

全部 checkpoint `True`、`train.py (none)`、GPU `0 MiB` 即可 **Hub Control Panel → Stop My Server**。不要点延長。

Stop 会清 NVMe（`/opt/dlami/nvme/parc20260911/**`）。下列会留下：

| 留下 | 路径 / 仓库 |
|---|---|
| persist canary100 | `~/data/parc20260911/t{1,2,3}_*_canary100/100` |
| persist R6 | `~/data/parc20260911/ckpts/r6` |
| HF T1 | `Huangpenguin/parc2026-t1-p20-canary100-step100` |
| HF T2 | `Huangpenguin/parc2026-t2-p30-canary100-step100` |
| HF T3 | `Huangpenguin/parc2026-t3-four-gold-canary100-step100` |
| HF R6 | `Huangpenguin/parc2026-r6-l2-params-step500` @ `e788d921` |
| runtime 代码包 | `Huangpenguin/parc2026-official-runtime-20260911` @ `836eb98` |
| T3 数据 payload | `Huangpenguin/parc2026-track3-q7-transfer` @ `669789a` |
| HF token / wandb | `~/.cache/huggingface/token`、`~/.netrc` |
| 官方 tar | `~/dataset/*.tar`（只读） |

## 1. 下次 Start 后：先恢复盘，再决定是否训

**默认不开训。** 只有本机 N5 `continue_to_step300_authorized=true` 才允许 T3 从 step100 optimizer 续到总 300。T1/T2 续 250 还要专用面板，当前未冻结。

### Cell A — 配额与空卡

```python
import subprocess
from pathlib import Path
print(subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv"], text=True))
print("persist", (Path.home() / "data/parc20260911").exists())
print("nvme", Path("/opt/dlami/nvme").exists())
```

若 persist 缺 canary100：`parc-home-sync data-pull`（只拉 `data`，不要把大数据写进 `$HOME`）。

### Cell B — NVMe 骨架 + 从 persist 回拷权重

```python
import shutil
from pathlib import Path

nvme = Path("/opt/dlami/nvme/parc20260911")
persist = Path.home() / "data/parc20260911"
for d in ("ckpts/r6", "dataset", "code", "work", "runs", "staging"):
    (nvme / d).mkdir(parents=True, exist_ok=True)

def copy_if_missing(src, dst):
    if dst.exists():
        print("keep", dst)
        return
    print("copy", src, "->", dst)
    shutil.copytree(src, dst)

copy_if_missing(persist / "ckpts/r6", nvme / "ckpts/r6")
for name in ("t1_p20_canary100", "t2_p30_canary100", "t3_four_gold_canary100"):
    copy_if_missing(persist / name / "100", nvme / "runs" / name / "100")
print("done")
```

persist 缺文件时从 HF 补（private model）：

```text
Huangpenguin/parc2026-r6-l2-params-step500                  -> /opt/dlami/nvme/parc20260911/ckpts/r6
Huangpenguin/parc2026-t3-four-gold-canary100-step100        -> .../runs/t3_four_gold_canary100/100
Huangpenguin/parc2026-t1-p20-canary100-step100              -> .../runs/t1_p20_canary100/100
Huangpenguin/parc2026-t2-p30-canary100-step100              -> .../runs/t2_p30_canary100/100
```

### Cell C — 代码与数据（不要 overlay Q7 OpenPI）

```text
runtime  tar/HF     Huangpenguin/parc2026-official-runtime-20260911  commit 836eb98
禁止覆盖            third_party/axis_openpi/scripts/train.py
                    src/openpi/training/config.py
                    src/openpi/training/data_loader.py
T3 payload          Huangpenguin/parc2026-track3-q7-transfer @ 669789a
                    只铺 data/mix/reverse gold，不铺 OpenPI
libero_plus         ~/dataset/Sylvest/libero_plus_lerobot.tar 或已有 NVMe 副本
IPEC                ~/dataset/IPEC-COMMUNITY/*.tar
```

正向数据仍解压到 `/opt/dlami/nvme/.../dataset/`，不要写进 `$HOME`。

### Cell D — 重建 trainvenv（JAX 训练，不要装 LIBERO）

沿用上次已验证栈：Python 3.12 + uv、`jax==0.11.1` + cuda13、`orbax-checkpoint==0.11.25`、`huggingface-hub==0.36.0`、`transformers==4.53.2`、Flax 0.10.2 JAX 0.11 补丁、Auto mesh、`AXIS_PI05_DISABLE_ACTIVATION_SHARDING_CONSTRAINT=1`、lerobot `@0cf8648`。  
**不要** `pip install -e third_party/axis_openpi`，不要往 trainvenv 装 `robosuite`/`libero`。

### Cell E — 续训门（现在不要跑）

```text
禁止  python tools/axis_pi05/launch_official_blackwell.py --steps 250/300/500
允许  仅当本机 paired_summary.json 里对应 continue_*_authorized=true
      才从该赛道 step100 的 train_state 另开 resume 入口
T3    reverse ≥ +1/4 且 C20 下降 ≤ 2/20 → 总 step300（再训 200）
T1/T2 专用面板至少 +2、无净退化、C20 最多 -2/20 → 总 step250
wandb 已在 ~/.netrc；正式长训才 online，canary 保持 disabled
```

每次开训前：`OPENPI_KEEP_STEPS`、日志在 RUN 外、禁止预先 `mkdir` 空 RUN、训完 `parc-home-sync data-push`。

## 2. 本机评测（与 Stop 无关）

```text
T3 N5     PID 以 cursor.json 为准；48 rollouts；GPU3
随后      T1 C20 40 次，再 T2 C20 40 次
专用面板  未冻结，不能凭 C20 授权 T1/T2 step250
```
