# N1 Evaluator / Runtime 非确定性恢复报告

## 结论

恢复执行在 D1 environment-only 阶段停止，当前状态为：

```text
n1_nondeterminism_root_cause_unresolved
```

没有执行 D2 policy-only、D3 integration、D4 N1 恢复，也没有启动主计划的 parity grid、V160/H160、训练、fold/export 或评分。

## D0

Comparator v2 已将 first nonzero diff 与 case max diff 分离，并输出 chunk 起点、common prefix、success flip、termination tuple、steps、failure reason、shape 和 byte mismatch。旧 N1 traces 已生成 legacy reanalysis，未覆盖原证据。GPU0 的 `false` 只有在 monitor evidence 完整时才有效；旧 summary 缺少证据时被标记为 legacy missing。

D0 测试结果为 `89 passed, 7 skipped, 21 warnings`，`py_compile` 和 `git diff --check` 通过。

## D1

| 运行 | 结果 | 首个差异 |
|---|---|---|
| baseline same-process | 2/3 mismatch | task00 E0 simulator `flattened` |
| baseline fresh-process | 2/3 mismatch | task00 E0 simulator `flattened` |
| seeded constructor same-process | 0/3 mismatch | exact |
| seeded constructor fresh-process | 1/3 mismatch | task11 E1 `agentview_image` |

seeded fresh 的唯一失败 case 为 `C20-task11-init0`：simulator state、qpos/qvel、wrist image exact，但 agentview raw uint8 image 有 9 个像素变化，最大绝对差为 1。该差异在 env.reset 后、init state 设置前已经出现，因此不是 policy/action 造成的闭环放大。

尝试 `MUJOCO_GL=osmesa` 时，本机 PyOpenGL/MuJoCo OSMesa binding 返回 `None`，无法获得 CPU renderer 对照；没有用 tolerance 跳过 EGL image 差异。

## 资源与契约

- environment-only probe 未加载 Axis/R6 policy，也未执行 JAX inference。
- 使用仓库内 `LIBERO` import、固定 C20 manifest、native 10 Hz、10-step warm-up、direct init probe、GPU1 EGL。
- `CUDA_VISIBLE_DEVICES=1`、`MUJOCO_EGL_DEVICE_ID=1`、`JAX_PLATFORMS=cpu`、`PYTHONHASHSEED=0`。
- 每个 probe 都有 before/active/after GPU monitor evidence；GPU0 外部 PID `237527` 未触碰，`gpu0_touched=false`。

## 产物

- D1 canonical environment summary：`environment_probe/C20_environment_determinism.json`
- baseline summary：`environment_probe/baseline_summary.json`
- seeded constructor summary：`environment_probe/seeded_constructor_summary.json`
- final machine summary：`final/summary.json`

后续只能先独立定位 EGL/renderer/runtime 的 fresh-process 非确定性；修复并重新满足 D1 后，才可人工决定是否进入 D2。不得直接恢复 N1.2 或主计划下游阶段。
