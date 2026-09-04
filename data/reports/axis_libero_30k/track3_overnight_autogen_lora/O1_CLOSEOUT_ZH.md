# O1 provenance/watchdog

O1 仅运行纯函数合同与 synthetic CPU I/O/watchdog；未导入 simulator/CUDA，未使用 GPU。结果传输为独立原子 JSON，stdout/stderr 仅诊断；超时按 TERM→KILL，重复 identity 跳过，连续两次基础设施错误触发全局停止。

状态：`track3_overnight_o1_provenance_watchdog_passed`
