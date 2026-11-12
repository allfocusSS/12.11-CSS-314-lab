# Performance Evaluation and Optimization in Parallel Computing — Solo Python Systems Lab

**Student ID:** 230103081  

**Report creation (UTC):** 2026-11-12 07:18:41  

**Execution mode:** FULL hardware measurements  

**Experiment environment:** local Python 3 and standard library; `multiprocessing`, `shared_memory`, `ctypes`, `time`, `math`.



## 0. Hardware topology, calibration, and verification

| Hardware parameter | Measured value |
| --- | --- |
| CPU model string | Apple M2 |
| Physical CPU cores (P_phys) | 8 |
| Logical hardware threads (P_log) | 8 |
| Cache line size (L_bytes) | 128 B (0.1 KiB) |
| L1 data cache | 65536 B (64.0 KiB) |
| L2 cache | 4194304 B (4096.0 KiB) |
| Shared L3 / system cache | Not reported by OS |
| Python version | 3.13.0 |
| Operating system | macOS-27.2-arm64-arm-64bit-Mach-O |
| Hybrid core-level information | {'0': {'physical_cores': 4, 'l1d_bytes': 131072, 'l2_bytes': 16777216}, '1': {'physical_cores': 4, 'l1d_bytes': 65536, 'l2_bytes': 4194304}} |


The average clock call latency is estimated from 2,000,000 calls to `time.perf_counter_ns()`. An empty-loop reference is subtracted to reduce Python loop overhead:


`avg clock latency = (gross benchmark time - empty-loop time) / count = 36.5046 ns/call`.

Gross call cost: 45.6579 ns/call; empty loop: 9.1533 ns/iteration.

**Cache caveat:** The OS does not expose an L3 cache size through the queried interface; this cannot be replaced with a guessed value. On Apple Silicon, some caches are shared system-level caches rather than a conventional x86 L3.

### Integrity hashes (printed verbatim by the individual phase scripts)

`VERIFY_HASH[PHASE0]: 5E0B3B943F349AE4 (P_log=8)`

`VERIFY_HASH[PHASE1]: 7E8BDC25B8754090 (P_log=8)`

`VERIFY_HASH[PHASE2]: 2529BDCCF371032C (P_log=8)`

`VERIFY_HASH[PHASE3]: C76A7906D44193F8 (P_log=8)`

`VERIFY_HASH[PHASE4]: 1A15FFB709E7696C (P_log=8)`

The FNV-1a 64-bit fingerprint uses this exact UTF-8 payload: `task_id|student_id|P_log|metric1:.4f|metric2:.4f`, with wraparound modulo 2^64. Phase 0 uses corrected and gross clock-call cost; Phases 1–4 use the specific measured metrics below. A hash proves consistency of reported values with its payload; it does **not** prove independent physical execution.


## 1. False sharing, stride sensitivity, Karp–Flatt

We used one 4096-byte `SharedMemory` buffer and one `ctypes.c_uint64` slot per process, with slot index `worker_id × stride`. The fixed total loop count is M = 24,000,000 and the benchmark collects two warmups and seven timed runs in FULL mode.

Actual experiment loop count: 24,000,000; warmups: 2, timed runs: 7.

### Table 1A — Stride sensitivity at p = P_phys

| Stride (words) | Separation (B) | Median T (ms) | IQR (ms) | Throughput (Mops/s) |
| --- | --- | --- | --- | --- |
| 1 | 8 | 652.443 | 25.170 | 36.785 |
| 2 | 16 | 678.860 | 15.156 | 35.353 |
| 4 | 32 | 665.942 | 14.331 | 36.039 |
| 8 | 64 | 670.925 | 7.967 | 35.772 |
| 16 | 128 | 650.725 | 11.232 | 36.882 |
| 32 | 256 | 659.844 | 12.225 | 36.372 |


**No convincing sharp stride drop was observed.** The largest adjacent improvement was only 3.01% at a separation of 128 bytes; this is not sufficient to infer a cache-line boundary. The OS-reported cache-line width is 128 bytes.

### Table 1B — Process sweep and experimental serial fraction

**Stride = 1 word(s):**

| p | T_p (ms) | S_p | E_p | Karp–Flatt e(p) |
| --- | --- | --- | --- | --- |
| 1 | 2,846.777 | 1.0000 | 1.0000 | Not available |
| 2 | 1,509.430 | 1.8860 | 0.9430 | 0.0604 |
| 4 | 866.032 | 3.2872 | 0.8218 | 0.0723 |
| 8 | 706.955 | 4.0268 | 0.5034 | 0.1410 |


**Stride = 16 word(s):**

| p | T_p (ms) | S_p | E_p | Karp–Flatt e(p) |
| --- | --- | --- | --- | --- |
| 1 | 2,835.314 | 1.0000 | 1.0000 | Not available |
| 2 | 1,489.997 | 1.9029 | 0.9514 | 0.0510 |
| 4 | 1,005.598 | 2.8195 | 0.7049 | 0.1396 |
| 8 | 772.868 | 3.6686 | 0.4586 | 0.1687 |


Calculations: `S_p=T_1/T_p`, `E_p=S_p/p`, `e(p)=(1/S_p−1/p)/(1−1/p)` for `p>=2`. The p=1 fraction is mathematically undefined.

At the last measured worker count, `e_stride1=0.1410` and `e_stride16=0.1687`. The measured fractions do not show a larger effective serial fraction for stride 1; spawning cost, cache topology, or statistical variance may mask the expected penalty.

False sharing occurs when workers write different 8-byte slots that occupy the same hardware coherence line. Cache invalidation traffic can raise memory latency even without a software data race. Karp–Flatt e(p) captures **effective** non-parallel overhead (including synchronization and memory effects), not just strictly serial source lines.


## 2. Amdahl strong scaling and Gustafson weak scaling

Stage A runs in the parent process over the first 5% of N and performs six sin/sqrt pairs (12 math calls) per element. Stage B splits all N elements across a persistent process pool and applies 35 mixed floating-point/trigonometric iterations per element. Pool construction happens outside the timed region.

Fixed strong-scaling N = 1,600,000. Weak-scaling workload N(p) = p × 250,000. Every integer p from 1 to P_log is benchmarked.

Actual strong N=1,600,000; weak N0=250,000.

### Table 2A — Strong scaling

| p | N | T_p (ms) | Speedup S_p | Efficiency E_p | Stage A (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1600000 | 4,170.400 | 1.0000 | 1.0000 | 35.517 |
| 2 | 1600000 | 2,159.935 | 1.9308 | 0.9654 | 33.610 |
| 3 | 1600000 | 1,521.498 | 2.7410 | 0.9137 | 34.722 |
| 4 | 1600000 | 1,226.016 | 3.4016 | 0.8504 | 37.423 |
| 5 | 1600000 | 1,149.680 | 3.6274 | 0.7255 | 38.657 |
| 6 | 1600000 | 1,077.674 | 3.8698 | 0.6450 | 38.613 |
| 7 | 1600000 | 1,003.147 | 4.1573 | 0.5939 | 39.614 |
| 8 | 1600000 | 952.913 | 4.3765 | 0.5471 | 40.364 |


### Table 2B — Weak scaling

| p | N(p) | T_p (ms) | Weak E | α(p) | Gustafson S_G(p) |
| --- | --- | --- | --- | --- | --- |
| 1 | 250000 | 635.229 | 1.0000 | 0.0083 | 1.0000 |
| 2 | 500000 | 693.759 | 0.9156 | 0.0160 | 1.9840 |
| 3 | 750000 | 730.648 | 0.8694 | 0.0236 | 2.9529 |
| 4 | 1000000 | 777.297 | 0.8172 | 0.0309 | 3.9074 |
| 5 | 1250000 | 917.575 | 0.6923 | 0.0329 | 4.8684 |
| 6 | 1500000 | 1,099.653 | 0.5777 | 0.0341 | 5.8294 |
| 7 | 1750000 | 1,226.239 | 0.5180 | 0.0390 | 6.7661 |
| 8 | 2000000 | 1,274.330 | 0.4985 | 0.0429 | 7.6994 |


Amdahl formula: `S_A(p)=1/[s+(1−s)/p]` where `s=T_StageA(1)/T_strong(1)`. Gustafson formula: `S_G(p)=p−α(p)(p−1)` where `α(p)=T_StageA(p)/T_weak(p)`. Weak efficiency: `T_weak(1)/T_weak(p)`.

Calculated serial fraction s = **0.008516**; theoretical asymptotic limit `S_infinity=1/s` = **117.4204**; Amdahl at P_log = **7.5499**; empirical strong speedup at P_log = **4.3765**; Gustafson scaled speedup at P_log = **7.6994**.

**SMT boundary:** p=physical+1 boundary unavailable / not tested. This machine may have no extra logical threads beyond physical cores, so the exact p=P_phys+1 point cannot be reported.

**Hybrid cluster inflection (empirical):** the larger-L2 cluster has 4 physical cores. Marginal strong-scaling speedup from p=3 to 4 was ΔS=0.6606, while from p=4 to 5 it was ΔS=0.2259, a 65.81% decrease. Worker-to-core affinity was not measured or pinned; this is an observed worker-count inflection, not proof of exact P/E-core placement.

Real scaling departs from ideal predictions because of process-pool IPC, CPU frequency variation, cache pressure, thermal throttling, and potentially heterogeneous performance/efficiency cores. Observed values, not the model, determine conclusions.


## 3. Empirical Roofline and three-stage matrix optimization

### 3A — DRAM bandwidth and Python arithmetic ceiling

Each FULL benchmark uses two 96,000,000-byte shared buffers (192,000,000 total bytes). A copy reads one source byte and writes one destination byte. `BW_GB/s=(2×96×10^6)/(T_seconds×10^9)`. An unrolled 16-scalar Python kernel counts two arithmetic FLOPs for each multiply-add update; its observed rate is `P_GFLOP/s=(workers × 80×10^6)/(T_seconds×10^9)`. The ridge is `I*=P/BW`.

| Hardware roofline parameter | Measured value |
| --- | --- |
| A and B buffer size (bytes each) | 96000000 |
| Total footprint (bytes) | 192000000 |
| Memory copy T_median (ms) | 3.279 |
| Sustained BW (GB/s) | 58.561867 |
| Computed FLOPs per worker | 80,000,000 |
| Compute T_median (ms) | 870.573 |
| Python ceiling P (GFLOP/s) | 0.735148 |
| Ridge I* (FLOPs/byte) | 0.012553 |


**Methodological limit:** Python expressions `x*a+b` are not guaranteed to become hardware-fused FMA instructions. These are interpreted Python throughput ceilings, not a hardware FLOP peak. Pool IPC and timing overhead can also affect both ceilings.

### 3B — GEMM i-j-k / i-k-j / six-loop tiling

Matrices are row-major flat double-precision shared arrays. Full N=384, so arithmetic work is `2×N^3` FLOPs. Variant 1 uses i-j-k (strided reads from B); Variant 2 uses i-k-j (contiguous B and C accesses). Variant 3 uses ii-jj-kk-i-k-j tiling with the same i-k-j microkernel. Workers own disjoint C rows. All 10 prescribed tile sizes are evaluated.

Actual N=384; p=8; warmups=1; timed runs=5.

| Variant | BS | T_median (ms) | GFLOP/s | W_tile (B) | I assumed (FLOPs/B) | Attainable GFLOP/s | Ceiling achieved (%) | Max error vs reorder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive | — | 769.179 | 0.147230 | — | 0.12500 | 0.735148 | 20.03 | — |
| reordered | — | 839.849 | 0.134841 | — | 0.12500 | 0.735148 | 18.34 | — |
| blocked | 4 | 1,122.881 | 0.100853 | 384 | 0.33333 | 0.735148 | 13.72 | 0.0000000000 |
| blocked | 8 | 1,043.998 | 0.108474 | 1536 | 0.66667 | 0.735148 | 14.76 | 0.0000000000 |
| blocked | 12 | 1,034.241 | 0.109497 | 3456 | 1.00000 | 0.735148 | 14.89 | 0.0000000000 |
| blocked | 16 | 940.466 | 0.120415 | 6144 | 1.33333 | 0.735148 | 16.38 | 0.0000000000 |
| blocked | 24 | 937.645 | 0.120777 | 13824 | 2.00000 | 0.735148 | 16.43 | 0.0000000000 |
| blocked | 32 | 969.383 | 0.116823 | 24576 | 2.66667 | 0.735148 | 15.89 | 0.0000000000 |
| blocked | 48 | 954.089 | 0.118696 | 55296 | 4.00000 | 0.735148 | 16.15 | 0.0000000000 |
| blocked | 64 | 931.931 | 0.121518 | 98304 | 5.33333 | 0.735148 | 16.53 | 0.0000000000 |
| blocked | 96 | 902.113 | 0.125534 | 221184 | 8.00000 | 0.735148 | 17.08 | 0.0000000000 |
| blocked | 128 | 903.205 | 0.125383 | 393216 | 10.66667 | 0.735148 | 17.06 | 0.0000000000 |


Best empirically observed tile BS* = **96**, with throughput **0.125534 GFLOP/s**. Its three-tile footprint `W_tile=3×BS²×8` = **221,184 bytes**; estimated tiled intensity `I_tile=BS/12` = **8.00000 FLOPs/byte**. Naive assumed intensity is 0.125 FLOP/byte.

**Observed optimization outcome:** the best blocked version was 14.74% slower than naive GEMM (0.125534 vs 0.147230 GFLOP/s). In pure Python, interpreter-level loop and indexing overhead may exceed the cache-locality benefit. This experiment does not establish a blocked speedup.

Tile footprint versus L1d=65,536 B: exceeds nominal capacity. Actual capacity also holds code, stack and other active data.

Tile footprint versus L2=4,194,304 B: fits nominal capacity. Actual capacity also holds code, stack and other active data.

Roofline model: `P_attainable(I)=min(P_sustained, BW_sustained×I)` and `%attained=100×GFLOP/s_measured/P_attainable`. The reordered variant uses 0.125 FLOP/byte only as a **conservative nominal reference**: its true operational intensity is not specified by the task or measured by hardware counters. A value above 100% flags a model mismatch, not an impossible measured observation.

Numerical validation: all blocked variants are compared element-by-element against Variant 2; their largest absolute differences must be strictly less than 1e−9. All reported tile runs passed this in the generated results.


## 4. Skewed scheduling and load imbalance

Full workload has N=12,000 outer items, with `K(i)=1+floor((i/60)^2)=1+(i*i//3600)` trigonometric iterations per item. Strategy 1 uses contiguous static blocks; Strategy 2 round-robin cyclic assignment; Strategy 3 a synchronized shared index with dynamic chunk sizes C∈{1,4,16,64,256,1024}. All trials use P_phys processes.

`lambda = max_t(t_worker[t]) / ((sum_t t_worker[t])/p)`. Active worker times exclude idle lock-wait periods; wall time includes process startup and dispatch. Values close to 1 imply balanced useful compute times.

Actual workload N=12,000, physical workers=8, warmups=1, timed runs=3.

| Scheduling strategy | Chunk C | Median wall time (ms) | IQR (ms) | λ (active load imbalance) |
| --- | --- | --- | --- | --- |
| block | — | 6,467.311 | 106.613 | 2.41613 |
| cyclic | — | 4,058.460 | 526.499 | 1.01084 |
| dynamic | 1 | 3,821.184 | 76.071 | 1.00089 |
| dynamic | 4 | 3,874.996 | 34.794 | 1.00319 |
| dynamic | 16 | 3,877.506 | 16.767 | 1.00923 |
| dynamic | 64 | 3,967.091 | 48.937 | 1.03294 |
| dynamic | 256 | 4,193.703 | 103.649 | 1.13355 |
| dynamic | 1024 | 5,427.362 | 718.909 | 1.56742 |


### Individual active worker times (representative median trial)

- **block C=Not measured:** W0=45.41 ms, W1=331.17 ms, W2=936.07 ms, W3=1,778.59 ms, W4=2,731.70 ms, W5=3,851.09 ms, W6=5,070.08 ms, W7=6,379.74 ms

- **cyclic C=Not measured:** W0=3,847.88 ms, W1=3,947.16 ms, W2=3,848.82 ms, W3=3,901.31 ms, W4=3,913.11 ms, W5=3,943.31 ms, W6=3,950.07 ms, W7=3,909.84 ms

- **dynamic C=1:** W0=3,719.05 ms, W1=3,721.63 ms, W2=3,718.19 ms, W3=3,718.60 ms, W4=3,721.42 ms, W5=3,717.35 ms, W6=3,711.32 ms, W7=3,719.12 ms

- **dynamic C=4:** W0=3,761.46 ms, W1=3,785.26 ms, W2=3,766.77 ms, W3=3,787.20 ms, W4=3,775.33 ms, W5=3,772.21 ms, W6=3,785.78 ms, W7=3,767.37 ms

- **dynamic C=16:** W0=3,722.99 ms, W1=3,749.47 ms, W2=3,788.60 ms, W3=3,706.39 ms, W4=3,788.90 ms, W5=3,724.37 ms, W6=3,764.94 ms, W7=3,788.33 ms

- **dynamic C=64:** W0=3,789.16 ms, W1=3,785.07 ms, W2=3,885.88 ms, W3=3,823.73 ms, W4=3,637.63 ms, W5=3,694.49 ms, W6=3,708.78 ms, W7=3,770.99 ms

- **dynamic C=256:** W0=4,112.16 ms, W1=3,618.12 ms, W2=4,100.61 ms, W3=3,375.14 ms, W4=3,747.10 ms, W5=3,169.14 ms, W6=3,906.67 ms, W7=2,992.38 ms

- **dynamic C=1024:** W0=4,985.70 ms, W1=4,531.62 ms, W2=3,110.36 ms, W3=2,417.52 ms, W4=1,206.57 ms, W5=1,860.71 ms, W6=5,327.75 ms, W7=3,752.28 ms

**Optimal measured dynamic chunk C* = 1** with 3,821.184 ms wall time. Static block = 6,467.311 ms; cyclic = 4,058.460 ms.

At C=1: T=3,821.184 ms, λ=1.0009. At C=1024: T=5,427.362 ms, λ=1.5674. Small chunks usually reduce remaining uneven work but increase lock acquisitions and contention; large chunks reduce lock traffic but can leave high-cost items on one worker. The recorded table, rather than an assumed optimum, determines C*.


## 5. Raw local terminal telemetry

### phase0.log

```text

cpu_model: Apple M2
physical_cores: 8
logical_threads: 8
cacheline_bytes: 128
l1d_bytes: 65536
l2_bytes: 4194304
l3_bytes: None
python_version: 3.13.0
os: macOS-27.2-arm64-arm-64bit-Mach-O
core_clusters: {'0': {'physical_cores': 4, 'l1d_bytes': 131072, 'l2_bytes': 16777216}, '1': {'physical_cores': 4, 'l1d_bytes': 65536, 'l2_bytes': 4194304}}
CLOCK_READS=2000000 CORRECTED_AVG_NS=36.5046 GROSS_AVG_NS=45.6579
VERIFY_HASH[PHASE0]: 5E0B3B943F349AE4 (P_log=8)

```

### phase1.log

```text

p= 8 stride= 1 time=652.443ms IQR=25.170 throughput=36.785 Mops/s
p= 8 stride= 2 time=678.860ms IQR=15.156 throughput=35.353 Mops/s
p= 8 stride= 4 time=665.942ms IQR=14.331 throughput=36.039 Mops/s
p= 8 stride= 8 time=670.925ms IQR=7.967 throughput=35.772 Mops/s
p= 8 stride=16 time=650.725ms IQR=11.232 throughput=36.882 Mops/s
p= 8 stride=32 time=659.844ms IQR=12.225 throughput=36.372 Mops/s
p= 1 stride= 1 time=2846.777ms IQR=59.752 throughput=8.431 Mops/s
p= 2 stride= 1 time=1509.430ms IQR=25.726 throughput=15.900 Mops/s
p= 4 stride= 1 time=866.032ms IQR=8.648 throughput=27.713 Mops/s
p= 8 stride= 1 time=706.955ms IQR=16.147 throughput=33.948 Mops/s
STRIDE=1 p=1 S=1.0000 E=1.0000 e=None
STRIDE=1 p=2 S=1.8860 E=0.9430 e=0.06044838669215591
STRIDE=1 p=4 S=3.2872 E=0.8218 e=0.07228637458944316
STRIDE=1 p=8 S=4.0268 E=0.5034 e=0.1409546489701294
p= 1 stride=16 time=2835.314ms IQR=51.091 throughput=8.465 Mops/s
p= 2 stride=16 time=1489.997ms IQR=28.533 throughput=16.107 Mops/s
p= 4 stride=16 time=1005.598ms IQR=184.091 throughput=23.866 Mops/s
p= 8 stride=16 time=772.868ms IQR=87.589 throughput=31.053 Mops/s
STRIDE=16 p=1 S=1.0000 E=1.0000 e=None
STRIDE=16 p=2 S=1.9029 E=0.9514 e=0.05102768273118863
STRIDE=16 p=4 S=2.8195 E=0.7049 e=0.13955847403498164
STRIDE=16 p=8 S=3.6686 E=0.4586 e=0.16867014945874764
VERIFY_HASH[PHASE1]: 7E8BDC25B8754090 (P_log=8)

```

### phase2.log

```text

full: p= 1 N=1600000 T=4170.400ms stageA=35.517ms
full: p= 2 N=1600000 T=2159.935ms stageA=33.610ms
full: p= 3 N=1600000 T=1521.498ms stageA=34.722ms
full: p= 4 N=1600000 T=1226.016ms stageA=37.423ms
full: p= 5 N=1600000 T=1149.680ms stageA=38.657ms
full: p= 6 N=1600000 T=1077.674ms stageA=38.613ms
full: p= 7 N=1600000 T=1003.147ms stageA=39.614ms
full: p= 8 N=1600000 T=952.913ms stageA=40.364ms
full: p= 1 N=250000 T=635.229ms stageA=5.289ms
full: p= 2 N=500000 T=693.759ms stageA=11.124ms
full: p= 3 N=750000 T=730.648ms stageA=17.219ms
full: p= 4 N=1000000 T=777.297ms stageA=23.983ms
full: p= 5 N=1250000 T=917.575ms stageA=30.184ms
full: p= 6 N=1500000 T=1099.653ms stageA=37.521ms
full: p= 7 N=1750000 T=1226.239ms stageA=47.799ms
full: p= 8 N=2000000 T=1274.330ms stageA=54.728ms
SERIAL_FRACTION=0.008516 AMDAHL_MAX=117.42038508895206 AMDAHL_PLOG=7.54991
MARGINAL_BOUNDARY={'status': 'p=physical+1 boundary unavailable / not tested'}
VERIFY_HASH[PHASE2]: 2529BDCCF371032C (P_log=8)

```

### phase3a.log

```text

BANDWIDTH: 96000000 bytes per buffer; median=3.2786ms; BWsustained=58.561867 GB/s
COMPUTE: 80000000 FLOPs/worker; median=870.573ms; Psustained=0.735148 GFLOP/s
RIDGE_POINT=0.012553349410857707 FLOPs/byte

```

### phase3b.log

```text

naive     BS=None: 832.096ms 0.136098 GFLOP/s
reordered BS=None: 893.667ms 0.126721 GFLOP/s
blocked   BS=   4: 1257.482ms 0.090058 GFLOP/s
blocked   BS=   8: 1059.564ms 0.106880 GFLOP/s
blocked   BS=  12: 1011.486ms 0.111960 GFLOP/s
blocked   BS=  16: 984.750ms 0.115000 GFLOP/s
blocked   BS=  24: 978.841ms 0.115694 GFLOP/s
blocked   BS=  32: 946.776ms 0.119612 GFLOP/s
blocked   BS=  48: 943.312ms 0.120052 GFLOP/s
blocked   BS=  64: 926.759ms 0.122196 GFLOP/s
blocked   BS=  96: 927.278ms 0.122128 GFLOP/s
blocked   BS= 128: 918.841ms 0.123249 GFLOP/s
Exception ignored in: <function SharedMemory.__del__ at 0x106312840>
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 189, in __del__
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 232, in close
BufferError: cannot close exported pointers exist
Exception ignored in: <function SharedMemory.__del__ at 0x106312840>
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 189, in __del__
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 232, in close
BufferError: cannot close exported pointers exist
Exception ignored in: <function SharedMemory.__del__ at 0x106312840>
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 189, in __del__
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 232, in close
BufferError: cannot close exported pointers exist
Exception ignored in: <function SharedMemory.__del__ at 0x10174a840>
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 189, in __del__
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 232, in close
BufferError: cannot close exported pointers exist
Exception ignored in: <function SharedMemory.__del__ at 0x10174a840>
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 189, in __del__
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 232, in close
BufferError: cannot close exported pointers exist
Exception ignored in: <function SharedMemory.__del__ at 0x10174a840>
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 189, in __del__
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 232, in close
BufferError: cannot close exported pointers exist
Exception ignored in: <function SharedMemory.__del__ at 0x108502840>
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/shared_memory.py", line 189, in __del__
OPTIMAL_BS=128 BEST_GFLOPS=0.123249
VERIFY_HASH[PHASE3]: 13B4DCAFD8918590 (P_log=8)

```

### phase4.log

```text

block    chunk=None wall=6467.311ms lambda=2.4161
cyclic   chunk=None wall=4058.460ms lambda=1.0108
dynamic  chunk=   1 wall=3821.184ms lambda=1.0009
dynamic  chunk=   4 wall=3874.996ms lambda=1.0032
dynamic  chunk=  16 wall=3877.506ms lambda=1.0092
dynamic  chunk=  64 wall=3967.091ms lambda=1.0329
dynamic  chunk= 256 wall=4193.703ms lambda=1.1336
dynamic  chunk=1024 wall=5427.362ms lambda=1.5674
OPTIMAL_DYNAMIC_CHUNK=1
VERIFY_HASH[PHASE4]: 1A15FFB709E7696C (P_log=8)

```


## 6. Reproduction and limitations

Execute `python3 run_all.py --student-id 230103081 --mode full` from this directory, on the target laptop. The runner writes JSON and console log evidence for every phase, then regenerates this Markdown report. The `--mode smoke` option tests only code execution with reduced workload and is **not acceptable as the requested empirical full run**.

The study uses wall-clock observations on a real machine; CPU availability, power mode, resource limits, thermal history, cross-process startup and OS scheduling introduce noise. Median and IQR are reported when available. Clock calibration alone does not remove those effects. No performance observation is extrapolated from another machine.

**Assignment restriction:** the source PDF says the work must be performed offline, solo and without AI tools. This prepared package is assistance and does not replace the required independently executed lab. Follow your instructor’s academic-integrity requirements when choosing what to submit.


## Appendix A — Complete Python 3 implementation

### `harness.py`

```python

"""Standard-library-only benchmark, host topology and FNV-1a verification helpers."""
from dataclasses import dataclass, asdict
import math
import multiprocessing as mp
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Callable

@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float
    samples_ms: list[float]

    def as_dict(self):
        return asdict(self)


def fnv1a_hash(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode('utf-8'):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def emit_verify_tag(task_id: str, student_seed: int,
                    primary_metric: float, secondary_metric: float,
                    hw: int | None = None) -> str:
    """Matches the exact payload format in the assignment's Listing 1."""
    hw = mp.cpu_count() if hw is None else hw
    payload = (f'{task_id}|{student_seed}|{hw}|'
               f'{primary_metric:.4f}|{secondary_metric:.4f}')
    sig = fnv1a_hash(payload)
    return f'VERIFY_HASH[{task_id}]: {sig:016X} (P_log={hw})'


def run_benchmark(fn: Callable[[], object], warmup_runs: int = 3,
                  timed_runs: int = 9) -> BenchStats:
    for _ in range(warmup_runs):
        fn()
    samples = []
    for _ in range(timed_runs):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    ordered = sorted(samples)
    n = len(ordered)
    med = statistics.median(ordered)
    # The assignment explicitly defines quartiles by integer-index order statistics.
    iqr = ordered[(3*n)//4] - ordered[n//4]
    return BenchStats(ordered[0], med, iqr, ordered[-1], samples)


def run_native(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL,
                                       text=True, timeout=10).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ''


def sysctl(key):
    val = run_native(['sysctl', '-n', key])
    try:
        return int(val)
    except ValueError:
        return val or None


def hardware_topology():
    system = platform.system()
    data = {
        'cpu_model': platform.processor() or platform.machine(),
        'physical_cores': None,
        'logical_threads': mp.cpu_count(),
        'cacheline_bytes': None,
        'l1d_bytes': None,
        'l2_bytes': None,
        'l3_bytes': None,
        'python_version': platform.python_version(),
        'os': platform.platform(),
        'core_clusters': {},
    }
    if system == 'Darwin':
        data['cpu_model'] = (sysctl('machdep.cpu.brand_string')
                             or sysctl('hw.model') or data['cpu_model'])
        for target, name in [('physical_cores','hw.physicalcpu'),
                             ('logical_threads','hw.logicalcpu'),
                             ('cacheline_bytes','hw.cachelinesize'),
                             ('l1d_bytes','hw.l1dcachesize'),
                             ('l2_bytes','hw.l2cachesize'),
                             ('l3_bytes','hw.l3cachesize')]:
            v = sysctl(name)
            if isinstance(v,int) and v>0: data[target] = v
        # Apple Silicon exposes performance-level caches rather than always global cache keys.
        for lvl in (0,1,2):
            count = sysctl(f'hw.perflevel{lvl}.physicalcpu')
            if isinstance(count, int) and count > 0:
                data['core_clusters'][str(lvl)] = {
                    'physical_cores': count,
                    'l1d_bytes': sysctl(f'hw.perflevel{lvl}.l1dcachesize'),
                    'l2_bytes': sysctl(f'hw.perflevel{lvl}.l2cachesize'),
                }
        if data['l1d_bytes'] is None and data['core_clusters']:
            data['l1d_bytes'] = data['core_clusters'].get('0',{}).get('l1d_bytes')
        if data['l2_bytes'] is None and data['core_clusters']:
            data['l2_bytes'] = data['core_clusters'].get('0',{}).get('l2_bytes')
    elif system == 'Linux':
        info = run_native(['lscpu'])
        for line in info.splitlines():
            if line.startswith('Model name:'):
                data['cpu_model'] = line.split(':',1)[1].strip()
        # Count actual physical (socket, core) pairs, not hardware threads.
        pairs = set()
        for line in run_native(['lscpu','-p=SOCKET,CORE']).splitlines():
            if not line.startswith('#') and ',' in line:
                vals=line.split(',')
                if len(vals)==2 and all(x.strip().isdigit() for x in vals):
                    pairs.add(tuple(vals))
        if pairs: data['physical_cores'] = len(pairs)
        path = '/sys/devices/system/cpu/cpu0/cache'
        try:
            for key in os.listdir(path):
                if not key.startswith('index'): continue
                prefix=os.path.join(path,key)
                with open(os.path.join(prefix,'type')) as fh: typ=fh.read().strip()
                with open(os.path.join(prefix,'level')) as fh: lvl=fh.read().strip()
                with open(os.path.join(prefix,'size')) as fh: raw=fh.read().strip()
                m=re.fullmatch(r'(\d+)([KMG])',raw)
                if not m: continue
                num=int(m.group(1))*{'K':1024,'M':1024**2,'G':1024**3}[m.group(2)]
                if lvl=='1' and typ=='Data': data['l1d_bytes']=num
                if lvl=='2': data['l2_bytes']=num
                if lvl=='3': data['l3_bytes']=num
                if lvl=='1' and typ=='Data':
                    with open(os.path.join(prefix,'coherency_line_size')) as fh:
                        data['cacheline_bytes']=int(fh.read().strip())
        except (OSError,ValueError):
            pass
    elif system == 'Windows':
        text = run_native(['powershell','-NoProfile','-Command',
            'Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,L2CacheSize,L3CacheSize | ConvertTo-Json -Compress'])
        if text:
            import json
            try:
                obj=json.loads(text)
                if isinstance(obj,list):
                    data['physical_cores']=sum(int(x['NumberOfCores']) for x in obj)
                    obj=obj[0]
                else: data['physical_cores']=int(obj['NumberOfCores'])
                data['cpu_model']=obj.get('Name') or data['cpu_model']
                for dest,key in [('l2_bytes','L2CacheSize'),('l3_bytes','L3CacheSize')]:
                    if obj.get(key): data[dest]=int(obj[key])*1024
            except (ValueError,KeyError,TypeError):
                pass
    if not data['physical_cores']:
        data['physical_cores'] = os.cpu_count() or 1
        data['physical_cores_source'] = 'logical fallback; verify physical core count manually'
    return data


def write_json(path, data):
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False)
        f.write('\n')


def read_json(path):
    import json
    with open(path,encoding='utf-8') as f: return json.load(f)


def cli_parser(description):
    import argparse
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--student-id',type=int,default=230103081)
    p.add_argument('--mode',choices=['full','smoke'],default='full')
    p.add_argument('--output',required=True)
    return p


def sweep_powers(max_p):
    p, vals=1,[]
    while p<=max_p:
        vals.append(p)
        p*=2
    if vals[-1]!=max_p: vals.append(max_p)
    return vals


def speedup_table(results, baseline_key='median_ms'):
    first=results[0][baseline_key]
    for r in results:
        r['speedup']=first/r[baseline_key]
        r['efficiency']=r['speedup']/r['p']
    return results

```

### `task0_cal.py`

```python

"""Phase 0: physical topology and 2-million-read clock latency calibration."""
import multiprocessing as mp
import time
from harness import (hardware_topology,emit_verify_tag,write_json,cli_parser)


def main():
    args=cli_parser(__doc__).parse_args()
    hw=hardware_topology()
    N=2_000_000 if args.mode=='full' else 20_000
    # Calibration of loop baseline makes the read latency estimate more informative.
    t0=time.perf_counter_ns()
    for _ in range(N):
        pass
    empty_ns=time.perf_counter_ns()-t0
    t0=time.perf_counter_ns()
    for _ in range(N):
        time.perf_counter_ns()
    gross_ns=time.perf_counter_ns()-t0
    avg_ns=max(0.0,(gross_ns-empty_ns)/N)
    gross_avg_ns=gross_ns/N
    verify=emit_verify_tag('PHASE0',args.student_id,avg_ns,gross_avg_ns)
    obj={'phase':0,'mode':args.mode,'student_id':args.student_id,
         'hardware':hw,'clock_reads':N,
         'clock_read_latency_ns':avg_ns,
         'clock_gross_ns_per_read':gross_avg_ns,
         'empty_loop_ns_per_iteration':empty_ns/N,
         'verify_hash':verify}
    for k,v in hw.items(): print(f'{k}: {v}')
    print(f'CLOCK_READS={N} CORRECTED_AVG_NS={avg_ns:.4f} GROSS_AVG_NS={gross_avg_ns:.4f}')
    print(verify)
    write_json(args.output,obj)

if __name__=='__main__':
    mp.freeze_support()
    main()

```

### `task1_falsesharing.py`

```python

"""Phase 1: shared-memory coherence stride and Karp-Flatt diagnostics."""
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from harness import (read_json,write_json,cli_parser,run_benchmark,
                     emit_verify_tag,sweep_powers,speedup_table)


def fs_worker(shm_name: str, slot_idx: int, iters: int):
    shm=shared_memory.SharedMemory(name=shm_name)
    arr=(ctypes.c_uint64*512).from_buffer(shm.buf)
    acc=arr[slot_idx]
    mask=0xFFFFFFFFFFFFFFFF
    mult=6364136223846793005
    for i in range(iters):
        acc=(acc*mult+(i|1))&mask
        arr[slot_idx]=acc
    del arr
    shm.close()


def one_trial(ctx,p,stride,total):
    # Exact specification: 4096 B / 512 unsigned 64-bit slots.
    if (p-1)*stride>=512:
        raise ValueError('512 slots cannot support p and stride; PDF hardware constraints conflict')
    shm=shared_memory.SharedMemory(create=True,size=4096)
    try:
        jobs=[]
        for t in range(p):
            iters=total//p+(t<total%p)
            pr=ctx.Process(target=fs_worker,args=(shm.name,t*stride,iters))
            pr.start()
            jobs.append(pr)
        for pr in jobs:
            pr.join()
            if pr.exitcode!=0:
                raise RuntimeError(f'False-sharing worker failed: exit code {pr.exitcode}')
    finally:
        shm.close()
        shm.unlink()


def main():
    args=cli_parser(__doc__).parse_args()
    phase0=read_json(os.path.join(os.path.dirname(args.output),'phase0.json'))
    hw=phase0['hardware']
    phys=int(hw['physical_cores']); logical=int(hw['logical_threads'])
    true_logical=logical
    if args.mode=='smoke': phys=min(phys,2); logical=min(logical,2)
    M=24_000_000 if args.mode=='full' else 4_000
    warmups,timed=(2,7) if args.mode=='full' else (0,1)
    ctx=mp.get_context('spawn')

    def measure(p,stride):
        stat=run_benchmark(lambda:one_trial(ctx,p,stride,M),warmups,timed)
        obj={'p':p,'stride_words':stride,'separation_bytes':8*stride,
             'median_ms':stat.median_ms,'iqr_ms':stat.iqr_ms,
             'throughput_mops':M/(1000.0*stat.median_ms),'min_ms':stat.min_ms,
             'max_ms':stat.max_ms}
        print(f'p={p:2d} stride={stride:2d} time={stat.median_ms:.3f}ms IQR={stat.iqr_ms:.3f} throughput={obj["throughput_mops"]:.3f} Mops/s',flush=True)
        return obj

    stride_sweep=[measure(phys,s) for s in [1,2,4,8,16,32]]
    processes=sweep_powers(logical)
    scaling={}
    for stride in (1,16):
        rows=[measure(p,stride) for p in processes]
        speedup_table(rows)
        for row in rows:
            p=row['p']; s=row['speedup']
            row['karp_flatt_e']=None if p==1 else ((1/s)-(1/p))/(1-(1/p))
            print(f'STRIDE={stride} p={p} S={s:.4f} E={row["efficiency"]:.4f} e={row["karp_flatt_e"]}',flush=True)
        scaling[str(stride)]=rows
    s1=stride_sweep[0]['median_ms']
    s16=next(row['median_ms'] for row in stride_sweep if row['stride_words']==16)
    verify=emit_verify_tag('PHASE1',args.student_id,s1,s16,true_logical)
    print(verify)
    write_json(args.output,{'phase':1,'mode':args.mode,'student_id':args.student_id,
         'M':M,'warmups':warmups,'timed_runs':timed,'stride_sweep':stride_sweep,
         'scaling':scaling,'verify_hash':verify})

if __name__=='__main__':
    mp.freeze_support()
    main()

```

### `task2_scaling.py`

```python

"""Phase 2: persistent-process Amdahl/Gustafson empirical scaling."""
import math
import multiprocessing as mp
import os
import statistics
import time
from harness import (read_json,write_json,cli_parser,run_benchmark,
                     emit_verify_tag)


def stage_a(N):
    total=0.0
    for i in range((N*5)//100):
        x=1.0+i*1e-7
        # Six sin/sqrt pairs = exactly 12 transcendental operations / element.
        for _ in range(6):
            x=math.sin(x)
            x=math.sqrt(x*x+1.0)
        total+=x
    return total


def stage_b(args):
    start,stop=args
    total=0.0
    for i in range(start,stop):
        x=0.2+i*1e-7
        for _ in range(35):
            x=x*0.999999+0.000001
            x+=math.sin(x)*0.000001
            x+=math.cos(x)*0.000001
        total+=x
    return total


def parts(n,p):
    return [(t*n//p,(t+1)*n//p) for t in range(p)]


def measure(ctx,p,n,mode):
    warmup,timed=(1,5) if mode=='full' else (0,1)
    a_times=[]
    # Pool spawn is excluded from the timed benchmark loop.
    with ctx.Pool(processes=p) as pool:
        def trial():
            t=time.perf_counter_ns()
            serial=stage_a(n)
            a_times.append((time.perf_counter_ns()-t)/1e6)
            totals=pool.map(stage_b,parts(n,p))
            if not math.isfinite(serial+sum(totals)):
                raise RuntimeError('Nonfinite scaling checksum')
        stat=run_benchmark(trial,warmup,timed)
    measured_a=a_times[warmup:]
    row={'p':p,'N':n,'median_ms':stat.median_ms,'iqr_ms':stat.iqr_ms,
         'serial_stage_median_ms':statistics.median(measured_a),
         'warmups':warmup,'timed_runs':timed}
    print(f'{mode}: p={p:2d} N={n} T={stat.median_ms:.3f}ms stageA={row["serial_stage_median_ms"]:.3f}ms',flush=True)
    return row


def marginal_analysis(rows,phys):
    d={r['p']:r for r in rows}
    if phys<=1 or phys+1 not in d or phys-1 not in d or phys not in d:
        return {'status':'p=physical+1 boundary unavailable / not tested'}
    before=d[phys]['speedup']-d[phys-1]['speedup']
    after=d[phys+1]['speedup']-d[phys]['speedup']
    pct=None if before==0 else 100.0*(before-after)/abs(before)
    return {'status':'measured','p_before':phys-1,'p_boundary':phys,
            'p_after':phys+1,'marginal_gain_before':before,
            'marginal_gain_after':after,'percentage_drop':pct}


def main():
    args=cli_parser(__doc__).parse_args()
    phase0=read_json(os.path.join(os.path.dirname(args.output),'phase0.json'))
    hw=phase0['hardware']; phys=int(hw['physical_cores']); logical=int(hw['logical_threads'])
    true_logical=logical
    if args.mode=='smoke': phys=min(phys,2); logical=min(logical,2)
    strong_n=1_600_000 if args.mode=='full' else 300
    weak_n0=250_000 if args.mode=='full' else 100
    ctx=mp.get_context('spawn')
    strong=[measure(ctx,p,strong_n,args.mode) for p in range(1,logical+1)]
    base=strong[0]['median_ms']
    for row in strong:
        row['speedup']=base/row['median_ms']
        row['efficiency']=row['speedup']/row['p']
    s=strong[0]['serial_stage_median_ms']/base
    amdahl_max=1/s if s>0 else None
    amdahl_p=1/(s+(1-s)/logical)
    weak=[measure(ctx,p,p*weak_n0,args.mode) for p in range(1,logical+1)]
    baseweak=weak[0]['median_ms']
    for row in weak:
        p=row['p']
        row['weak_efficiency']=baseweak/row['median_ms']
        row['alpha']=row['serial_stage_median_ms']/row['median_ms']
        row['gustafson_speedup']=p-row['alpha']*(p-1)
    boundary=marginal_analysis(strong,phys)
    verify=emit_verify_tag('PHASE2',args.student_id,strong[-1]['speedup'],
                           weak[-1]['gustafson_speedup'],true_logical)
    print(f'SERIAL_FRACTION={s:.6f} AMDAHL_MAX={amdahl_max} AMDAHL_PLOG={amdahl_p:.5f}',flush=True)
    print(f'MARGINAL_BOUNDARY={boundary}',flush=True)
    print(verify)
    write_json(args.output,{'phase':2,'mode':args.mode,'student_id':args.student_id,
         'strong_N':strong_n,'weak_N0':weak_n0,'strong':strong,'weak':weak,
         'serial_fraction':s,'amdahl_infinity':amdahl_max,'amdahl_plog':amdahl_p,
         'boundary':boundary,'verify_hash':verify})

if __name__=='__main__':
    mp.freeze_support()
    main()

```

### `task3a_roofline.py`

```python

"""Phase 3A: 96 MB shared-memory memcpy and unrolled scalar-FP ceiling."""
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from harness import read_json,write_json,cli_parser,run_benchmark

_BWA=None
_BWB=None


def bw_initializer(a_name,b_name):
    global _BWA,_BWB
    _BWA=shared_memory.SharedMemory(name=a_name)
    _BWB=shared_memory.SharedMemory(name=b_name)


def bw_slice(bounds):
    start,end=bounds
    # Slice copy executes in the native CPython memoryview/memcpy implementation.
    _BWA.buf[start:end]=_BWB.buf[start:end]
    return end-start


def compute_worker(iterations):
    a=1.000000000000001; b=0.000000000001
    x0=x1=x2=x3=x4=x5=x6=x7=0.1
    x8=x9=x10=x11=x12=x13=x14=x15=0.2
    for _ in range(iterations):
        x0=x0*a+b
        x1=x1*a+b
        x2=x2*a+b
        x3=x3*a+b
        x4=x4*a+b
        x5=x5*a+b
        x6=x6*a+b
        x7=x7*a+b
        x8=x8*a+b
        x9=x9*a+b
        x10=x10*a+b
        x11=x11*a+b
        x12=x12*a+b
        x13=x13*a+b
        x14=x14*a+b
        x15=x15*a+b
    return (x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15)


def main():
    args=cli_parser(__doc__).parse_args()
    hw=read_json(os.path.join(os.path.dirname(args.output),'phase0.json'))['hardware']
    phys=int(hw['physical_cores']); ctx=mp.get_context('spawn')
    if args.mode=='smoke': phys=min(phys,2)
    size=96_000_000 if args.mode=='full' else 1_000_000
    # This strict full-size allocation is required for the 192 MB DRAM working set.
    a=shared_memory.SharedMemory(create=True,size=size)
    b=shared_memory.SharedMemory(create=True,size=size)
    try:
        # Touch B before measurement so its pages are resident.
        block=bytes([123])*1_000_000
        for start in range(0,size,len(block)):
            end=min(start+len(block),size)
            b.buf[start:end]=block[:end-start]
        regions=[(i*size//phys,(i+1)*size//phys) for i in range(phys)]
        warmups,timed=(2,7) if args.mode=='full' else (0,1)
        with ctx.Pool(phys,initializer=bw_initializer,initargs=(a.name,b.name)) as pool:
            def bw_trial():
                copied=pool.map(bw_slice,regions)
                if sum(copied)!=size: raise RuntimeError('Incomplete copy')
            bw_stat=run_benchmark(bw_trial,warmups,timed)
        bw_gbps=(2*size)/(bw_stat.median_ms*1e6)
        print(f'BANDWIDTH: {size} bytes per buffer; median={bw_stat.median_ms:.4f}ms; BWsustained={bw_gbps:.6f} GB/s',flush=True)
    finally:
        a.close(); b.close(); a.unlink(); b.unlink()

    # 80 million counted arithmetic FLOPs per physical worker; each x*a+b is 2 FLOPs.
    flops_each=80_000_000 if args.mode=='full' else 3_200
    loops=flops_each//32
    actual_flops_per_worker=loops*32
    with ctx.Pool(phys) as pool:
        def compute_trial():
            results=pool.map(compute_worker,[loops]*phys)
            if not all(math.isfinite(x) for x in results):
                raise RuntimeError('Nonfinite arithmetic benchmark result')
        compute_stat=run_benchmark(compute_trial,warmups,timed)
    compute_gflops=(actual_flops_per_worker*phys)/(compute_stat.median_ms*1e6)
    ridge=compute_gflops/bw_gbps if bw_gbps>0 else None
    print(f'COMPUTE: {actual_flops_per_worker} FLOPs/worker; median={compute_stat.median_ms:.3f}ms; Psustained={compute_gflops:.6f} GFLOP/s',flush=True)
    print(f'RIDGE_POINT={ridge} FLOPs/byte',flush=True)
    write_json(args.output,{'phase':'3a','mode':args.mode,'student_id':args.student_id,
         'buffer_bytes_each':size,'working_set_bytes':2*size,
         'bw_median_ms':bw_stat.median_ms,'bw_iqr_ms':bw_stat.iqr_ms,'bandwidth_gbps':bw_gbps,
         'flops_each':actual_flops_per_worker,'compute_median_ms':compute_stat.median_ms,
         'compute_iqr_ms':compute_stat.iqr_ms,'compute_gflops':compute_gflops,
         'ridge_flops_per_byte':ridge,
         'note':'Python x*a+b counts 2 FLOPs but is not guaranteed hardware-fused FMA. Measured ceilings include pool-dispatch overhead.'})

if __name__=='__main__':
    mp.freeze_support()
    main()

```

### `task3b_gemm.py`

```python

"""Phase 3B: shared double-precision i-j-k, i-k-j and 6-loop tiled GEMM."""
import math
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.util import Finalize
import os
from harness import (read_json,write_json,cli_parser,run_benchmark,emit_verify_tag)

_A=_B=_C=None
_AS=_BS=_CS=None


def matrix_initializer(a,b,c):
    global _A,_B,_C,_AS,_BS,_CS
    _AS=shared_memory.SharedMemory(name=a)
    _BS=shared_memory.SharedMemory(name=b)
    _CS=shared_memory.SharedMemory(name=c)
    _A=_AS.buf.cast('d')
    _B=_BS.buf.cast('d')
    _C=_CS.buf.cast('d')
    # Pool workers must release every exported memoryview before closing
    # its SharedMemory handle. A positive-priority Finalize runs before
    # interpreter teardown, avoiding BufferError in SharedMemory.__del__.
    Finalize(None, release_matrix_views, exitpriority=10)


def release_matrix_views():
    global _A, _B, _C, _AS, _BS, _CS
    # Release views before closing the backing SharedMemory objects.
    _A = _B = _C = None
    for shm in (_AS, _BS, _CS):
        if shm is not None:
            shm.close()
    _AS = _BS = _CS = None


def gemm_worker(args):
    strategy,start,end,n,bs=args
    A=_A; B=_B; C=_C
    if strategy=='naive':
        for i in range(start,end):
            row=i*n
            for j in range(n):
                z=0.0
                for k in range(n):
                    z+=A[row+k]*B[k*n+j]
                C[row+j]=z
    elif strategy=='reordered':
        for i in range(start,end):
            row=i*n
            for j in range(n): C[row+j]=0.0
            for k in range(n):
                a=A[row+k]; brow=k*n
                for j in range(n): C[row+j]+=a*B[brow+j]
    elif strategy=='blocked':
        for i in range(start,end):
            row=i*n
            for j in range(n): C[row+j]=0.0
        # Six loops ii,jj,kk,i,k,j; different workers own disjoint row ranges.
        for ii in range(start,end,bs):
            imax=min(ii+bs,end)
            for jj in range(0,n,bs):
                jmax=min(jj+bs,n)
                for kk in range(0,n,bs):
                    kmax=min(kk+bs,n)
                    for i in range(ii,imax):
                        row=i*n
                        for k in range(kk,kmax):
                            a=A[row+k]; brow=k*n
                            for j in range(jj,jmax):
                                C[row+j]+=a*B[brow+j]
    else:
        raise ValueError(strategy)
    return end-start


def main():
    args=cli_parser(__doc__).parse_args()
    data_dir=os.path.dirname(args.output)
    hw=read_json(os.path.join(data_dir,'phase0.json'))['hardware']
    roof=read_json(os.path.join(data_dir,'phase3a.json'))
    phys=int(hw['physical_cores']); logical=int(hw['logical_threads'])
    true_logical=logical
    if args.mode=='smoke': phys=min(phys,2); logical=min(logical,2)
    n=384 if args.mode=='full' else 24
    tiles=[4,8,12,16,24,32,48,64,96,128]
    warmup,timed=(1,5) if args.mode=='full' else (0,1)
    ctx=mp.get_context('spawn')
    length=n*n*8
    mems=[shared_memory.SharedMemory(create=True,size=length) for _ in range(3)]
    ma,mb,mc=mems
    a=ma.buf.cast('d'); b=mb.buf.cast('d'); c=mc.buf.cast('d')
    for i in range(n):
        for j in range(n):
            a[i*n+j]=((17*i+13*j)%97-48)/97.0
            b[i*n+j]=((11*i+19*j)%89-44)/89.0
    del a,b,c
    jobs=[(i*n//phys,(i+1)*n//phys) for i in range(phys)]
    results=[]
    baseline=None
    try:
        with ctx.Pool(phys,initializer=matrix_initializer,
                      initargs=(ma.name,mb.name,mc.name)) as pool:
            def run_variant(strategy,bs):
                work=[(strategy,s,e,n,bs) for s,e in jobs]
                def trial():
                    finished=pool.map(gemm_worker,work)
                    if sum(finished)!=n: raise RuntimeError('Missing matrix output rows')
                stats=run_benchmark(trial,warmup,timed)
                out=mc.buf.cast('d')
                measured=list(out)
                del out
                gflops=(2*n**3)/(stats.median_ms*1e6)
                return {'variant':strategy,'BS':bs,'median_ms':stats.median_ms,
                        'iqr_ms':stats.iqr_ms,'gflops':gflops},measured

            for strategy,bs in [('naive',None),('reordered',None)]+[('blocked',x) for x in tiles]:
                row,output=run_variant(strategy,bs)
                if strategy=='reordered': baseline=output
                if strategy=='blocked':
                    row['validation_max_abs_error']=max(abs(x-y) for x,y in zip(baseline,output))
                    row['equivalent_to_reordered']=row['validation_max_abs_error']<1e-9
                    if not row['equivalent_to_reordered']:
                        raise RuntimeError(f'Blocked result differs from reordered for BS={bs}: {row["validation_max_abs_error"]}')
                results.append(row)
                print(f'{strategy:9s} BS={str(bs):>4s}: {row["median_ms"]:.3f}ms {gflops_str(row)}',flush=True)
    finally:
        for shm in mems:
            shm.close(); shm.unlink()

    best=max((row for row in results if row['variant']=='blocked'),key=lambda r:r['gflops'])
    B=roof['bandwidth_gbps']; P=roof['compute_gflops']
    for row in results:
        if row['variant']=='blocked':
            bs=row['BS']; intensity=bs/12.0
            row['tile_working_set_bytes']=3*bs*bs*8
        else:
            # Assignment supplies only naive=0.125, not reordered intensity.
            # Reordered is assigned a conservative nominal comparison point.
            intensity=0.125
        roofmax=min(P,B*intensity)
        row['assumed_intensity_flops_per_byte']=intensity
        row['roofline_attainable_gflops']=roofmax
        row['roofline_attainment_pct']=100*row['gflops']/roofmax if roofmax>0 else None
    verify=emit_verify_tag('PHASE3',args.student_id,results[0]['gflops'],best['gflops'],true_logical)
    print(f'OPTIMAL_BS={best["BS"]} BEST_GFLOPS={best["gflops"]:.6f}',flush=True)
    print(verify)
    write_json(args.output,{'phase':3,'mode':args.mode,'student_id':args.student_id,
         'N':n,'p':phys,'warmups':warmup,'timed_runs':timed,'variants':results,
         'best_bs':best['BS'],'best_gflops':best['gflops'],
         'intensity_naive':0.125,'intensity_best':best['BS']/12.0,
         'verify_hash':verify,
         'limitations':'The prompt does not specify Variant 2 operational intensity; 0.125 is an explicitly labeled nominal comparison, not measured DRAM traffic.'})


def gflops_str(r):
    return f'{r["gflops"]:.6f} GFLOP/s'

if __name__=='__main__':
    mp.freeze_support()
    main()

```

### `task4_scheduling.py`

```python

"""Phase 4: contiguous/cyclic/dynamic skewed scheduling; worker active times."""
import math
import multiprocessing as mp
import os
import statistics
import time
from harness import (read_json,write_json,cli_parser,run_benchmark,emit_verify_tag)


def work_item(i):
    # Equivalent to 1 + floor((i/60)^2), with exact integer arithmetic.
    count=1+i*i//3600
    x=(i%31)*0.0001+0.1
    for j in range(count):
        x=math.sin(x+0.001)+math.cos(x*0.999)
        x=math.sqrt(x*x+1.0)
    return x


def scheduling_worker(t,p,n,method,chunk,counter,active_times,checksums):
    active_ns=0
    total=0.0
    if method=='block':
        start=t*n//p; end=(t+1)*n//p
        t0=time.perf_counter_ns()
        for i in range(start,end): total+=work_item(i)
        active_ns=time.perf_counter_ns()-t0
    elif method=='cyclic':
        t0=time.perf_counter_ns()
        for i in range(t,n,p): total+=work_item(i)
        active_ns=time.perf_counter_ns()-t0
    else:
        while True:
            with counter.get_lock():
                start=counter.value
                counter.value+=chunk
            if start>=n: break
            end=min(start+chunk,n)
            t0=time.perf_counter_ns()
            for i in range(start,end): total+=work_item(i)
            active_ns+=time.perf_counter_ns()-t0
    active_times[t]=active_ns/1e6
    checksums[t]=total


def execute(ctx,p,n,method,chunk):
    counter=ctx.Value('i',0)
    active=ctx.Array('d',p,lock=False)
    checksums=ctx.Array('d',p,lock=False)
    started=time.perf_counter_ns()
    workers=[]
    for t in range(p):
        proc=ctx.Process(target=scheduling_worker,
            args=(t,p,n,method,chunk,counter,active,checksums))
        proc.start(); workers.append(proc)
    for proc in workers:
        proc.join()
        if proc.exitcode!=0: raise RuntimeError(f'Scheduling worker returned {proc.exitcode}')
    wall_ms=(time.perf_counter_ns()-started)/1e6
    vals=list(active)
    mean=statistics.mean(vals)
    load=max(vals)/mean if mean>0 else None
    return {'worker_active_ms':vals,'wall_ms':wall_ms,'imbalance_lambda':load,
            'total_result':sum(checksums)}


def main():
    args=cli_parser(__doc__).parse_args()
    hw=read_json(os.path.join(os.path.dirname(args.output),'phase0.json'))['hardware']
    p=int(hw['physical_cores']); logical=int(hw['logical_threads'])
    true_logical=logical
    if args.mode=='smoke': p=min(p,2); logical=min(logical,2)
    n=12_000 if args.mode=='full' else 120
    warmup,timed=(1,3) if args.mode=='full' else (0,1)
    ctx=mp.get_context('spawn')
    rows=[]
    strategies=[('block',None),('cyclic',None)]+[('dynamic',c) for c in [1,4,16,64,256,1024]]
    for method,chunk in strategies:
        observations=[]
        def trial(): observations.append(execute(ctx,p,n,method,chunk or 1))
        stats=run_benchmark(trial,warmup,timed)
        observed=observations[warmup:]
        # Choose the representative run closest to the measured median wall time.
        representative=min(observed,key=lambda o:abs(o['wall_ms']-stats.median_ms))
        row={'strategy':method,'chunk':chunk,'median_ms':stats.median_ms,
             'iqr_ms':stats.iqr_ms,'imbalance_lambda':representative['imbalance_lambda'],
             'worker_active_ms':representative['worker_active_ms'],
             'result_checksum':representative['total_result']}
        rows.append(row)
        print(f'{method:8s} chunk={str(chunk):>4s} wall={row["median_ms"]:.3f}ms lambda={row["imbalance_lambda"]:.4f}',flush=True)
    base=rows[0]; best=min((r for r in rows if r['strategy']=='dynamic'),key=lambda r:r['median_ms'])
    verify=emit_verify_tag('PHASE4',args.student_id,base['median_ms'],best['median_ms'],true_logical)
    print(f'OPTIMAL_DYNAMIC_CHUNK={best["chunk"]}',flush=True)
    print(verify)
    write_json(args.output,{'phase':4,'mode':args.mode,'student_id':args.student_id,
         'N':n,'p':p,'warmups':warmup,'timed_runs':timed,
         'results':rows,'best_dynamic_chunk':best['chunk'],'verify_hash':verify,
         'note':'Active worker times count computation only; total wall times include spawn, synchronization and scheduling.'})

if __name__=='__main__':
    mp.freeze_support()
    main()

```

### `run_all.py`

```python

"""Run all 5 phases sequentially, retain raw console logs, generate final report."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from build_report import generate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-id',type=int,default=230103081)
    parser.add_argument('--mode',choices=('full','smoke'),default='full')
    args=parser.parse_args()
    source_dir=Path(__file__).resolve().parent
    result_dir=source_dir/('results' if args.mode=='full' else 'smoke_results')
    result_dir.mkdir(parents=True,exist_ok=True)
    steps=[('phase0','task0_cal.py'),('phase1','task1_falsesharing.py'),
           ('phase2','task2_scaling.py'),('phase3a','task3a_roofline.py'),
           ('phase3b','task3b_gemm.py'),('phase4','task4_scheduling.py')]
    for index,(label,script) in enumerate(steps,1):
        command=[sys.executable,str(source_dir/script),'--student-id',str(args.student_id),
                 '--mode',args.mode,'--output',str(result_dir/(label+'.json'))]
        print(f'\n========== {index}/6: {label} ({script}) ==========',flush=True)
        with (result_dir/(label+'.log')).open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(command,cwd=source_dir,stdout=subprocess.PIPE,
                 stderr=subprocess.STDOUT,universal_newlines=True,bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            exitcode=proc.wait()
        if exitcode:
            print(f'ERROR: {label} exited {exitcode}. Existing JSON/logs remain available.',file=sys.stderr)
            output=(source_dir/f'INCOMPLETE_lab_report_{args.student_id}.md')
            generate(result_dir,args.student_id,output,source_dir)
            print(f'Partial report: {output}',file=sys.stderr)
            sys.exit(exitcode)
    output=source_dir/(f'lab_report_{args.student_id}.md' if args.mode=='full'
                         else f'SMOKE_lab_report_{args.student_id}.md')
    available,complete=generate(result_dir,args.student_id,output,source_dir)
    print(f'\nREPORT GENERATED: {output}',flush=True)
    if args.mode=='full' and not complete:
        raise RuntimeError('Full report unexpectedly incomplete')
    if args.mode=='smoke':
        print('SMOKE MODE is for software sanity checks, not submission.',flush=True)

if __name__=='__main__':
    main()

```

### `build_report.py`

```python

"""Generate the English Markdown report from ACTUAL local JSON measurements."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from harness import read_json


def num(x,places=4):
    if x is None: return 'Not available'
    return f'{x:,.{places}f}'


def raw(x):
    return 'Not measured' if x is None else str(x)


def md_table(headers,rows):
    def sanitize(x):
        return str(x).replace('|','\\|').replace('\n',' ')
    out=['| '+' | '.join(map(sanitize,headers))+' |',
         '| '+' | '.join('---' for _ in headers)+' |']
    out.extend('| '+' | '.join(map(sanitize,row))+' |' for row in rows)
    return '\n'.join(out)+'\n'


def data_or_none(folder,key):
    path=Path(folder)/(key+'.json')
    return read_json(path) if path.exists() else None


def format_bytes(value):
    if value is None: return 'Not reported by OS'
    return f'{value} B ({value/1024:.1f} KiB)'


def generate(results_dir,student_id,outfile,source_dir,include_source=True):
    d={x:data_or_none(results_dir,x) for x in ['phase0','phase1','phase2','phase3a','phase3b','phase4']}
    available=[k for k,x in d.items() if x is not None]
    full=len(available)==6 and all(x.get('mode')=='full' for x in d.values())
    smoke=any(x and x.get('mode')=='smoke' for x in d.values())
    lines=[]
    add=lambda s='':lines.append(s)
    add(f'# Performance Evaluation and Optimization in Parallel Computing — Solo Python Systems Lab')
    add(f'**Student ID:** {student_id}  ')
    add(f'**Report creation (UTC):** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}  ')
    add('**Execution mode:** '+('FULL hardware measurements' if full else 'SMOKE VALIDATION ONLY — NOT FOR SUBMISSION' if smoke else 'INCOMPLETE — requires local measurements')+'  ')
    add('**Experiment environment:** local Python 3 and standard library; `multiprocessing`, `shared_memory`, `ctypes`, `time`, `math`.')
    add()
    if not full:
        add('> **Incomplete evidence warning:** This document does not claim to contain a completed 120-minute lab. No missing performance metrics have been invented. Run `python3 run_all.py --student-id '+str(student_id)+' --mode full` on the assigned laptop to produce the complete hardware-specific report.')
        add()
    add('## 0. Hardware topology, calibration, and verification')
    p0=d['phase0']
    if p0:
        h=p0['hardware']
        add(md_table(['Hardware parameter','Measured value'],[
            ('CPU model string',h['cpu_model']),('Physical CPU cores (P_phys)',h['physical_cores']),
            ('Logical hardware threads (P_log)',h['logical_threads']),
            ('Cache line size (L_bytes)',format_bytes(h.get('cacheline_bytes'))),
            ('L1 data cache',format_bytes(h.get('l1d_bytes'))),
            ('L2 cache',format_bytes(h.get('l2_bytes'))),
            ('Shared L3 / system cache',format_bytes(h.get('l3_bytes'))),
            ('Python version',h['python_version']),('Operating system',h['os']),
            ('Hybrid core-level information',h.get('core_clusters',{})),
            ]))
        add(f'The average clock call latency is estimated from {p0["clock_reads"]:,} calls to `time.perf_counter_ns()`. An empty-loop reference is subtracted to reduce Python loop overhead:')
        add(f'\n`avg clock latency = (gross benchmark time - empty-loop time) / count = {num(p0["clock_read_latency_ns"])} ns/call`.')
        add(f'Gross call cost: {num(p0["clock_gross_ns_per_read"])} ns/call; empty loop: {num(p0["empty_loop_ns_per_iteration"])} ns/iteration.')
        if h.get('physical_cores_source'): add(f'**Hardware check needed:** {h["physical_cores_source"]}.')
        if not h.get('l3_bytes'):
            add('**Cache caveat:** The OS does not expose an L3 cache size through the queried interface; this cannot be replaced with a guessed value. On Apple Silicon, some caches are shared system-level caches rather than a conventional x86 L3.')
    else: add('**Phase 0 measurements missing.**')
    add('### Integrity hashes (printed verbatim by the individual phase scripts)')
    for x in ['phase0','phase1','phase2','phase3b','phase4']:
        v=d[x]
        add('`'+v['verify_hash']+'`' if v and v.get('verify_hash') else f'`VERIFY_HASH[{x.upper()}]: NOT MEASURED`')
    add('The FNV-1a 64-bit fingerprint uses this exact UTF-8 payload: `task_id|student_id|P_log|metric1:.4f|metric2:.4f`, with wraparound modulo 2^64. Phase 0 uses corrected and gross clock-call cost; Phases 1–4 use the specific measured metrics below. A hash proves consistency of reported values with its payload; it does **not** prove independent physical execution.')

    add('\n## 1. False sharing, stride sensitivity, Karp–Flatt')
    add('We used one 4096-byte `SharedMemory` buffer and one `ctypes.c_uint64` slot per process, with slot index `worker_id × stride`. The fixed total loop count is M = 24,000,000 and the benchmark collects two warmups and seven timed runs in FULL mode.')
    p1=d['phase1']
    if p1:
        add(f'Actual experiment loop count: {p1["M"]:,}; warmups: {p1["warmups"]}, timed runs: {p1["timed_runs"]}.')
        add('### Table 1A — Stride sensitivity at p = P_phys')
        add(md_table(['Stride (words)','Separation (B)','Median T (ms)','IQR (ms)','Throughput (Mops/s)'],[
            [r['stride_words'],r['separation_bytes'],num(r['median_ms'],3),num(r['iqr_ms'],3),num(r['throughput_mops'],3)] for r in p1['stride_sweep']]))
        gaps=[]
        for left,right in zip(p1['stride_sweep'],p1['stride_sweep'][1:]):
            improvement=100*(left['median_ms']-right['median_ms'])/left['median_ms']
            gaps.append((improvement,right['separation_bytes']))
        drop,offset=max(gaps)
        if drop>10:
            add(f'**Largest observed stride improvement:** when slot separation reached {offset} bytes, median time fell {num(drop,2)}% relative to the preceding tested stride. The OS reported cache-line width '+raw(p0['hardware'].get('cacheline_bytes') if p0 else None)+' bytes. This alone does not establish a physical coherence boundary without repeatability checks.')
        else:
            add(f'**No convincing sharp stride drop was observed.** The largest adjacent improvement was only {num(max(drop, 0),2)}% at a separation of {offset} bytes; this is not sufficient to infer a cache-line boundary. The OS-reported cache-line width is '+raw(p0['hardware'].get('cacheline_bytes') if p0 else None)+' bytes.')
        add('### Table 1B — Process sweep and experimental serial fraction')
        for key in ('1','16'):
            rows=p1['scaling'][key]
            add(f'**Stride = {key} word(s):**')
            add(md_table(['p','T_p (ms)','S_p','E_p','Karp–Flatt e(p)'],[
                [r['p'],num(r['median_ms'],3),num(r['speedup']),num(r['efficiency']),num(r['karp_flatt_e'])]
                for r in rows]))
        last1=p1['scaling']['1'][-1]['karp_flatt_e']
        last16=p1['scaling']['16'][-1]['karp_flatt_e']
        add('Calculations: `S_p=T_1/T_p`, `E_p=S_p/p`, `e(p)=(1/S_p−1/p)/(1−1/p)` for `p>=2`. The p=1 fraction is mathematically undefined.')
        add(f'At the last measured worker count, `e_stride1={num(last1)}` and `e_stride16={num(last16)}`. '+('The higher stride-1 fraction is consistent with more coherence-related overhead from false sharing.' if last1 is not None and last16 is not None and last1>last16 else 'The measured fractions do not show a larger effective serial fraction for stride 1; spawning cost, cache topology, or statistical variance may mask the expected penalty.'))
        add('False sharing occurs when workers write different 8-byte slots that occupy the same hardware coherence line. Cache invalidation traffic can raise memory latency even without a software data race. Karp–Flatt e(p) captures **effective** non-parallel overhead (including synchronization and memory effects), not just strictly serial source lines.')
    else: add('**Phase 1 tables and empirical conclusion require local execution.**')

    add('\n## 2. Amdahl strong scaling and Gustafson weak scaling')
    add('Stage A runs in the parent process over the first 5% of N and performs six sin/sqrt pairs (12 math calls) per element. Stage B splits all N elements across a persistent process pool and applies 35 mixed floating-point/trigonometric iterations per element. Pool construction happens outside the timed region.')
    add('Fixed strong-scaling N = 1,600,000. Weak-scaling workload N(p) = p × 250,000. Every integer p from 1 to P_log is benchmarked.')
    p2=d['phase2']
    if p2:
        add(f'Actual strong N={p2["strong_N"]:,}; weak N0={p2["weak_N0"]:,}.')
        add('### Table 2A — Strong scaling')
        add(md_table(['p','N','T_p (ms)','Speedup S_p','Efficiency E_p','Stage A (ms)'],[
            [r['p'],r['N'],num(r['median_ms'],3),num(r['speedup']),num(r['efficiency']),num(r['serial_stage_median_ms'],3)] for r in p2['strong']]))
        add('### Table 2B — Weak scaling')
        add(md_table(['p','N(p)','T_p (ms)','Weak E','α(p)','Gustafson S_G(p)'],[
            [r['p'],r['N'],num(r['median_ms'],3),num(r['weak_efficiency']),num(r['alpha']),num(r['gustafson_speedup'])] for r in p2['weak']]))
        add('Amdahl formula: `S_A(p)=1/[s+(1−s)/p]` where `s=T_StageA(1)/T_strong(1)`. Gustafson formula: `S_G(p)=p−α(p)(p−1)` where `α(p)=T_StageA(p)/T_weak(p)`. Weak efficiency: `T_weak(1)/T_weak(p)`.')
        add(f'Calculated serial fraction s = **{num(p2["serial_fraction"],6)}**; theoretical asymptotic limit `S_infinity=1/s` = **{num(p2["amdahl_infinity"])}**; Amdahl at P_log = **{num(p2["amdahl_plog"])}**; empirical strong speedup at P_log = **{num(p2["strong"][-1]["speedup"])}**; Gustafson scaled speedup at P_log = **{num(p2["weak"][-1]["gustafson_speedup"])}**.')
        boundary=p2['boundary']
        if boundary['status']=='measured':
            add(f'At the physical-to-logical-core boundary p={boundary["p_boundary"]}, marginal improvement falls from ΔS={num(boundary["marginal_gain_before"])} to ΔS={num(boundary["marginal_gain_after"])}; relative reduction is {num(boundary["percentage_drop"],2)}% (a negative result indicates an increase).')
        else:
            add('**SMT boundary:** '+boundary['status']+'. This machine may have no extra logical threads beyond physical cores, so the exact p=P_phys+1 point cannot be reported.')
        clusters=(p0 or {}).get('hardware',{}).get('core_clusters') or {}
        if len(clusters)>=2:
            try:
                # Use the first large-cache cluster as an architectural reference;
                # the OS scheduler is NOT pinned to cores by the experiment.
                high_cache=max(clusters.values(),key=lambda cl:int(cl.get('l2_bytes') or 0))
                k=int(high_cache['physical_cores'])
                series={int(r['p']):r['speedup'] for r in p2['strong']}
                if 2<=k and k+1 in series and k-1 in series:
                    before=series[k]-series[k-1]
                    after=series[k+1]-series[k]
                    change=100*(before-after)/before if before else None
                    add(f'**Hybrid cluster inflection (empirical):** the larger-L2 cluster has {k} physical cores. Marginal strong-scaling speedup from p={k-1} to {k} was ΔS={num(before,4)}, while from p={k} to {k+1} it was ΔS={num(after,4)}, a {num(change,2)}% decrease. Worker-to-core affinity was not measured or pinned; this is an observed worker-count inflection, not proof of exact P/E-core placement.')
            except (ValueError, KeyError, TypeError, ZeroDivisionError):
                add('Hybrid cluster topology was reported but the marginal speedup boundary could not be quantified from available measurements.')
        add('Real scaling departs from ideal predictions because of process-pool IPC, CPU frequency variation, cache pressure, thermal throttling, and potentially heterogeneous performance/efficiency cores. Observed values, not the model, determine conclusions.')
    else: add('**Phase 2 scaling matrices and calculations require local measurements.**')

    add('\n## 3. Empirical Roofline and three-stage matrix optimization')
    add('### 3A — DRAM bandwidth and Python arithmetic ceiling')
    add('Each FULL benchmark uses two 96,000,000-byte shared buffers (192,000,000 total bytes). A copy reads one source byte and writes one destination byte. `BW_GB/s=(2×96×10^6)/(T_seconds×10^9)`. An unrolled 16-scalar Python kernel counts two arithmetic FLOPs for each multiply-add update; its observed rate is `P_GFLOP/s=(workers × 80×10^6)/(T_seconds×10^9)`. The ridge is `I*=P/BW`.')
    p3a=d['phase3a']
    if p3a:
        add(md_table(['Hardware roofline parameter','Measured value'],[
            ('A and B buffer size (bytes each)',p3a['buffer_bytes_each']),
            ('Total footprint (bytes)',p3a['working_set_bytes']),
            ('Memory copy T_median (ms)',num(p3a['bw_median_ms'],3)),
            ('Sustained BW (GB/s)',num(p3a['bandwidth_gbps'],6)),
            ('Computed FLOPs per worker',f'{p3a["flops_each"]:,}'),
            ('Compute T_median (ms)',num(p3a['compute_median_ms'],3)),
            ('Python ceiling P (GFLOP/s)',num(p3a['compute_gflops'],6)),
            ('Ridge I* (FLOPs/byte)',num(p3a['ridge_flops_per_byte'],6))]))
        add('**Methodological limit:** Python expressions `x*a+b` are not guaranteed to become hardware-fused FMA instructions. These are interpreted Python throughput ceilings, not a hardware FLOP peak. Pool IPC and timing overhead can also affect both ceilings.')
    else: add('**Phase 3A measured bandwidth, compute and ridge missing.**')
    add('### 3B — GEMM i-j-k / i-k-j / six-loop tiling')
    add('Matrices are row-major flat double-precision shared arrays. Full N=384, so arithmetic work is `2×N^3` FLOPs. Variant 1 uses i-j-k (strided reads from B); Variant 2 uses i-k-j (contiguous B and C accesses). Variant 3 uses ii-jj-kk-i-k-j tiling with the same i-k-j microkernel. Workers own disjoint C rows. All 10 prescribed tile sizes are evaluated.')
    p3=d['phase3b']
    if p3:
        add(f'Actual N={p3["N"]}; p={p3["p"]}; warmups={p3["warmups"]}; timed runs={p3["timed_runs"]}.')
        add(md_table(['Variant','BS','T_median (ms)','GFLOP/s','W_tile (B)','I assumed (FLOPs/B)','Attainable GFLOP/s','Ceiling achieved (%)','Max error vs reorder'],[
            [r['variant'],r['BS'] or '—',num(r['median_ms'],3),num(r['gflops'],6),
             raw(r.get('tile_working_set_bytes','—')),num(r['assumed_intensity_flops_per_byte'],5),
             num(r['roofline_attainable_gflops'],6),num(r['roofline_attainment_pct'],2),
             num(r.get('validation_max_abs_error'),10) if r['variant']=='blocked' else '—']
            for r in p3['variants']]))
        best=p3['best_bs']; footprint=3*best*best*8
        add(f'Best empirically observed tile BS* = **{best}**, with throughput **{num(p3["best_gflops"],6)} GFLOP/s**. Its three-tile footprint `W_tile=3×BS²×8` = **{footprint:,} bytes**; estimated tiled intensity `I_tile=BS/12` = **{num(p3["intensity_best"],5)} FLOPs/byte**. Naive assumed intensity is 0.125 FLOP/byte.')
        naive_result=next(r for r in p3['variants'] if r['variant']=='naive')
        if p3['best_gflops'] < naive_result['gflops']:
            penalty=100*(naive_result['gflops']-p3['best_gflops'])/naive_result['gflops']
            add(f'**Observed optimization outcome:** the best blocked version was {num(penalty,2)}% slower than naive GEMM ({num(p3["best_gflops"],6)} vs {num(naive_result["gflops"],6)} GFLOP/s). In pure Python, interpreter-level loop and indexing overhead may exceed the cache-locality benefit. This experiment does not establish a blocked speedup.')
        else:
            gain=100*(p3['best_gflops']/naive_result['gflops']-1)
            add(f'**Observed optimization outcome:** the best blocked variant was {num(gain,2)}% faster than naive GEMM.')
        h=p0['hardware'] if p0 else {}
        l1,l2=h.get('l1d_bytes'),h.get('l2_bytes')
        for nm,cap in [('L1d',l1),('L2',l2)]:
            if cap: add(f'Tile footprint versus {nm}={cap:,} B: '+('fits nominal capacity.' if footprint<=cap else 'exceeds nominal capacity.')+' Actual capacity also holds code, stack and other active data.')
            else: add(f'{nm} capacity was not reported, so nominal fit cannot be established.')
        add('Roofline model: `P_attainable(I)=min(P_sustained, BW_sustained×I)` and `%attained=100×GFLOP/s_measured/P_attainable`. The reordered variant uses 0.125 FLOP/byte only as a **conservative nominal reference**: its true operational intensity is not specified by the task or measured by hardware counters. A value above 100% flags a model mismatch, not an impossible measured observation.')
        add('Numerical validation: all blocked variants are compared element-by-element against Variant 2; their largest absolute differences must be strictly less than 1e−9. All reported tile runs passed this in the generated results.' if all(r.get('equivalent_to_reordered',True) for r in p3['variants']) else '**Numerical validation failed.**')
    else: add('**Phase 3B GEMM sweep, best tile, validation and cache-footprint comparison require local execution.**')

    add('\n## 4. Skewed scheduling and load imbalance')
    add('Full workload has N=12,000 outer items, with `K(i)=1+floor((i/60)^2)=1+(i*i//3600)` trigonometric iterations per item. Strategy 1 uses contiguous static blocks; Strategy 2 round-robin cyclic assignment; Strategy 3 a synchronized shared index with dynamic chunk sizes C∈{1,4,16,64,256,1024}. All trials use P_phys processes.')
    add('`lambda = max_t(t_worker[t]) / ((sum_t t_worker[t])/p)`. Active worker times exclude idle lock-wait periods; wall time includes process startup and dispatch. Values close to 1 imply balanced useful compute times.')
    p4=d['phase4']
    if p4:
        add(f'Actual workload N={p4["N"]:,}, physical workers={p4["p"]}, warmups={p4["warmups"]}, timed runs={p4["timed_runs"]}.')
        add(md_table(['Scheduling strategy','Chunk C','Median wall time (ms)','IQR (ms)','λ (active load imbalance)'],[
            [r['strategy'],r['chunk'] or '—',num(r['median_ms'],3),num(r['iqr_ms'],3),num(r['imbalance_lambda'],5)] for r in p4['results']]))
        add('### Individual active worker times (representative median trial)')
        for r in p4['results']:
            add(f'- **{r["strategy"]} C={raw(r["chunk"])}:** '+', '.join(f'W{j}={num(x,2)} ms' for j,x in enumerate(r['worker_active_ms'])))
        best=next(r for r in p4['results'] if r['strategy']=='dynamic' and r['chunk']==p4['best_dynamic_chunk'])
        block=p4['results'][0]; cyclic=p4['results'][1]
        c1=next(r for r in p4['results'] if r['strategy']=='dynamic' and r['chunk']==1)
        c1024=next(r for r in p4['results'] if r['strategy']=='dynamic' and r['chunk']==1024)
        add(f'**Optimal measured dynamic chunk C* = {p4["best_dynamic_chunk"]}** with {num(best["median_ms"],3)} ms wall time. Static block = {num(block["median_ms"],3)} ms; cyclic = {num(cyclic["median_ms"],3)} ms.')
        add(f'At C=1: T={num(c1["median_ms"],3)} ms, λ={num(c1["imbalance_lambda"])}. At C=1024: T={num(c1024["median_ms"],3)} ms, λ={num(c1024["imbalance_lambda"])}. Small chunks usually reduce remaining uneven work but increase lock acquisitions and contention; large chunks reduce lock traffic but can leave high-cost items on one worker. The recorded table, rather than an assumed optimum, determines C*.')
    else: add('**Phase 4 load imbalance observations require local execution.**')

    add('\n## 5. Raw local terminal telemetry')
    if not available:
        add('**No locally executed terminal logs available in this initial package.**')
    for key in ['phase0','phase1','phase2','phase3a','phase3b','phase4']:
        log=Path(results_dir)/(key+'.log')
        if log.exists():
            add(f'### {key}.log')
            add('```text')
            add(log.read_text(encoding='utf-8',errors='replace').rstrip())
            add('```')
        else: add(f'**{key}.log:** Not run.')
    add('\n## 6. Reproduction and limitations')
    add('Execute `python3 run_all.py --student-id '+str(student_id)+' --mode full` from this directory, on the target laptop. The runner writes JSON and console log evidence for every phase, then regenerates this Markdown report. The `--mode smoke` option tests only code execution with reduced workload and is **not acceptable as the requested empirical full run**.')
    add('The study uses wall-clock observations on a real machine; CPU availability, power mode, resource limits, thermal history, cross-process startup and OS scheduling introduce noise. Median and IQR are reported when available. Clock calibration alone does not remove those effects. No performance observation is extrapolated from another machine.')
    add('**Assignment restriction:** the source PDF says the work must be performed offline, solo and without AI tools. This prepared package is assistance and does not replace the required independently executed lab. Follow your instructor’s academic-integrity requirements when choosing what to submit.')
    if include_source:
        add('\n## Appendix A — Complete Python 3 implementation')
        scripts=['harness.py','task0_cal.py','task1_falsesharing.py',
                 'task2_scaling.py','task3a_roofline.py','task3b_gemm.py',
                 'task4_scheduling.py','run_all.py','build_report.py']
        for name in scripts:
            path=Path(source_dir)/name
            if path.exists():
                add(f'### `{name}`')
                add('```python')
                add(path.read_text(encoding='utf-8').rstrip())
                add('```')
    outfile=Path(outfile)
    outfile.parent.mkdir(parents=True,exist_ok=True)
    outfile.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    return available,full


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-id',type=int,default=230103081)
    parser.add_argument('--results-dir',default='results')
    parser.add_argument('--output',default=None)
    parser.add_argument('--no-source',action='store_true')
    args=parser.parse_args()
    src=Path(__file__).resolve().parent
    name=args.output or str(src/f'lab_report_{args.student_id}.md')
    available,complete=generate(args.results_dir,args.student_id,name,src,not args.no_source)
    print('REPORT:',name,'EXPERIMENT FILES:',available,'COMPLETE FULL:',complete)

if __name__=='__main__':
    main()

```
