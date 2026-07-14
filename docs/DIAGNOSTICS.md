# TrionScope Diagnostic Rules

Complete reference for all diagnostic rules.

## Rule Format

Each rule has:
- **ID:** TS### (e.g., TS001)
- **Category:** Type of issue (Occupancy, Memory, Registers, etc.)
- **Severity:** error, warning, or info
- **Basis:** artifact, computed, or heuristic

## Rules by Category

### Occupancy (TS1xx)

#### TS101: Very Low Occupancy
- **Severity:** ERROR
- **Basis:** computed
- **Trigger:** Occupancy < 25%
- **Evidence:** Too few blocks per SM
- **Fix:** Reduce BLOCK_SIZE or increase num_warps

#### TS102: Low Occupancy
- **Severity:** WARNING
- **Basis:** computed
- **Trigger:** Occupancy 25-50%
- **Evidence:** Below optimal blocks per SM
- **Fix:** Tune BLOCK_SIZE for better utilization

### Registers (TS2xx)

#### TS201: Deep Loop Nesting
- **Severity:** WARNING
- **Basis:** heuristic
- **Trigger:** Loop depth > 2
- **Evidence:** Nested loops reduce occupancy
- **Fix:** Flatten or simplify loop structure

#### TS202: Complex Branching
- **Severity:** INFO
- **Basis:** heuristic
- **Trigger:** Branch depth > 3
- **Evidence:** Branches create control flow divergence
- **Fix:** Simplify branching if possible

### Register Pressure (TS3xx)

#### TS103: High Register Pressure
- **Severity:** WARNING
- **Basis:** heuristic
- **Trigger:** Register usage > 80% limit
- **Evidence:** Estimated high register allocation
- **Fix:** Reduce BLOCK_SIZE or simplify logic

## False Positive Rates

| Rule | Rate | Notes |
|------|------|-------|
| TS101 | <1% | Artifact-based |
| TS102 | 2-3% | Depends on actual compiled kernel |
| TS201 | ~10% | Loop depth heuristic |
| TS202 | ~5% | Branch heuristic |
| TS103 | 15-20% | Register estimates vary by compiler |

## Roadmap (Future Phases)

**Phase 4:** 30+ rules covering:
- Memory bandwidth utilization
- Cache efficiency
- Shared memory usage
- Instruction throughput
- Control flow efficiency

**Phase 10:** Empirical validation study showing impact of each rule on actual performance.
