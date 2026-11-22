# TMLR reviewer experiment results

All intervals use independent training-seed summaries. Branch/episode counts are not the inference sample size.

## Fresh feasible sender subsets
|Condition|Weights|Mean best-worst suffix range [95% CI]|Normal selection hindsight regret|States with return effect (%)|
|---|---|---|---|---|
|slow_5ms|rr_cap|1.965 [1.519,2.412]|1.034|18.9|
|slow_5ms|dac|1.411 [0.941,1.880]|0.460|13.9|
|medium_2ms|rr_cap|1.828 [0.927,2.728]|0.725|17.0|
|medium_2ms|dac|2.098 [0.788,3.409]|1.203|18.2|

## Shared-prefix control
|Condition|Weights|Cold DAC-RR|Prefix DAC-RR total [95% CI]|Prefix survivors|
|---|---|---|---|---|
|slow_5ms|rr_cap|-0.850|+0.000 [0.000,0.000]|183/200|
|slow_5ms|dac|+0.200|+0.000 [0.000,0.000]|190/200|
|medium_2ms|rr_cap|-1.700|+0.350 [-0.235,0.935]|199/200|
|medium_2ms|dac|-0.200|-0.350 [-1.025,0.325]|200/200|

## Score scale and hindsight regret
|Condition|Weights|Mean gap|Mean age term|Age dominates fraction|alpha0 regret|alpha.03|alpha.1|alpha.3|alpha1|Normalized regret|Rescaling rank change|
|---|---|---|---|---|---|---|---|---|---|---|---|
|slow_5ms|rr_cap|0.106|0.159|0.575|0.801|0.845|0.908|0.943|0.943|0.801|0.519|
|slow_5ms|dac|0.092|0.165|0.608|0.473|0.430|0.460|0.489|0.489|0.473|0.565|
|medium_2ms|rr_cap|0.107|0.068|0.367|1.098|1.128|1.120|1.069|1.069|1.112|0.597|
|medium_2ms|dac|0.080|0.070|0.436|1.207|1.254|1.203|1.174|1.174|1.207|0.672|

## Withholding an actual fresh update
|Condition|Weights|Withhold minus normal Return [95% CI]|Mean absolute effect|Changed branch %|
|---|---|---|---|---|
|slow_5ms|rr_cap|+0.063 [-0.318,0.444]|1.088|10.9|
|slow_5ms|dac|-0.230 [-0.472,0.012]|0.646|6.5|
|medium_2ms|rr_cap|-0.158 [-0.380,0.064]|0.663|6.4|
|medium_2ms|dac|+0.242 [-0.017,0.501]|0.554|4.6|

Phase-stratified observational summaries are retained in JSON; conditioning on all-valid cache is not the common-prefix causal intervention and may select different trajectories.
## Broadcast pipeline
|Condition|Weights|Access|Source delivery %|Accuracy %|Reference %|Probe test %|
|---|---|---|---|---|---|---|
|slow_5ms|no_comm|normal|0.00|49.06|50.00|100.00|
|slow_5ms|no_comm|source_aware|0.00|49.06|50.00|100.00|
|slow_5ms|ideal_full|normal|100.00|100.00|100.00|100.00|
|slow_5ms|ideal_full|source_aware|100.00|100.00|100.00|100.00|
|slow_5ms|periodic|normal|10.00|50.31|55.00|100.00|
|slow_5ms|periodic|source_aware|100.00|71.56|100.00|100.00|
|slow_5ms|rr_cap|normal|20.00|53.36|60.00|100.00|
|slow_5ms|rr_cap|source_aware|100.00|78.79|100.00|100.00|
|slow_5ms|dac|normal|16.97|51.27|58.49|100.00|
|slow_5ms|dac|source_aware|100.00|71.32|100.00|100.00|
|medium_2ms|no_comm|normal|0.00|49.06|50.00|100.00|
|medium_2ms|no_comm|source_aware|0.00|49.06|50.00|100.00|
|medium_2ms|ideal_full|normal|100.00|100.00|100.00|100.00|
|medium_2ms|ideal_full|source_aware|100.00|100.00|100.00|100.00|
|medium_2ms|periodic|normal|30.00|51.96|65.00|100.00|
|medium_2ms|periodic|source_aware|100.00|57.29|100.00|100.00|
|medium_2ms|rr_cap|normal|40.00|56.13|70.00|100.00|
|medium_2ms|rr_cap|source_aware|100.00|65.04|100.00|100.00|
|medium_2ms|dac|normal|31.97|53.93|65.99|100.00|
|medium_2ms|dac|source_aware|100.00|65.47|100.00|100.00|
|medium_5ms|no_comm|normal|0.00|49.06|50.00|100.00|
|medium_5ms|no_comm|source_aware|0.00|49.06|50.00|100.00|
|medium_5ms|ideal_full|normal|100.00|100.00|100.00|100.00|
|medium_5ms|ideal_full|source_aware|100.00|100.00|100.00|100.00|
|medium_5ms|periodic|normal|50.00|53.17|75.00|100.00|
|medium_5ms|periodic|source_aware|100.00|57.03|100.00|100.00|
|medium_5ms|rr_cap|normal|50.00|58.04|75.00|100.00|
|medium_5ms|rr_cap|source_aware|100.00|68.47|100.00|100.00|
|medium_5ms|dac|normal|39.37|55.20|69.69|100.00|
|medium_5ms|dac|source_aware|100.00|65.63|100.00|100.00|

## Paired source-aware access effects
|Condition|Weights|Accuracy change percentage points [95% CI]|Source-aware current delivery %|
|---|---|---|---|
|slow_5ms|periodic|+21.25 [6.01,36.49]|100.00|
|slow_5ms|rr_cap|+25.43 [12.93,37.93]|100.00|
|slow_5ms|dac|+20.04 [4.99,35.10]|100.00|
|medium_2ms|periodic|+5.34 [0.20,10.48]|100.00|
|medium_2ms|rr_cap|+8.91 [2.25,15.56]|100.00|
|medium_2ms|dac|+11.54 [0.84,22.23]|100.00|
|medium_5ms|periodic|+3.86 [0.30,7.41]|100.00|
|medium_5ms|rr_cap|+10.44 [2.50,18.38]|100.00|
|medium_5ms|dac|+10.43 [0.65,20.21]|100.00|

## Readiness boundary
|Weights|Deadline ms|Mode|Return|K delivered|Readiness us|Gate us|
|---|---|---|---|---|---|---|
|rr_cap|1.5|measured|6.300|0.921|331.780|126.444|
|rr_cap|1.5|fixed|5.800|1.000|500.000|124.352|
|rr_cap|1.75|measured|6.000|1.011|333.112|125.609|
|rr_cap|1.75|fixed|5.800|1.000|500.000|123.606|
|rr_cap|2.0|measured|5.800|1.903|338.352|126.525|
|rr_cap|2.0|fixed|7.900|2.000|500.000|124.856|
|rr_cap|2.25|measured|7.300|2.175|344.580|128.654|
|rr_cap|2.25|fixed|7.900|2.000|500.000|125.561|
|rr_cap|2.5|measured|6.500|2.495|348.375|130.279|
|rr_cap|2.5|fixed|6.500|2.495|500.000|131.517|
|dac|1.5|measured|6.500|0.914|409.945|210.941|
|dac|1.5|fixed|5.500|1.000|500.000|209.565|
|dac|1.75|measured|5.500|0.999|411.052|210.266|
|dac|1.75|fixed|5.500|1.000|500.000|207.349|
|dac|2.0|measured|7.200|1.925|404.171|206.203|
|dac|2.0|fixed|7.800|2.000|500.000|208.121|
|dac|2.25|measured|8.000|1.999|411.181|208.423|
|dac|2.25|fixed|7.800|2.000|500.000|206.270|
|dac|2.5|measured|7.200|2.499|414.594|210.015|
|dac|2.5|fixed|7.200|2.499|500.000|209.005|

## Paired readiness effects
|Weights|Deadline ms|Measured minus fixed Return [95% CI]|K difference|
|---|---|---|---|
|rr_cap|1.5|+0.500 [-0.195,1.195]|-0.079|
|rr_cap|1.75|+0.200 [-0.612,1.012]|+0.011|
|rr_cap|2.0|-2.100 [-3.509,-0.691]|-0.097|
|rr_cap|2.25|-0.600 [-1.871,0.671]|+0.175|
|rr_cap|2.5|+0.000 [0.000,0.000]|+0.000|
|dac|1.5|+1.000 [-0.816,2.816]|-0.086|
|dac|1.75|+0.000 [-0.337,0.337]|-0.001|
|dac|2.0|-0.600 [-1.566,0.366]|-0.075|
|dac|2.25|+0.200 [-0.457,0.857]|-0.001|
|dac|2.5|+0.000 [0.000,0.000]|+0.000|

The warm-prefix unconditional effect includes zero suffix effects for episodes terminating before branching. Survivor-only effects are retained separately in JSON. Neither is a pure coverage-mediated effect: payload, aggregation denominator and validity features jointly change.
Source-aware fixes the realized packet count at the visited state; it can change later histories. True-bit reference ignores signaling through silence. The probe tests linear decodability, not optimal decoding.
Alpha/normalized/reward-scale diagnostics use every sampled state; they are explanatory comparisons, not a tuned winning scheduler.
Current CPU is single-host data. Fixed-readiness measurements are not deployable latency.
