#!/bin/bash

#export BINARY="./spmv-base"
#export BINARY="./spmv-znver4"
export BINARY="./spmv-znver4-opt"

export GOMP_CPU_AFFINITY="0-23"
export OMP_NUM_THREADS=24 
export OMP_PROC_BIND=close;


#Linux perf commands for profiling

#perf stat -e cycles,instructions,branches,branch-misses,stalled-cycles-frontend,stalled-cycles-backend,cache-references,cache-misses,page-faults,alignment-faults,ls_hw_pf_dc_fills.all numactl --physcpubind=${GOMP_CPU_AFFINITY} --membind 0 $BINARY HV15R/HV15R.mtx

#perf stat -e cycles,instructions,branches,branch-misses,stalled-cycles-frontend,stalled-cycles-backend,cache-references,cache-misses,page-faults,alignment-faults,ls_hw_pf_dc_fills.all numactl --physcpubind=${GOMP_CPU_AFFINITY} --membind 0 $BINARY PR02R/PR02R.mtx

#perf stat -M frontend_bound_group,backend_bound_group numactl --physcpubind=${GOMP_CPU_AFFINITY} --membind 0 $BINARY HV15R/HV15R.mtx
#perf stat -M frontend_bound_group,backend_bound_group numactl --physcpubind=${GOMP_CPU_AFFINITY} --membind 0 $BINARY PR02R/PR02R.mtx

#perf stat -M PipelineL1,PipelineL2 numactl --physcpubind=${GOMP_CPU_AFFINITY} --membind 0 $BINARY HV15R/HV15R.mtx
#perf stat -e L1-dcache-loads,L1-dcache-load-misses,l2_cache_req_stat.dc_access_in_l2,l2_cache_req_stat.dc_hit_in_l2,l2_cache_req_stat.ls_rd_blk_c,l3_lookup_state.l3_hit,l3_lookup_state.l3_miss numactl --physcpubind=${GOMP_CPU_AFFINITY} --membind 0 $BINARY HV15R/HV15R.mtx

numactl --physcpubind=${GOMP_CPU_AFFINITY} --membind 0 $BINARY HV15R/HV15R.mtx
