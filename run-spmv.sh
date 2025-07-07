#!/bin/bash

export OMP_NUM_THREADS=96
export GOMP_CPU_AFFINITY="0-95"
export OMP_PROC_BIND=close

export matrices=("DK01R" "GT01R" "PR02R" "RM07R" "HV15R" "dendrimer" "nv1" "nv2" "sme3Da" "sme3Db" "sme3Dc" "vibrobox" "k3plates" "m3plates" "rail_5177" "rail_79841" "epb2" "epb3")

for i in ${matrices[@]}; do
	echo "Matrix: $i"
	./spmv ./$i/${i}.mtx #| grep "Performance:" | awk '{print $2}'
        echo "*******"	
done

