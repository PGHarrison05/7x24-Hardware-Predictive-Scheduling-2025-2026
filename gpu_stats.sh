#!/bin/bash

while true
do
    clear
    echo "GPU Monitoring"
    echo "----------------------------"

    nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,utilization.gpu \
    --format=csv,noheader

    sleep 1
done
