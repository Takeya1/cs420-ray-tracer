#!/usr/bin/env bash
set -euo pipefail

SCENES=(simple medium complex)

for sc in "${SCENES[@]}"; do
    echo "-> Rendering $sc"
    out=$(./ray_cuda "scenes/${sc}.txt" 2>&1 || true)
    echo "---- output for $sc ----"
    echo "$out"
    # extract the line 'GPU rendering time: <seconds> seconds'
    gpu_time=$(echo "$out" | grep "GPU rendering time" || true)
    if [[ -n "$gpu_time" ]]; then
        # extract the numeric value
        num=$(echo "$gpu_time" | awk '{for(i=1;i<=NF;i++) if($i=="time:") print $(i+1); else if($(i)=="time:") print $(i+1)}')
        # fallback: extract any floating number in the line
        if [[ -z "$num" ]]; then
            num=$(echo "$gpu_time" | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
        fi
        echo "  GPU rendering time for $sc: ${num:-(not found)} seconds"
    else
        echo "  GPU rendering time for $sc: (not found)"
    fi

    # move ppm and convert
    mv output_gpu.ppm output_gpu_${sc}.ppm 2>/dev/null || true
    if command -v magick >/dev/null 2>&1; then
        magick output_gpu_${sc}.ppm output_gpu_${sc}.png 2>/dev/null || true
    else
        if command -v convert >/dev/null 2>&1; then
            convert output_gpu_${sc}.ppm output_gpu_${sc}.png 2>/dev/null || true
        fi
    fi

done
