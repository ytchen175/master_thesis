#!/bin/bash
start_time=$(date +%s)
python_script="use_qlora_to_inference.py"

echo "[Start] Starting infer all models..."

for ft_num in {250..10000..250}
do
    python "$python_script" --ft_num "$ft_num"

    sleep 1
done

end_time=$(date +%s)

execution_time=$((end_time - start_time))
execution_time_minutes=$(echo "scale=2; $execution_time / 60" | bc)

echo "[Finished] All models has been inferred, cost ${execution_time_minutes} mins"