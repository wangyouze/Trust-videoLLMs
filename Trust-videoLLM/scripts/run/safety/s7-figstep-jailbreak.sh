if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=SafeBench

python run_task.py --config TrustVideoLLM/configs/safety/safety-figstep.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/safety/s6-figstep/${model_id}/${dataset_id}.json"