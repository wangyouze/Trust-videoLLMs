if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=TimeSensitivity

python run_task.py --config TrustVideoLLM/configs/fairness/fairness-time-sensitivity.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/fairness/f5-time-sensitivity/${model_id}/${dataset_id}.json"