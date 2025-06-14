if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=hallucination

python run_task.py --config TrustVideoLLM/configs/truthfulness/truthfulness-video-hallucination.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/truthfulness/t4-video-hallucination/${model_id}/${dataset_id}.json"