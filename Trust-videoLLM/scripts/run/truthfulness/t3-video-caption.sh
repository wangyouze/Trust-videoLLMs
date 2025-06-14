if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=OpenVid

python run_task.py --config TrustVideoLLM/configs/truthfulness/truthfulness-videocaptioning.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/truthfulness/t2-video-caption/${model_id}/${dataset_id}.json"