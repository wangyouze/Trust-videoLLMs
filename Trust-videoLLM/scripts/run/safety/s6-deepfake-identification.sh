if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=DeepFake

python run_task.py --config TrustVideoLLM/configs/safety/safety-deepfake.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/safety/s5-deepfake-identification/${model_id}/${dataset_id}.json"