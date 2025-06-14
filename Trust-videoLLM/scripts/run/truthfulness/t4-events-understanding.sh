if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
# export CUDA_VISIBLE_DEVICES="2"
model_id=$1

dataset_id=YouCook2_sampled

python run_task.py --config TrustVideoLLM/configs/truthfulness/truthfulness-events-understanding.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/truthfulness/t3-events-understanding/${model_id}/${dataset_id}.json"