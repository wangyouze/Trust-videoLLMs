
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
export CUDA_VISIBLE_DEVICES="1"
model_id=$1

dataset_id=VQA_Contextual

python run_task.py --config TrustVideoLLM/configs/truthfulness/truthfulness-vqa-contextual.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/truthfulness/t1-VQA-contextual/${model_id}/${dataset_id}.json"