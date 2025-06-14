if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
# export CUDA_VISIBLE_DEVICES="2"
model_id=$1

dataset_id=Privacy_QA

python run_task.py --config TrustVideoLLM/configs/privacy/privacy-VQA.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/privacy/p1-privacy-VQA/${model_id}/${dataset_id}.json"