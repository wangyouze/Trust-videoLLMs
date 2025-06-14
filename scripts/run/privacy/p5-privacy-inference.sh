if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
# export CUDA_VISIBLE_DEVICES="3"
model_id=$1

dataset_id=privacy_inference_openvid

python run_task.py --config TrustVideoLLM/configs/privacy/privacy-inference.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/privacy/p4-privacy-inference-openvid/${model_id}/${dataset_id}.json"