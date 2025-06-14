if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
# export CUDA_VISIBLE_DEVICES="3"
model_id=$1

dataset_id=RiskyContentIdentificationDataset

python run_task.py --config TrustVideoLLM/configs/safety/safety-VideoRiskyContentIdentification.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/safety/s2-identification-video-risky-video/${model_id}/${dataset_id}.json"