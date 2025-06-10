if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
# export CUDA_VISIBLE_DEVICES="1"
model_id=$1

dataset_id=StereotypicalGenerationDataset

python run_task.py --config TrustVideoLLM/configs/fairness/fairness_stereotype_Impact.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/fairness/f1-stereotype-impact-generation/${model_id}/${dataset_id}.json"