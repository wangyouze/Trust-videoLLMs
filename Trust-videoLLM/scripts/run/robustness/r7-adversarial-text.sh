if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1
# export CUDA_VISIBLE_DEVICES="1"
dataset_ids=AdversarialTextDataset
for dataset_id in "${dataset_ids[@]}";
do
    python evaluate_tasks.py --config TrustVideoLLM/configs/robustness/robustness-adversarial-texts.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/robustness/r7-adversarial-texts/${model_id}/${dataset_id}.json"
done