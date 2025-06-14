if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
# export CUDA_VISIBLE_DEVICES="2"
model_id=$1

dataset_ids=("UntargetedAttackDataset"
            "Clean_untargetedAttackDataset"
        )           
for dataset_id in "${dataset_ids[@]}";
do
    python evaluate_tasks.py --config TrustVideoLLM/configs/robustness/OOD-adversarial-attack.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/robustness/r4-untargeted-attacks/${model_id}/${dataset_id}.json"
done