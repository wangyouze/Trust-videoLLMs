if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "toxicity-prompt-text"
    "toxicity-prompt-video"
    "toxicity-prompt-unrelated"
)

for dataset_id in "${dataset_ids[@]}";
do
    python evaluate_tasks.py --config TrustVideoLLM/configs/safety/safety-RealToxicityPrompts.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/safety/s4-toxicity-content-generation/${model_id}/${dataset_id}.json"
done