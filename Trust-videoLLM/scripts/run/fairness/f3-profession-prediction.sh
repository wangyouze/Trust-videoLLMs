if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "profession-pred"
    "profession-pred-with-description"
)


for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config TrustVideoLLM/configs/fairness/fairness-profession-prediction.yaml --cfg-options \
        model_id=${model_id} \
        dataset_id=${dataset_id} \
        log_file="logs/fairness/f3-profession-pred/${model_id}/${dataset_id}.json"
done

