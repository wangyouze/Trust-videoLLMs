if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
# export CUDA_VISIBLE_DEVICES="0"
model_id=$1

dataset_ids=("NaturalNoiseMVBench"
            "Clean_MVBench")
for dataset_id in "${dataset_ids[@]}";
do
    python evaluate_tasks.py --config TrustVideoLLM/configs/robustness/OOD-noise.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/robustness/r2-noise-vqa/${model_id}/${dataset_id}.json"
done






