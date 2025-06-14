if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "stereo-agreement-video"
    "stereo-agreement-unrelated-video-color"
    "stereo-agreement-unrelated-video-natural"
    "stereo-agreement-unrelated-video-noise"
)


for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config TrustVideoLLM/configs/fairness/fairness-stereo-agreement.yaml --cfg-options \
        model_id=${model_id} \
        dataset_id=${dataset_id} \
        log_file="logs/fairness/f4-agreement-on-stereotype/${model_id}/${dataset_id}.json"
done

