if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
export CUDA_VISIBLE_DEVICES="0"
model_id=$1

dataset_ids=("confaide-video"
            "confaide-text"
            "confaide-unrelated-video-color"
            "confaide-unrelated-video-natural"
            "confaide-unrelated-video-noise"
            )


for dataset_id in "${dataset_ids[@]}";
do
    python evaluate_tasks.py --config TrustVideoLLM/configs/privacy/privacy-InfoFlow-Expectation.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/privacy/p2-privacy-infoflow-expection/${model_id}/${dataset_id}.json"
done


