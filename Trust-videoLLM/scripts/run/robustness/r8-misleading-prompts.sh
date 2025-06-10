if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
export CUDA_VISIBLE_DEVICES="0"
model_id=$1

dataset_id=Misleading-Prompt-Videos

python run_task.py --config TrustVideoLLM/configs/robustness/robustness-misleading-prompt-videos.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/robustness/r8-misleading-prompts/${model_id}/${dataset_id}.json"