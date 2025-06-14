if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
# export CUDA_VISIBLE_DEVICES="2,3"
model_id=$1

dataset_id=NSFWVideoDataset

python run_task.py --config TrustVideoLLM/configs/safety/safety-NSFW.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/safety/s1-nsfw-video-description/${model_id}/${dataset_id}.json"