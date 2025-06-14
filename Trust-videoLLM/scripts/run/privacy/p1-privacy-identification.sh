if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=privacy-identification-BIV-Priv

python run_task.py --config TrustVideoLLM/configs/privacy/privacy-identification.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/privacy/p1-privacy-identification-BIV-Priv/${model_id}/${dataset_id}.json"