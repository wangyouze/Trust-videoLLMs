#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
# export CUDA_VISIBLE_DEVICES="2"
model_id=$1


dataset_id=celebrities
data_type_ids=(
    "personal-email-name-occupation"
    "personal-email-wo-name-occupation"
    "personal-phone-name-occupation"
    "personal-phone-wo-name-occupation"
    "personal-address-name-occupation"
    "personal-address-wo-name-occupation"
    "personal-email-name-wo-occupation"
    "personal-email-wo-name-wo-occupation"
    "personal-phone-name-wo-occupation"
    "personal-phone-wo-name-wo-occupation"
    "personal-address-name-wo-occupation"
    "personal-address-wo-name-wo-occupation"
    )

for data_type_id in "${data_type_ids[@]}";
do
    echo "Processing data_type_id: ${data_type_id}" 
    python run_task.py --config TrustVideoLLM/configs/privacy/privacy-celebrities.yaml --cfg-options \
        model_id=${model_id} \
        dataset_id=${dataset_id} \
        dataset_cfg.data_type_id=${data_type_id} \
        log_file="logs/privacy/p3-celebrities/${model_id}/${dataset_id}-${data_type_id}.json"
done

