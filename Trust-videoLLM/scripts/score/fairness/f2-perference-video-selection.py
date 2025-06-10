import argparse
import json
import os
from glob import glob
from pprint import pprint

import numpy as np

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, default='LLaVA-Video-7B-Qwen2', help='Model ID to be processed')
args = parser.parse_args()

jsonlist = glob("logs/fairness/f3-perference-video-selection/{}/*.json".format(args.model_id))
outfile = "results/fairness/f3-perference-video-selection/{}/f3-perference-video-selection.json".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id

keyname_mapping = {
    "rule_reject_template_eval:pred_mean" : "reject_rate",
    "classifer_reject_eval:pred_mean": "reject_acc"
}



results = {}
results['model_id'] = model_id
results['scores'] = {}
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, 'w') as fp:
    for jsonfile in jsonlist:
        filename = os.path.splitext(os.path.basename(jsonfile))[0]
        with open(jsonfile, 'r') as data_file:
            data = json.load(data_file)

            file_scores = {}
            for keyname, newkeyname in keyname_mapping.items():
                file_scores[newkeyname] = round(data['total_results'][keyname], 4)
            
            results['scores'][filename] = file_scores

    pprint(results)
    json.dump(results, fp, indent=4)



