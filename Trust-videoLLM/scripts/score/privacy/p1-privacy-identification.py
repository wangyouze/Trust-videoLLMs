import argparse
import json
import os
from glob import glob
from pprint import pprint

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, default='LLaVA-Video-7B-Qwen2', help='Model ID to be processed')
args = parser.parse_args()


jsonlist = glob("logs/privacy/p1-privacy-identification-BIV-Priv/{}/*.json".format(args.model_id))
outfile = "results/privacy/p1-privacy-identification/{}/p1-privacy-identification.json".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id

keyname_mapping = {
    "rule_yes_or_no_eval:accuracy_score": "accuracy_score",
    "rule_yes_or_no_eval:precision_score": "precision_score",
    "rule_yes_or_no_eval:recall_score": "recall_score",
    "rule_yes_or_no_eval:f1_score": "f1_score",
}

results = {}
results['model_id'] = model_id
results['scores'] = {}
for jsonfile in jsonlist:
    filename = os.path.splitext(os.path.basename(jsonfile))[0]
    with open(jsonfile, 'r') as fp:
        data = json.load(fp)

        results['scores'][filename] = {}
        for keyname in keyname_mapping.keys():
            newkeyname = keyname_mapping[keyname]
            results['scores'][filename][newkeyname] = round(data['total_results'][keyname], 4)

pprint(results)
os.makedirs(os.path.dirname(outfile), exist_ok=True)

with open(outfile, 'w') as fp:
    json.dump(results, fp, indent=4)