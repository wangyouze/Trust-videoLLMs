import argparse
import json
import os
from glob import glob
from pprint import pprint

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, default='LLaVA-Video-7B-Qwen2', help='Model ID to be processed')
args = parser.parse_args()


jsonlist = glob("logs/robustness/r4-untargeted-attacks/{}/*.json".format(args.model_id))
outfile = "results/robustness/r4-untargeted-attacks/{}/r4-untargeted-attacks.json".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id

keyname_mapping = {
    "choice_tasks_eval:pred_mean": "accuracy",
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