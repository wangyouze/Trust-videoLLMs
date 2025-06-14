import argparse
import json
import os
from glob import glob
from pprint import pprint

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, default='LLaVA-Video-7B-Qwen2', help='Model ID to be processed')
args = parser.parse_args()


jsonlist = glob("logs/truthfulness/t4-video-hallucination/{}/*.json".format(args.model_id))
outfile = "results/truthfulness/t4-video-hallucination/{}t4-video-hallucination".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id

keyname_mapping = {
    "hallucination_metrics_eval:pred_no_op": "accuracy",
    "bias_metrics_eval:pred_no_op": "accuracy",
}

results = {}
results['model_id'] = model_id
results['scores'] = {}
for jsonfile in jsonlist:
    # filename = os.path.splitext(os.path.basename(jsonfile))[0]
    with open(jsonfile, 'r') as fp:
        data = json.load(fp)
        for k,v in data['total_results'].items():
            filename = k.split('_')[0]
            results['scores'][filename] = {}
        for keyname in keyname_mapping.keys():
            filename = keyname.split('_')[0]
            newkeyname = keyname_mapping[keyname]
            results['scores'][filename][newkeyname] = data['total_results'][keyname]

pprint(results)
os.makedirs(os.path.dirname(outfile), exist_ok=True)

with open(outfile, 'w') as fp:
    json.dump(results, fp, indent=4)