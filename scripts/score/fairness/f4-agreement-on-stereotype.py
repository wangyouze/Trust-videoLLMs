import argparse
import json
import os
from glob import glob
from pprint import pprint

parser = argparse.ArgumentParser(description='Process JSON files for fairness analysis.')
parser.add_argument('--model_id', type=str, help='Model ID to be processed')
args = parser.parse_args()

jsonlist = glob("logs/fairness/f4-agreement-on-stereotype/{}/*.json".format(args.model_id))
outfile = "results/fairness/f4-agreement-on-stereotype/{}/f4-agreement-on-stereotype.json".format(args.model_id)

out_dir = os.path.dirname(outfile)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_id = args.model_id

keyname_mapping = {
    "rule_yes_or_no_eval:pred_mean": "Agreement_Rate_Average",
    "rule_yes_or_no_eval:pred_mean_Race": "Agreement_Rate_Race",
    "rule_yes_or_no_eval:pred_mean_Socioeconomic": "Agreement_Rate_Socioeconomic",
    "rule_yes_or_no_eval:pred_mean_Gender": "Agreement_Rate_Gender",
    "rule_yes_or_no_eval:pred_mean_Age": "Agreement_Rate_Age",
    "rule_yes_or_no_eval:pred_mean_Nationality": "Agreement_Rate_Nationality",
    "rule_yes_or_no_eval:pred_mean_Religion": "Agreement_Rate_Religion"
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