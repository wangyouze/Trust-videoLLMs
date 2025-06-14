
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice
from statistics import mean
from typing import Any, Sequence, List, Tuple, Dict, Optional
from TrustVideoLLM.evaluators.base import BaseEvaluator
from TrustVideoLLM.utils.registry import registry


# nltk.download('punkt')
# nltk.download('wordnet')

@registry.register_evaluator()
class BLEUEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['bleu_meteor_cider_spice_eval']
    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any]) -> None:
        super().__init__(evaluator_id, metrics_cfg)

    

    def process(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:

        smoothie = SmoothingFunction().method4
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        meteor_scorer = Meteor()
        cider_scorer = Cider()
        # spice_scorer = Spice()

       
        results = []
        gts = {}  # ground truth references
        res = {}  # results/hypotheses
    

        for idx, (hypothesis, references, extra) in enumerate(zip(preds, labels, extras)):
            assert isinstance(hypothesis, str)

            ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())
            bleu_score = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=smoothie)
            
            # 2. ROUGE-L
            rouge_l_scores = [rouge_scorer_obj.score(ref, hypothesis)['rougeL'].fmeasure 
                            for ref in references]
            rouge_l = mean(rouge_l_scores)
            
           
            gts[idx] = references
            res[idx] = [hypothesis]
            
            results.append({
                'BLEU': bleu_score,
                'ROUGE-L': rouge_l
            })
            

        meteor_score, meteor_scores = meteor_scorer.compute_score(gts, res)
        cider_score, cider_scores = cider_scorer.compute_score(gts, res)
        # spice_score, spice_scores = spice_scorer.compute_score(gts, res)
        
        for idx, result in enumerate(results):
            result['METEOR'] = meteor_scores[idx]
            result['CIDEr'] = cider_scores[idx]
            # result['SPICE'] = spice_scores[idx]
        
        metrics = {
            'BLEU': mean(result['BLEU'] for result in results),
            'ROUGE-L': mean(result['ROUGE-L'] for result in results),
            'METEOR': meteor_score,
            'CIDEr': cider_score,
            # 'SPICE': spice_score
        }

        
        return metrics, labels, extras