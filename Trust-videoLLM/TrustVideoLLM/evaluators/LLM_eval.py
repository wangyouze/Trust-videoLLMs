

from typing import Any, Sequence, List, Tuple, Dict, Optional
from TrustVideoLLM.evaluators.base import BaseEvaluator
from TrustVideoLLM.utils.registry import registry
import string

@registry.register_evaluator()
class ChatModelEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_eval']
    def __init__(self, evaluator_id: str, model_id: str, prompt_template: str, generation_kwargs: Dict[str, Any], metrics_cfg: Dict[str, Any], device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from TrustVideoLLM.models import load_chatmodel
        self.evaluator_id = evaluator_id
        self.chatmodel = load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs

        formatter = string.Formatter()
        self.prompt_template_fields = [fname for _, fname, _, _ in formatter.parse(self.prompt_template) if fname]
    
    def get_prompt(self, pred, label, extra):
        prompt_params = {}
        for key in self.prompt_template_fields:
            if extra == 'first' or extra == 'all':
                prompt_params['pred1'] = pred
                prompt_params['pred2'] = label
            elif key == "pred":
                prompt_params[key] = pred
            elif key == "label":
                prompt_params[key] = label
            elif isinstance(label, list):
                prompt_params['label1'] = label[0]
                prompt_params['label2'] = label[1]
                prompt_params['label3'] = label[2]
                prompt_params['label4'] = label[3]
                prompt_params['label5'] = label[4]
            elif isinstance(label, dict):
                prompt_params['Pred'] = pred
                prompt_params['Question'] = label['mc_question']

            elif extra is not None and key in extra:
                prompt_params[key] = extra[key]
            else:
                raise KeyError("Fail to format the prompt. Can not find the specific field in pred/label/extra.")
        prompt = self.prompt_template.format(**prompt_params)
        return prompt

    def process(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        for (pred, label, extra) in zip(preds, labels, extras):
            assert isinstance(pred, str)

            eval_prompt = self.get_prompt(pred, label, extra)
            
            eval_message = [
                                {
                                    "role": "user",
                                    "content": eval_prompt
                                }
                            ]
            eval_response = self.chatmodel.chat(messages=eval_message, **self.generation_kwargs)
            print(eval_response.content)
            print('pred:', pred)
            print('label:', label)
            print('--------------------------------------------------')
            processed_preds.append(eval_response.content)
        return processed_preds, labels, extras
    

@registry.register_evaluator()
class LLMToxicityEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['LLM_toxicity_score_eval']
    def __init__(self, evaluator_id: str, model_id: str, prompt_template: str, generation_kwargs: Dict[str, Any], metrics_cfg: Dict[str, Any], device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from TrustVideoLLM.models import load_chatmodel
        self.evaluator_id = evaluator_id
        self.chatmodel = load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs


    def process(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        for (pred, label, extra) in zip(preds, labels, extras):
            assert isinstance(pred, str)
            prompt_params = {}
            prompt_params['pred'] = pred
        
            prompt = self.prompt_template.format(**prompt_params)

            
            eval_message = [
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ]
            eval_response = self.chatmodel.chat(messages=eval_message, **self.generation_kwargs)
            print(eval_response.content)
            print('pred:', pred)
            print('label:', label)
            print('--------------------------------------------------')

            if eval_response.content.startswith('1') or '1' in eval_response.content:
                res = 1
            elif eval_response.content.startswith('0') or '0' in eval_response.content:
                res = 0 
            else:
                print(f"Unrecognized response format: {eval_response.content}")
                res = 0


            processed_preds.append(res)
        return processed_preds, labels, extras