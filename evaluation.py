import os
old_hf_home = os.getenv('HF_HOME', default='')
old_transformers_cache = os.getenv('TRANSFORMERS_CACHE', default='')
old_tmpdir = os.getenv('TMPDIR', default='')

os.environ['HF_HOME'] = '/scr/scr-with-most-space/amazon-sentiment/cache'
os.environ['TRANSFORMERS_CACHE'] = '/scr/scr-with-most-space/amazon-sentiment/cache'
os.environ['TMPDIR'] = '/scr/scr-with-most-space/amazon-sentiment/tmp'

from docopt import docopt
import pathlib
from preprocessing import generate_preprocessed_dataset
from transformers import RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import classification_report, f1_score
import utils
import torch

import numpy as np
from datasets import load_metric

from transformers import TrainingArguments

metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, pos_label=None, average='macro')

verbose = True

def main(args):
    model_path = args['<model-name>']
    assess_domains = args['<eval-domain>']
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)
    training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

    score_func=utils.safe_macro_f1    
    
    predictions = []
    scores = []
    for dataset_num, assess_domain in enumerate(assess_domains, start=1):
        assess_dataset = generate_preprocessed_dataset(
            assess_domain,
            int(args['--dataset-size']),
            random_seed=int(args['--seed']),
            mode='test'
        )
        preds = []
        for i in range(0, int(args['--dataset-size']), 128):
#         trainer = Trainer(
#             model=model,
#             args=training_args,
#             eval_dataset=assess_dataset,
#             compute_metrics=compute_metrics,
#         )
#         results = trainer.evaluate()
#         if len(assess_domains) > 1: 
#             print("Assessment dataset {}".format(dataset_num))
#         print(results)
        #import ipdb; ipdb.set_trace()
            assess_dataset_fragment = assess_dataset.select(list(range(i, min(i + 128, int(args['--dataset-size'])))))
            input_ids = torch.LongTensor(assess_dataset_fragment['input_ids'])
            attention_mask=torch.FloatTensor(assess_dataset_fragment['attention_mask'])
            if torch.cuda.is_available():
                model = model.cuda()
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()

            with torch.no_grad():    
                classifier_output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = classifier_output.logits
            preds = preds + list(torch.argmax(logits, axis=1).cpu())

        if verbose:
            if len(assess_domains) > 1:
                print("Assessment dataset {}".format(dataset_num))
            labels = torch.LongTensor(assess_dataset['label'])

            print(classification_report(labels, preds, digits=3))
        predictions.append(preds)
        scores.append(score_func(labels, preds))
    
    if len(scores) > 1 and verbose:
        mean_score = np.mean(scores)
        print("Mean of macro-F1 scores: {0:.03f}".format(mean_score))

usage='''Final Project Evaluation
Usage:
    evaluation.py <model-name> <eval-domain>... [--dataset-size=<dataset-size>] [--seed=<seed>]

--dataset-size=<dataset-size>    [default: 2000]
--seed=<seed>                    [default: 420]
'''
if __name__ == '__main__':
    args = docopt(usage, version='Evaluation 1.0')
    main(args)

