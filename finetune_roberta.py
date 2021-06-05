import os
old_hf_home = os.getenv('HF_HOME', default='')
old_transformers_cache = os.getenv('TRANSFORMERS_CACHE', default='')
old_tmpdir = os.getenv('TMPDIR', default='')

os.environ['HF_HOME'] = '/scr/scr-with-most-space/amazon-sentiment/cache'
os.environ['TRANSFORMERS_CACHE'] = '/scr/scr-with-most-space/amazon-sentiment/cache'
os.environ['TMPDIR'] = '/scr/scr-with-most-space/amazon-sentiment/tmp'

from datasets import load_metric
from docopt import docopt
import numpy as np
import pathlib
from preprocessing import generate_preprocessed_dataset
from transformers import RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments


def main(args):
    experiment_name = args['<experiment-name>']
    if args['--full-finetune']:
        experiment_name += '_finetune'
    else:
        experiment_name += '_linear'
    experiment_path = pathlib.Path('results') / experiment_name
    overwrite = bool(args['--overwrite'])
    experiment_path.mkdir(parents=True, exist_ok=True)
    dataset = generate_preprocessed_dataset(
        args['<dataset-name>'],
        int(args['--dataset-size']),
        random_seed=int(args['--seed'])
    )
    
    eval_dataset = generate_preprocessed_dataset(
        args['<dataset-name>'],
        int(args['--dataset-size']),
        random_seed=int(args['--seed']),
        mode='test'
    )
    
    training_args = TrainingArguments(
        output_dir=str(experiment_path),
        overwrite_output_dir=overwrite,
        num_train_epochs=int(args['--epochs']),
        per_device_train_batch_size=int(args['--batch-size']),
        save_steps=500,
        save_total_limit=1,
        seed=int(args['--seed']),
        evaluation_strategy='epoch',
        learning_rate=float(args['--learning-rate']),
        weight_decay=float(args['--weight-decay'])
    )
    
    if args['--checkpoint']:
        pretrained_model = RobertaForSequenceClassification.from_pretrained(pathlib.Path(args['--checkpoint']), num_labels=3)
    else:
        pretrained_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
    if not args['--full-finetune']:
        for param in pretrained_model.roberta.parameters():
            param.requires_grad = False
    
    metric = load_metric('accuracy')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model=pretrained_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model()
    

usage='''RoBERTa finetuning
Usage:
    finetune_roberta.py <dataset-name> <experiment-name> [--overwrite] [--full-finetune] [--dataset-size=<dataset-size>] [--epochs=<epochs>] [--seed=<seed>] [--batch-size=<batch-size>] [--checkpoint=<checkpoint>] [--learning-rate=<lr>] [--weight-decay=<l2>]

--dataset-size=<dataset-size>    [default: 2000]
--epochs=<epochs>                [default: 100]
--seed=<seed>                    [default: 420]
--batch-size=<batch-size>        [default: 128]
--learning-rate=<lr>             [default: 5e-5]
--weight-decay=<l2>              [default: 0.0]
'''
if __name__ == '__main__':
    args = docopt(usage, version='Finetune RoBERTa 1.0')
    main(args)
