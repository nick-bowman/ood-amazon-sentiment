# from transformers import Trainer, TrainingArguments
# from transformers import DataCollatorForLanguageModeling
# from transformers import RobertaTokenizer, RobertaForMaskedLM
# from datasets import load_from_disk
# from docopt import docopt
# import pathlib
# import shutil


# GLOBAL_DATASET_PATH = pathlib.Path('/juice/scr/rmjones/amazon-sentiment/processed_datasets')
# DATASET_FOLDER = pathlib.Path('/scr/scr-with-most-space/amazon-sentiment/processed_datasets')


# def main(args):
#     if not DATASET_FOLDER.exists():
#         DATASET_FOLDER.parent.mkdir(parents=True)
#         shutil.copytree(GLOBAL_DATASET_PATH, DATASET_FOLDER)

#     dataset_name = args['<dataset-name>']
#     experiment_name = args['<experiment-name>']
#     experiment_path = pathlib.Path('results') / experiment_name
#     overwrite = bool(args['--overwrite'])
#     experiment_path.mkdir(parents=True, exist_ok=True)
#     dataset_path = DATASET_FOLDER / f'{dataset_name}_Processed'
#     encoded_dataset = load_from_disk(str(dataset_path))
#     encoded_dataset = encoded_dataset.remove_columns('label')
#     training_args = TrainingArguments(
#         output_dir=str(experiment_path),
#         overwrite_output_dir=overwrite,
#         num_train_epochs=int(args['--epochs']),
#         per_device_train_batch_size=4,
#         save_steps=500,
#         save_total_limit=2,
#         seed=1
#     )
#     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, mlm=True, mlm_probability=float(args['--mlm_probability'])
#     )
#     model = RobertaForMaskedLM.from_pretrained('roberta-base')
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=encoded_dataset
#     )
#     trainer.train("checkpoint-973500")
#     trainer.save_model()

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
from transformers import RobertaForMaskedLM, RobertaTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

def main(args):
    dataset_name = args['<dataset-name>']
    experiment_name = args['<experiment-name>']
    experiment_path = pathlib.Path('results') / experiment_name
    overwrite = bool(args['--overwrite'])
    dataset = generate_preprocessed_dataset(
        args['<dataset-name>'],
        int(args['--dataset-size']),
        random_seed=int(args['--seed'])
    )
    dataset = dataset.remove_columns('label')

    training_args = TrainingArguments(
        output_dir=str(experiment_path),
        overwrite_output_dir=overwrite,
        num_train_epochs=int(args['--epochs']),
        per_device_train_batch_size=int(args['--batch-size']),
        save_steps=500,
        save_total_limit=1,
        seed=int(args['--seed'])
    )
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=float(args['--mlm_probability'])
    )
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )
    trainer.train()
    trainer.save_model()

usage='''RoBERTa (re)-pretraining
Usage:
    pretrain_roberta.py <dataset-name> <experiment-name> [--overwrite] [--dataset-size=<dataset-size>] [--epochs=<epochs>] [--seed=<seed>] [--batch-size=<batch-size>] [--mlm_probability=<prob>]

--dataset-size=<dataset-size>    [default: 2000]
--epochs=<epochs>                [default: 100]
--seed=<seed>                    [default: 420]
--batch-size=<batch-size>        [default: 4]
--mlm_probability=<prob>         [default: .15]
'''

if __name__ == '__main__':
    args = docopt(usage, version='Pretrain RoBERTa 1.0')
    main(args)
    
