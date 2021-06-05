def generate_label(overall_score):
    if overall_score == None: return 'MISFORMED'
    overall_score = int(overall_score)
    if overall_score < 3: return 'negative'
    elif overall_score == 3: return 'neutral'
    else: return 'positive'

def main(args):
    import os
    old_hf_home = os.getenv('HF_HOME', default='')
    old_transformers_cache = os.getenv('TRANSFORMERS_CACHE', default='')
    old_tmpdir = os.getenv('TMPDIR', default='')

    os.environ['HF_HOME'] = '/scr/amazon-sentiment/cache'
    os.environ['TRANSFORMERS_CACHE'] = '/scr/amazon-sentiment/cache'
    os.environ['TMPDIR'] = '/scr/amazon-sentiment/tmp'

    import transformers
    from transformers import RobertaTokenizer, RobertaForMaskedLM
    from docopt import docopt
    from datasets import load_dataset

    DATA_BASE = '/scr/amazon-sentiment'
    MAX_SEQ_LENGTH = 512
    NUM_CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK'))
    
    roberta_model = 'roberta-base'
    if args['--roberta-model']: roberta_model = args['--roberta-model']
    dataset_name = args['<dataset-name>']
    tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
    dataset = load_dataset('json', data_files=os.path.join(DATA_BASE, f'{dataset_name}.json.gz'), split='train')
    columns_to_remove = list(dataset.features.keys())
    columns_to_remove.remove('reviewText')
    columns_to_remove.remove('summary')
    columns_to_remove.remove('overall')
    dataset = dataset.remove_columns(columns_to_remove)
    dataset = dataset.map(lambda row: {'text': (row['summary'] if row['summary'] else '') + '. ' + (row['reviewText'] if row['reviewText'] else ''), 'label': generate_label(row['overall'])}, num_proc=NUM_CPUS)
    dataset = dataset.map(lambda row: tokenizer(row['text']), batched=True, num_proc=NUM_CPUS)
    dataset = dataset.filter(lambda row: len(row['input_ids']) <= MAX_SEQ_LENGTH)
    columns_to_remove = ['reviewText', 'summary', 'overall']
    dataset = dataset.remove_columns(columns_to_remove)    
    dataset.save_to_disk(os.path.join(DATA_BASE, f'{dataset_name}_Processed'))

    os.environ['HF_HOME'] = old_hf_home
    os.environ['TRANSFORMERS_CACHE'] = old_transformers_cache
    os.environ['TMPDIR'] = old_tmpdir


usage ="""CS224U Amazon Sentiment Dataset Preprocessing

Usage: 
    preprocess_amazon_dataset <dataset-name> [--roberta-model=<roberta-model>]

"""
if __name__ == '__main__':
    arguments = docopt(usage, version="CS224U HW3 Amazon Dataset Preprocessing")
    main(arguments)

