def generate_label(overall_score):
    if overall_score == None: return 'MISFORMED'
    overall_score = int(overall_score)
    if overall_score < 3: return 0
    elif overall_score == 3: return 1
    else: return 2

def generate_preprocessed_dataset(domain_name, dataset_size, mode='train', random_seed=420, roberta_model='roberta-base'):
    import os
    old_hf_home = os.getenv('HF_HOME', default='')
    old_transformers_cache = os.getenv('TRANSFORMERS_CACHE', default='')
    old_tmpdir = os.getenv('TMPDIR', default='')

    os.environ['HF_HOME'] = '/scr/scr-with-most-space/amazon-sentiment/cache'
    os.environ['TRANSFORMERS_CACHE'] = '/scr/scr-with-most-space/amazon-sentiment/cache'
    os.environ['TMPDIR'] = '/scr/scr-with-most-space/amazon-sentiment/tmp'

    import transformers
    from transformers import RobertaTokenizer, RobertaForMaskedLM
    from docopt import docopt
    from datasets import load_dataset, ClassLabel

    DATA_BASE = '/scr/scr-with-most-space/amazon-sentiment/full_datasets'
    MAX_SEQ_LENGTH = 512
#     NUM_CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK'))
    NUM_CPUS = 1
    
    tokenizer = RobertaTokenizer.from_pretrained(roberta_model)

    dataset = load_dataset('json', data_files=os.path.join(DATA_BASE, f'{domain_name}.json.gz'), split='train')
    dataset = dataset.shuffle(seed=random_seed)
    if mode == 'train':
        dataset = dataset.select(list(range(0, dataset_size)))
    elif mode == 'test':
        dataset = dataset.select(list(range(dataset.num_rows - dataset_size, dataset.num_rows)))
    else:
        print('Mode not recognized, operating on full dataset.')
    columns_to_remove = list(dataset.features.keys())
    columns_to_remove.remove('reviewText')
    columns_to_remove.remove('summary')
    columns_to_remove.remove('overall')
    dataset = dataset.remove_columns(columns_to_remove)
    dataset = dataset.map(lambda row: {'text': (row['summary'] if row['summary'] else '') + '. ' + (row['reviewText'] if row['reviewText'] else ''), 'label': generate_label(row['overall'])}, num_proc=NUM_CPUS)
    print('Num negative: ', len(dataset.filter(lambda example: example['label'] == 0)))
    print('Num neutral: ', len(dataset.filter(lambda example: example['label'] == 1)))
    print('Num positive: ', len(dataset.filter(lambda example: example['label'] == 2)))
                                

    dataset = dataset.map(lambda row: tokenizer(row['text'], padding='max_length', truncation=True, return_tensors='pt'), num_proc=NUM_CPUS)
    dataset = dataset.map(lambda row: {'input_ids': row['input_ids'][0], 'attention_mask': row['attention_mask'][0]}, num_proc=NUM_CPUS)
#     dataset = dataset.filter(lambda row: len(row['input_ids']) <= MAX_SEQ_LENGTH)
    columns_to_remove = ['reviewText', 'summary', 'overall']
    dataset = dataset.remove_columns(columns_to_remove)    
    new_features = dataset.features.copy()
    new_features['label'] = ClassLabel(names=['negative', 'neutral', 'positive'])
    dataset = dataset.cast(new_features)
    

    os.environ['HF_HOME'] = old_hf_home
    os.environ['TRANSFORMERS_CACHE'] = old_transformers_cache
    os.environ['TMPDIR'] = old_tmpdir
    
    return dataset


