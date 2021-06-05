import subprocess
import shlex
import sys

MEM_REQUESTS = {'Appliances': 4, 'Electronics': 20, 'Home_and_Kitchen': 20, 'Movies_and_TV': 8, 'Books': 64}
DATASETS = list(MEM_REQUESTS.keys())
DATASETS.remove('Books')
DATASET_SIZES = [2000]

if __name__ == '__main__':
    for dataset in DATASETS:
        for dataset_size in DATASET_SIZES:
            experiment_name = f'dataset{dataset}_datasetsize{dataset_size}_lr1e-3_wd1e-6'
            python_command = 'python finetune_roberta.py'
            python_command += f' {dataset}'
            python_command += f' {experiment_name}'
            python_command += f' --dataset-size={dataset_size}'
            python_command += f' --learning-rate=1e-3'
            python_command += f' --weight-decay=1e-6'
            #python_command += f' --checkpoint=./results/pre_dataset{dataset}_size{dataset_size}_epochs10_seed420'
            slurm_command = f'sbatch -J {experiment_name} -o logs/%x.out --mem={MEM_REQUESTS[dataset]}GB --gres=gpu:2 run_sbatch.sh "{python_command}"'
            if len(sys.argv) > 1 and sys.argv[1] == '--dryrun':
                print(slurm_command)
            else:
                subprocess.run(shlex.split(slurm_command))
