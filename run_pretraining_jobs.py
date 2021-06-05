import subprocess
import shlex
import sys

MEM_REQUESTS = {'Appliances': 4, 'Electronics': 20, 'Home_and_Kitchen': 20, 'Movies_and_TV': 8, 'Books': 64}
DATASETS = ['Appliances', 'Electronics', 'Home_and_Kitchen', 'Movies_and_TV', 'Books']
EPOCHS = [10]
SIZES = [2000, 5000]
SEED = 420
#HIDDEN_DIMS = [200, 300]
#DROPOUT_PROB = 0.25
#GLOVE_FILENAMES = ['glove.6B.{}d.txt'.format(dim) for dim in (200, 300)]
#BIDIRECTIONAL = False
#NUM_LAYERS = 2

if __name__ == '__main__':
    for dataset in DATASETS:
        for epochs in EPOCHS:
            for size in SIZES:
                experiment_name = 'pre'
                experiment_name += f'_dataset{dataset}'
                experiment_name += f'_size{size}'
                experiment_name += f'_epochs{epochs}'
                experiment_name += f'_seed{SEED}'

                python_command = 'python pretrain_roberta.py'
                python_command += f' {dataset}'
                python_command += f' {experiment_name}'
                python_command += f' --dataset-size={size}'
                python_command += f' --epochs={epochs}'
                python_command += f' --seed={SEED}'

                slurm_command = f'sbatch -J {experiment_name} -o logs/%x.out --mem={MEM_REQUESTS[dataset]}GB run_sbatch.sh "{python_command}"'
                if len(sys.argv) > 1 and sys.argv[1] == '--dryrun':
                    print(slurm_command)
                else:
                    subprocess.run(shlex.split(slurm_command))

"""    for hidden_dim in HIDDEN_DIMS:
        for glove_filename in GLOVE_FILENAMES:
            experiment_name = f'hidden{hidden_dim}'
            experiment_name += f'_dropout{DROPOUT_PROB}'
            experiment_name += f'_{glove_filename}'
            experiment_name += f'_bidirectional{False}'
            experiment_name += f'_numlayers{NUM_LAYERS}'
            output_filename = f'results/{experiment_name}.txt'
            python_command = 'python run_color_experiment.py'
            python_command += f' {output_filename} {hidden_dim}'
            python_command += f' {DROPOUT_PROB} {glove_filename}'
            python_command += f' {BIDIRECTIONAL} {NUM_LAYERS}'
            slurm_command = f'sbatch -J {experiment_name} -o logs/%x.out run_sbatch.sh "{python_command}"'
            subprocess.run(shlex.split(slurm_command))
"""
