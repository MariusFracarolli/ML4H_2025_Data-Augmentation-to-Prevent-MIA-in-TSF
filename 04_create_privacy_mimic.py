import os
import shutil
from itertools import product

# Standard values
method = 'only'
lam_values = [0]
mu_values = [0]
alpha_values = [1]
MSE_sign_values = [1]
pca_ratios = [0]
kappas = [0]
num_pert_values = [3]
num_branch_values = [5]
num_batcheses = [1000]
syn_ratios = [0.5]
beta_values = [1]
seed_value = 21
noise_multipliers = [0]
clip_norms = [0]

# # Parameterwerte für lam, mu, alpha
# lam_values = [3000, 30000]#,10000, 30000]
# mu_values = [3, 30, 300]
# alpha_values = [1, 0.75, 0.5, 0]
# # kappa_values = [0]
# MSE_sign_values = [1, -1]#[1, -1]
# # 12 models time 6 --> 72:
# # num_pert_values = [3,5]
# # num_branch_values = [5,10,30]
# pca_ratios = [0]#[0.8, 0.95]

# # lam = 10000
# # mu = 100 # real number
# # alpha = 1 # between 0 (only PL) and 1 (only MSE)
# kappa = 0
# # MSE_sign = -1 # or -1 - is fixed
# # num_pert = 3
# # num_branch = 10 
# # num_batches = 100 # how many new synthetic data per epoch * 32 - is fixed

# # all zoo models
# methods = ['zoo']
# lam_values = [3000, 30000]#,10000, 30000]
# mu_values = [3, 30, 300]
# alpha_values = [1, 0.5, 0]
# MSE_sign_values = [1, -1]

# # all pca models
# methods = ['pca']
# lam_values = [3000]#,10000, 30000]
# mu_values = [300]
# alpha_values = [1, 0.5, 0]
# MSE_sign_values = [1, -1]#[1, -1]
# pca_ratios = [0.6, 0.9]
# seed_value = 0

# # all mixup models
# methods = ['mixup']
# beta_values = [0.2, 0.5, 1, 2, 5]


# Ordnername und Dateipfade
source_folder = '03_privacy/0_start_folder'
script_filename = '04_DA.py'
slurm_filename = '02_train_model.sh'
end_folder = '03_privacy'
dir_train = "../../02_train/"

def writer(text):
    with open('00_sbatch_writer.txt', 'a') as file:
        file.write(f'{text}\n')

# Hauptfunktion zum Erstellen der Ordner und Modifizieren der Dateien
def create_variants():
    method_counter = 1
    for method, lam, mu, alpha, MSE_sign, pca_ratio, syn_ratio, num_batches, kappa, beta, clip_norm, noise_multiplier in \
        product(methods, lam_values, mu_values, alpha_values, MSE_sign_values, pca_ratios, syn_ratios, num_batcheses, kappas, beta_values, clip_norms, noise_multipliers):
        if MSE_sign == -1:
            num_pert = 3
            num_branch = 10
        else:
            num_pert = 10
            num_branch = 20

        # Neuen Ordnernamen erstellen
        new_folder = f'{end_folder}/{job_counter:02d}_{method}_{method_counter:02d}'
        if not os.path.exists(new_folder):
            shutil.copytree(source_folder, new_folder)
                            
        if job_counter == 1:
            writer(f"cd {new_folder}; sbatch {slurm_filename};")    
        else:
            writer(f"cd ../../{new_folder}; sbatch {slurm_filename};")
        # 241019_hyp.py anpassen
        modify_hyp_file(new_folder, seed_value, method,  lam, mu, alpha, num_pert, num_branch, MSE_sign, pca_ratio, syn_ratio, num_batches, kappa, beta, clip_norm, noise_multiplier)
        
        # run_new_MSE-py.sh anpassen
        modify_slurm_file(new_folder, method)
        
        print(f'Ordner {new_folder} erstellt.')
        method_counter += 1 

# Funktion zum Modifizieren von 241019_hyp.py
def modify_hyp_file(folder, seed_value, method,  lam, mu, alpha, num_pert, num_branch, MSE_sign, pca_ratio, syn_ratio, num_batches, kappa, beta, clip_norm, noise_multiplier):
    # folder = method + '/' + folder
    file_path = os.path.join(folder, script_filename)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Zeilen anpassen
    for i, line in enumerate(lines):
        if line.startswith('seed_value ='):
            lines[i] = f'seed_value = {seed_value}\n'
        if line.startswith('method ='):
            lines[i] = f"method = '{method}'\n"
        elif line.startswith('mu ='):
            lines[i] = f'mu = {mu}\n'
        elif line.startswith('lam ='):
            lines[i] = f'lam = {lam}\n'
        elif line.startswith('alpha ='):
            lines[i] = f'alpha = {alpha}\n'
        elif line.startswith('num_pert ='):
            lines[i] = f'num_pert = {num_pert}\n'
        elif line.startswith('num_branches ='):
            lines[i] = f'num_branches = {num_branch}\n'
        elif line.startswith('kappa ='):
            lines[i] = f'kappa = {kappa}\n'
        elif line.startswith('MSE_sign ='):
            lines[i] = f'MSE_sign = {MSE_sign}\n'
        elif line.startswith('pca_ratio ='):
            lines[i] = f'pca_ratio = {pca_ratio}\n'
        elif line.startswith('syn_ratio ='):
            lines[i] = f'syn_ratio = {syn_ratio}\n'
        elif line.startswith('num_batches ='):
            lines[i] = f'num_batches = {num_batches}\n'
        elif line.startswith('beta ='):
            lines[i] = f'beta = {beta}\n'
        elif line.startswith('dir_train ='):
            lines[i] = f"dir_train = '{dir_train}'\n"
        elif line.startswith('noise_multiplier ='):
            lines[i] = f"noise_multiplier = {noise_multiplier}\n"
        elif line.startswith('clip_norm ='):
            lines[i] = f"clip_norm = {clip_norm}\n"
        # elif line.startswith(' ='):
        #     lines[i] = f" = '{}'"
                
        
    
    # Geänderte Datei schreiben
    with open(file_path, 'w') as file:
        file.writelines(lines)

# Funktion zum Modifizieren von run_new_MSE-py.sh
def modify_slurm_file(folder, method):
    global job_counter
    file_path = os.path.join(folder, slurm_filename)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Zeile mit #SBATCH --job-name anpassen
    for i, line in enumerate(lines):
        if line.startswith('#SBATCH --job-name='):
            lines[i] = f'#SBATCH --job-name={method[0]}{job_counter}\n'
        if line.startswith('#SBATCH --mem='):
            lines[i] = f'#SBATCH --mem=145000\n'

    job_counter += 1
    # Geänderte Datei schreiben
    with open(file_path, 'w') as file:
        file.writelines(lines)

    

# job_counter = 1


# methods = ['only']
# create_variants()

job_counter = 33

syn_ratios = [0.5]
# # all zoo models
# methods = ['zoo']
# lam_values = [3000]#,10000, 30000]
# mu_values = [300]
# alpha_values = [1, 0]
# MSE_sign_values = [1, -1]
# create_variants()


# # all pca models
# methods = ['pca']
# lam_values = [3000]#,10000, 30000]
# mu_values = [300]
# alpha_values = [1, 0]
# MSE_sign_values = [1, -1]
# pca_ratios = [0.7]
# create_variants()


# # all mixup models
# methods = ['mixup']
# alpha_values = [0]
# MSE_sign_values = [0]
# beta_values = [0.2, 1, 5]
# create_variants()

# job_counter = 33
# # all dp-sgd models
# methods = ['dp-sgd']
# noise_multipliers = [1.1, 1.5, 2.0]
# clip_norms = [1.1, 1.5, 2.0]
# create_variants()
# noise_multipliers = [0]
# clip_norms = [0]


# # all pca models
# methods = ['pca']
# lam_values = [3000]#,10000, 30000]
# mu_values = [300]
# alpha_values = [1, 0]
# MSE_sign_values = [1, -1]
# pca_ratios = [0.5, 0.9]
# create_variants()


job_counter = 50
# all dp-sgd models
methods = ['dp-sgd']
noise_multipliers = [5]
clip_norms = [1.1, 1.5, 2.0]
create_variants()
noise_multipliers = [0]
clip_norms = [0]
