# %%
import os
from datetime import datetime
import importlib
import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd 
import torch  # type: ignore
from torch.optim.lr_scheduler import StepLR  # type: ignore
from utils.config import get_parser
import models.InformerAutoregressive_emb as autoformer
from utils.pca import pca_sklearn, noise_significant_dims
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm  # type: ignore
tqdm.pandas()

def writer(text):
    current_date = datetime.now().strftime("%y%m%d-%H%M")
    with open("00_results.txt", "a") as file:
        file.write(f"{current_date}: {text}\n")
        print(text)


def writer2(text):
    with open("01_resultlist.txt", "a") as file:
        file.write(f"{text}\n")


def eval_set(model, emb_ev, fore, batch_size, args):
    loss_list, old_loss_list = [], []
    model.eval()
    pbar2 = tqdm(range(0, len(fore), batch_size), file=open(os.devnull, "w"))
    for start in pbar2:
        matrix = torch.tensor(fore[start : start + batch_size]).cuda()
        output_matrix = matrix[:, 24:, :V]
        output_mask = matrix[:, 24:, V:]
        emb = torch.tensor(emb_ev[start : start + batch_size]).cuda()
        output, _, _ = model(emb, 0, 0, 0, 0, trainn=False)
        loss = (output_mask[:, -args.pred_len :, :] 
                * (output - output_matrix[:, -args.pred_len :, :]) ** 2)
        old_loss_list.extend(loss.sum(axis=-1).mean(axis=-1).detach().cpu().tolist())
        loss = loss.sum(axis=(-1, -2)) / output_mask.sum(axis=(-1, -2))
        loss_list.extend(loss.detach().cpu().tolist())
    return loss_list, old_loss_list

writer("Start")

# %%
# load patient data from preprocessing and embeddings (train and valid)
dir_train = '../../train/0_train_final_seed0_optXtr/'
writer(f'use {dir_train} as train')

fore_train_op = joblib.load(dir_train + "fore_train_op_f32.pkl")
fore_valid_op = joblib.load(dir_train + "fore_valid_op_f32.pkl")
emb_train = joblib.load(dir_train + "emb_train.pkl")
emb_valid = joblib.load(dir_train + "emb_valid.pkl")
writer(f"shapes of fore_train_op and fore_valid_op: {fore_train_op.shape}, {fore_valid_op.shape}")
writer(f"shapes of emb_train and emb_valid: {emb_train.shape}, {emb_valid.shape}")

# %%
# load model parameters and model
d_model, e_layers, d_layers, d_ff = 512, 2, 1, 2048  # hyperparameter training for model
V = 131  # number of different variables in dataset

# 204800 samples --> 6400 batches, 102400 samples --> 3200 batches
batch_size, samples_per_epoch = 32, 204800
patience_break, patience_scheduler = 6, 3 # patience for early stopping and scheduler
d, N, he, dropout = 50, 2, 4, 0.2
lr = 0.00005
number_of_epochs = 200


# all model parameters
method = 'dp-sgd'

red_fore_privacy = 1 # 1: can delete syn data, 0: doesn't delete any syn data.
num_batches = 1000
syn_ratio = 0.5
MSE_inc = 1.005  # 0.5% accepted error
privacy_inc = 1.005 # 0.5% accepted error
ap_to = 3 # accuracy privacy tradeoff (1 accuracy : ap_to privacy)
seed_value = 0

# zoo parameters
alpha = 0.25
MSE_sign = -1
num_pert = 3
num_branches = 10
lam = 3000
mu = 300
kappa = 0

# pca parameter (and all zoo)
pca_ratio = 0.7

# mixup parameters
beta = 1

#DP-SGD
noise_multiplier = 1.1
clip_norm = 1.1

if method == 'only' or num_batches == 0 or syn_ratio == 0:
    method, num_batches, syn_ratio = 'only', 0, 0
if method == 'dp-sgd':
    num_batches, syn_ratio = 0, 0


variables = [
    "method", "red_fore_privacy", "num_batches", "syn_ratio", "MSE_inc", "privacy_inc", 
    "ap_to", "seed_value", "alpha", "MSE_sign", "num_pert", "num_branches", "lam", "mu", "kappa",  
    "pca_ratio", "beta", "num_epochs", "d", "N", "he", "dropout", "lr", "samples_per_epoch", 
    "patience_break", "patience_scheduler", "d_model", "e_layers", "d_layers", "d_ff", "dir_preprocess", "dir_train"
]

output = []
output.append(f"eICU with {V} variables: ")

for name in variables:
    if name in globals():  # Check if the variable exists
        value = globals()[name]
        output.append(f"{name} = {value}")

writer(", ".join(output))

# %%
# initialise new model
parser = get_parser(V, d_model, e_layers, d_layers, d_ff)
args = parser.parse_args([])
importlib.reload(autoformer)
model = autoformer.Model(args).cuda()
if method == 'dp-sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

start_model = dir_train + "normalised.pytorch"
model.load_state_dict(torch.load(start_model))
writer("model " + start_model + " loaded.")


# %%
# get MSE(X_train/val) with init model
loss_listT, old_loss_list = eval_set(model, emb_train, fore_train_op, batch_size, args)
MSE_X_train_theta_init, MSE_X_tr_old = np.mean(loss_listT), np.mean(old_loss_list)
TP, FN = sum(loss_listT < MSE_X_train_theta_init), sum(loss_listT > MSE_X_train_theta_init)
PL_X_train_theta_init = TP / len(loss_listT)

loss_listV, old_loss_list = eval_set(model, emb_valid, fore_valid_op, batch_size, args)
MSE_X_val_theta_init, MSE_X_val_old = np.mean(loss_listV), np.mean(old_loss_list)
MSE_X_val_best = MSE_X_val_theta_init
FP, TN = sum(loss_listV < MSE_X_train_theta_init), sum(loss_listV > MSE_X_train_theta_init)

TPR, FPR = TP / (FN + TP), FP / (FP + TN)
TPRFPR, TPRFPR_best = TPR / FPR, TPR / FPR

labels = np.concatenate([np.ones(len(loss_listT)), np.zeros(len(loss_listV))])
mse_values = np.concatenate([loss_listT, loss_listV])

# Compute TPR, FPR at different thresholds
fpr, tpr, thresholds = roc_curve(labels, -mse_values)  # Negate MSE since lower = member
auroc = auc(fpr, tpr)

best_epoch = 0
writer("TPRFPR first" + str(TPRFPR_best))
writer("on theta_init: MSE(X_train) = " + str(MSE_X_train_theta_init)
    + ", PL(X_train, X_train) = " + str(PL_X_train_theta_init)
    + ", MSE(X_val) = " + str(MSE_X_val_theta_init))
writer("FRL Epoch MSE(X_train) MSE(X_train_new) MSE(X_train_syn) MSE(X_train_last_syn) "
       "PL(X_train) MSE(X_val) TPR FPR len(emb_privacy)")
writer(f"FRL 0 {MSE_X_train_theta_init} {MSE_X_train_theta_init} "
       f"{MSE_X_train_theta_init} {MSE_X_train_theta_init} "
       f"{PL_X_train_theta_init} {MSE_X_val_theta_init} {len(loss_listT)} "
       f"{MSE_X_tr_old} {MSE_X_val_old} {auroc}")
writer2("Epoch MSE(X_train) MSE(X_train_new) MSE(X_train_syn) MSE(X_train_last_syn) PL(X_train) "
        "MSE(X_val) TPR FPR len(emb_privacy) TPR/FPR TP TN FP FN TPi TNi FPi FNi "
        "TPii TNii FPii FNii MSE_X_tr_old MSE_X_val_old auroc result")
writer2(f"0 {MSE_X_train_theta_init} {MSE_X_train_theta_init} {MSE_X_train_theta_init} "
        f"{MSE_X_train_theta_init} {PL_X_train_theta_init} {MSE_X_val_theta_init} {TPR} {FPR} "
        f"{len(emb_train)} {TPR/FPR} {TP} {TN} {FP} {FN} 0 0 0 0 0 0 0 0 "
        # {TPi} {TNi} {FPi} {FNi} {TPii} {TNii} {FPii} {FNii} '
        f"{MSE_X_tr_old} {MSE_X_val_old} {auroc} 0")
# result = 0: no success, 1: only MSE, 2: success
torch.save(model.state_dict(), "model_from_epoch_0.pytorch")

# %%
# Training
emb_privacy = emb_train
fore_privacy_op = fore_train_op
train_len = len(fore_privacy_op)
MSE_X_new_for_PL = MSE_X_train_theta_init

centroid = torch.tensor(np.mean(emb_train, axis=0)).cuda()
if method == 'pca': V_pca = pca_sklearn(emb_train, ratio=pca_ratio)

# Ensure deterministic behaviour in Numpy and PyTorch
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If using GPU

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for e in range(1, number_of_epochs + 1):
    writer("epoch " + str(e) + " started.")

    # a) perturb embeddings with DA method and add new syn data to fore_privacy_op_new/emb_privacy_new
    with torch.no_grad():
        for minibatch in range(num_batches):
            if method == 'zoo' or method == 'pca':
                sampled_indices = np.random.choice(len(emb_privacy), size=32, replace=False)
                emb = torch.tensor(emb_privacy[sampled_indices]).cuda().to(dtype=torch.float32)
                matrix = torch.tensor(fore_privacy_op[sampled_indices], dtype=torch.float32).cuda()
                output_matrix, output_mask = matrix[:, 24:, :V], matrix[:, 24:, V:]
                output, _,_ = model(emb, 0,0,0,0, trainn=False)
                loss_start = output_mask[:, -args.pred_len:, :]*(output-output_matrix[:, -args.pred_len:, :])**2
                PL_sum_ges, MSE_change_ges = 0, 0
                for i in range(num_pert):
                    for k in range(num_branches):
                        if method == 'zoo':
                            rd_emb = np.random.normal(loc=0, scale=1, size=emb.shape) 
                            normed = np.linalg.norm(rd_emb, axis=(1,2))
                            rd_emb = (rd_emb / normed[:, None, None]) * mu
                            emb_plus = emb.detach() + torch.tensor(rd_emb).cuda().to(dtype=torch.float32)
                            emb_minus = emb.detach() - torch.tensor(rd_emb).cuda().to(dtype=torch.float32)
                        if method == 'pca':
                            emb_plus, emb_minus, rd_emb = noise_significant_dims(emb, V_pca, mu)
                        
                        output, _,_ = model(emb_plus, 0,0,0,0, trainn=False)
                        loss = output_mask[:, -args.pred_len:, :]*(output-output_matrix[:, -args.pred_len:, :])**2
                        loss_plus = loss.sum(axis=(-1,-2))/output_mask.sum(axis=(-1,-2))
                        output, _,_ = model(emb_minus, 0,0,0,0, trainn=False)
                        loss = output_mask[:, -args.pred_len:, :]*(output-output_matrix[:, -args.pred_len:, :])**2
                        loss_minus = loss.sum(axis=(-1,-2))/output_mask.sum(axis=(-1,-2))
                        MSE_pm = loss_plus - loss_minus
                        PL_pm = (loss_plus<MSE_X_new_for_PL).int() - (loss_minus<MSE_X_new_for_PL).int()
                        norm_p = torch.linalg.norm(emb_plus - centroid, axis=(1, 2))
                        norm_m = torch.linalg.norm(emb_minus - centroid, axis=(1, 2))
                        centroid_norm = norm_p - norm_m
                        emb_weights = MSE_sign * alpha * MSE_pm - (1-alpha) * PL_pm - kappa * centroid_norm.cuda()
                        if k== 0: change = emb_weights.view(32,1,1) * torch.tensor(rd_emb, dtype=torch.float32).cuda()
                        else:     change += emb_weights.view(32,1,1) * torch.tensor(rd_emb, dtype=torch.float32).cuda()
                    emb -= change * lam/num_branches/2/mu          


            if method == 'mixup':
                samples = []
                while len(samples) < 32:  # Take at least 32 values and make 32 to cuda tensor
                    lam = np.random.beta(beta, beta, 40)  # Sample in batches of 40 for efficiency
                    filtered = lam[lam > 0.5]  # Keep only values > 0.5
                    samples.extend(filtered)  # Add to our list
                mixup = torch.tensor(np.array(samples[:32]), dtype=torch.float32).cuda()
                sampled_ind1, sampled_ind2 = [np.random.choice(len(emb_privacy), size=32, replace=False) for _ in range(2)]
                emb1, emb2 = [torch.tensor(emb_privacy[idx]).cuda().to(dtype=torch.float32) 
                            for idx in (sampled_ind1, sampled_ind2)]
                mixup = mixup.view(-1, *[1] * (emb1.dim() - 1))
                emb = mixup * emb1 + (1 - mixup) * emb2

                matrix1, matrix2 = [torch.tensor(fore_privacy_op[idx], dtype=torch.float32).cuda() 
                                    for idx in (sampled_ind1, sampled_ind2)]
                output_matrix = (mixup * matrix1[:, 24:, :V] + (1 - mixup) * matrix2[:, 24:, :V])
                io_matrix = mixup * matrix1[:, :, :V] + (1 - mixup) * matrix2[:, :, :V]
                io_mask = matrix1[:, :, V:]
                matrix = torch.cat((io_matrix, io_mask), dim=-1)

            if minibatch == 0:
                add2fore_privacy_op_new = matrix.cpu()
                add2emb_privacy_new = emb.cpu()
            else:
                add2fore_privacy_op_new = torch.cat([torch.tensor(add2fore_privacy_op_new), matrix.cpu()])
                add2emb_privacy_new = torch.cat([torch.tensor(add2emb_privacy_new), emb.cpu()])
                
                
        if (len(fore_privacy_op) > 1.5 * train_len) & (red_fore_privacy == 1):
            fore_privacy_op_new = np.concatenate(
                [fore_privacy_op[:train_len], 
                 fore_privacy_op[train_len + num_batches * batch_size :],
                 add2fore_privacy_op_new.numpy()])
            emb_privacy_new = np.concatenate(
                [emb_privacy[:train_len], emb_privacy[train_len + num_batches * batch_size :],
                 add2emb_privacy_new.numpy()])
            writer("fore_privacy_op is reduced.")
        elif method == 'only' or method =='dp-sgd':
            fore_privacy_op_new, emb_privacy_new = fore_privacy_op, emb_privacy
        else:
            fore_privacy_op_new = np.concatenate(
                [fore_privacy_op, add2fore_privacy_op_new.numpy()])
            emb_privacy_new = np.concatenate([emb_privacy, add2emb_privacy_new.numpy()])
    
    # b) Train the model with advanced datasets
    if method == 'only' or method == 'dp-sgd':
        e_indices = np.random.choice(range(train_len), size=int(samples_per_epoch), replace=True)
    else: 
        first_half_indices = np.random.choice(range(train_len), 
                            size=int(samples_per_epoch * (1 - syn_ratio)), replace=True)
        second_half_indices = np.random.choice(range(train_len, len(fore_privacy_op_new)), 
                            size=int(samples_per_epoch * syn_ratio), replace=True)
        e_indices = np.concatenate([first_half_indices, second_half_indices])
    np.random.shuffle(e_indices)

    pbar = tqdm(range(0, len(e_indices), batch_size), file=open(os.devnull, "w"))
    model.train()
    for start in pbar:
        ind = e_indices[start : start + batch_size]
        matrix = torch.tensor(fore_privacy_op_new[ind], dtype=torch.float32).cuda()
        output_matrix = matrix[:, 24:, :V]
        output_mask = matrix[:, 24:, V:]
        emb = torch.tensor(emb_privacy_new[ind]).cuda().to(dtype=torch.float32)

        if method == 'dp-sgd':
            # DP-SGD: The 4-step wonder algorithm
            current_batch_size = len(ind)
            optimizer.zero_grad()
            
            accumulated_grads = {}
            batch_loss = 0
            # total_loss = 0
            # Step 1 & 2: Train each sample and clip its gradient
            for i in range(current_batch_size):
                sample_emb = emb[i:i+1]
                sample_target = output_matrix[i:i+1] 
                sample_mask = output_mask[i:i+1]
                
                # Forward pass
                output, _, _ = model(sample_emb, 0, 0, 0, 0, 
                                        tgt=sample_target, trainn=False, backprop=True)
                
                # Loss
                diff = output - sample_target
                masked_diff = sample_mask * diff
                loss = (masked_diff ** 2).sum() / sample_mask.sum()
                batch_loss += loss.item()
                
                # Backward
                loss.backward()
                
                # Clip gradient for this sample
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                
                # Store clipped gradient
                if i == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            accumulated_grads[name] = param.grad.clone()
                else:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            accumulated_grads[name] += param.grad
                
                optimizer.zero_grad()
            
            # Step 3: Average clipped gradients
            for name in accumulated_grads:
                accumulated_grads[name] /= current_batch_size
            
            # Step 4: Add noise with factor 1/current_batch_size
            noise_scale = noise_multiplier * clip_norm / current_batch_size
            for name, param in model.named_parameters():
                if name in accumulated_grads:
                    noise = torch.normal(0, noise_scale, param.shape, 
                                    device=param.device, dtype=param.dtype)
                    param.grad = accumulated_grads[name] + noise
            
            # Update
            optimizer.step()
            # total_loss += batch_loss / current_batch_size

        else: 
            output, _, _ = model(emb, 0, 0, 0, 0, tgt=output_matrix, trainn=False, backprop=True)
            loss = (output_mask[:, -args.pred_len :, :]
                    * (output - output_matrix[:, -args.pred_len :, :]) ** 2)
            loss = loss.sum(axis=(-1, -2)) / output_mask.sum(axis=(-1, -2))
            loss = loss.mean()
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

    # c) Recalculate MSE(X_train), MSE(X_val), TPRFPR rate for evaluation checks
    loss_listT, old_loss_listT = eval_set(model, emb_privacy_new, fore_privacy_op_new, 
                                        batch_size, args)
    MSE_X_tr_new, MSE_X_tr_init = np.mean(loss_listT), np.mean(loss_listT[:train_len])
    MSE_X_newest = np.mean(loss_listT[-num_batches * batch_size :])
    MSE_X_syn, MSE_X_tr_old= np.mean(loss_listT[train_len:]), np.mean(old_loss_listT[:train_len])
    loss_listT = loss_listT[:train_len]
    PL_X_train_X_tilde_theta_tilde = sum(loss_listT < MSE_X_tr_new) / len(loss_listT)
    TP, FN = sum(loss_listT < MSE_X_tr_new), sum(loss_listT > MSE_X_tr_new)
    TPi, FNi = sum(loss_listT < MSE_X_tr_init), sum(loss_listT > MSE_X_tr_init)
    TPii, FNii = sum(loss_listT < MSE_X_train_theta_init), sum(loss_listT > MSE_X_train_theta_init)

    loss_listV, old_loss_listV = eval_set(model, emb_valid, fore_valid_op, batch_size, args)
    MSE_X_val_theta_tilde = np.mean(loss_listV)
    MSE_X_val_old = np.mean(old_loss_listV[:train_len])
    FP, TN = sum(loss_listV < MSE_X_tr_new), sum(loss_listV > MSE_X_tr_new)
    FPi, TNi = sum(loss_listV < MSE_X_tr_init), sum(loss_listV > MSE_X_tr_init)
    FPii, TNii = sum(loss_listV < MSE_X_train_theta_init), sum(loss_listV > MSE_X_train_theta_init)

    TPR, FPR = TP / (FN + TP), FP / (FP + TN)
    TPRFPR = max(TPR / FPR, 1) # needs a realistic TPRFPR
    writer(f"TPR:FPR ratio is {TPR/FPR} TP {TP} TN {TN} FP {FP} FN {FN}.")

    labels = np.concatenate([np.ones(len(loss_listT)), np.zeros(len(loss_listV))])
    mse_values = np.concatenate([loss_listT, loss_listV])

    # Compute TPR, FPR at different thresholds
    fpr, tpr, thresholds = roc_curve(labels, -mse_values)  # Negate MSE since lower = member
    auroc = auc(fpr, tpr)


    # d) check conditions, save model and update if best model
    cond_MSEval = MSE_X_val_theta_tilde < MSE_X_val_best * MSE_inc
    cond_privacy = TPRFPR < TPRFPR_best * privacy_inc
    cond_tradeoff = MSE_X_val_theta_tilde + ap_to * TPRFPR < MSE_X_val_best + ap_to * TPRFPR_best

    if cond_privacy & cond_MSEval & cond_tradeoff:
        best_epoch = e
        writer(f"SUCCESS: all conditions are fulfilled in epoch {e}")
        fore_privacy_op = fore_privacy_op_new
        emb_privacy = emb_privacy_new
        writer("save model model_from_epoch_" + str(e) + ".pytorch")
        torch.save(model.state_dict(), "finalmodel_from_epoch_" + str(e) + ".pytorch")
        TPRFPR_best = min(TPRFPR_best, TPR / FPR)
        MSE_X_val_best = min(MSE_X_val_best, MSE_X_val_theta_tilde)
        MSE_X_new_for_PL = MSE_X_tr_init
        res = 2
    else:
        try:
            model.load_state_dict(torch.load("finalmodel_from_epoch_" + str(best_epoch) + ".pytorch"))
            writer("reload model model_from_epoch_" + str(best_epoch) + ".pytorch")
        except:
            model.load_state_dict(torch.load(start_model))
            writer("load the first model because there was no improved model yet.")
        if cond_MSEval == 0:
            writer("FAIL: MSE(X_val) not decreased: "
                   f"{MSE_X_val_theta_tilde} < {MSE_X_val_best} * {MSE_inc} = {MSE_X_val_best * MSE_inc}")
            res = 1
        else: res = 0
        if cond_privacy == 0:
            writer(f"FAIL: privacy not increased: {TPRFPR} < {TPRFPR_best} * {privacy_inc} = {TPRFPR_best * privacy_inc}")
        if cond_tradeoff == 0:
            writer(f"FAIL: tradeoff too small: {MSE_X_val_theta_tilde} + {ap_to} * {TPRFPR} "
                   f"< {MSE_X_val_best} + {ap_to} * {TPRFPR_best}, "
                   f"{MSE_X_val_theta_tilde + ap_to * TPRFPR} "
                   f"< {MSE_X_val_best + ap_to * TPRFPR_best}")
   
    writer(f"FRL {e} {MSE_X_tr_init} {MSE_X_tr_new} {MSE_X_syn} {MSE_X_newest} "
           f"{PL_X_train_X_tilde_theta_tilde} {MSE_X_val_theta_tilde} {TPR} {FPR} "
           f"{len(emb_privacy)} {MSE_X_tr_old} {MSE_X_val_old}")
    writer2(f"{e} {MSE_X_tr_init} {MSE_X_tr_new} {MSE_X_syn} {MSE_X_newest} "
           f"{PL_X_train_X_tilde_theta_tilde} {MSE_X_val_theta_tilde} {TPR} {FPR} "
           f"{len(emb_privacy)} {TPR/FPR} {TP} {TN} {FP} {FN} {TPi} {TNi} {FPi} {FNi} "
           f"{TPii} {TNii} {FPii} {FNii} {MSE_X_tr_old} {MSE_X_val_old} {auroc} {res}")

    if e - best_epoch > patience_scheduler:
        scheduler.step()
        writer(f'{patience_scheduler} times no improvements, so scheduler step used.')
    if e - best_epoch > patience_break:
        writer(f"{patience_break} times no improvements, so break.")
        writer(f"Best epoch was {best_epoch}")
        break
    
writer(f"Training finished after {e} epochs and with {len(emb_privacy)} examples.")

joblib.dump(emb_privacy[train_len:], filename="00_emb_syn.pkl")
joblib.dump(fore_privacy_op[train_len:], filename="00_fore_syn.pkl")


if best_epoch > 0:
    final_model = "finalmodel_from_epoch_" + str(best_epoch) + ".pytorch"
else:
    final_model = "model_from_epoch_0.pytorch"
model.load_state_dict(torch.load(final_model))

loss_listT, _ = eval_set(model, emb_privacy[:train_len], fore_privacy_op[:train_len], batch_size, args)
loss_listV, _ = eval_set(model, emb_valid, fore_valid_op, batch_size, args)

labels = np.concatenate([np.ones(len(loss_listT)), np.zeros(len(loss_listV))])
mse_values = np.concatenate([loss_listT, loss_listV])

# Compute TPR, FPR at different thresholds
fpr, tpr, thresholds = roc_curve(labels, -mse_values)  # Negate MSE since lower = member
auroc = auc(fpr, tpr)


# Save to CSV
df_roc = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Thresholds": thresholds})
df_roc.to_csv("roc_curve_data.csv", index=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Membership Inference Attack ROC Curve")
plt.legend()
plt.grid()
plt.savefig("AUROC.png")
plt.loglog()
plt.savefig("AUROC_loglog.png")

writer(f"best_epoch =  {best_epoch}, MSE(X_train) = {np.mean(loss_listT)}, MSE(X_val) = {np.mean(loss_listV)}")
