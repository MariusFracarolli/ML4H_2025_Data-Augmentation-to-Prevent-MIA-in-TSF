# %%
import os
from datetime import datetime
import importlib
import copy
import pandas as pd 
import joblib  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from torch.optim.lr_scheduler import StepLR  # type: ignore
from utils.config import get_parser
import models.InformerAutoregressive as autoformer
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm  # type: ignore
tqdm.pandas()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "heuristic"
torch.cuda.memory._record_memory_history(True)

current_date = datetime.now().strftime("%y%m%d-%H%M")
print(f"{current_date}")

def writer(text):
    current_date = datetime.now().strftime("%y%m%d-%H%M")
    with open("00_results.txt", "a") as file:
        file.write(f"{current_date}: {text}\n")
        print(text)


def writer2(text):
    with open("01_resultlist.txt", "a") as file:
        file.write(f"{text}\n")


def eval_set(model, fore, batch_size, args):
  with torch.no_grad():
    loss_list, old_loss_list, all_emb = [], [], []
    model.eval()
    pbar2 = tqdm(range(0, len(fore), batch_size), file=open(os.devnull, "w"))
    for start in pbar2:
        matrix = torch.tensor(fore[start : start + batch_size], dtype=torch.float32).cuda()
        input_matrix = matrix[:, :24, :V*2]
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        input_mask = matrix[:, :24, V:] # not used
        output_matrix = matrix[:, 24:, :V]
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        output_mask = matrix[:, 24:, V:]
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        if 'Linear' not in str(type(model)) and "Autoregressive" in str(type(model)):
            output, emb, _ = model(input_matrix, input_mark, dec_inp, output_mark, tgt=output_matrix, trainn=False, backprop=True) #check this
        elif "Linear" not in str(type(model)):
            output, emb, _ = model(input_matrix, input_mark, dec_inp, output_mark)
        else:
            output, emb, _ = model(input_matrix)[:, :, :V]
        all_emb.append(emb)
        loss = (output_mask[:, -args.pred_len :, :] 
                * (output - output_matrix[:, -args.pred_len :, :]) ** 2)
        old_loss_list.extend(loss.sum(axis=-1).mean(axis=-1).detach().cpu().tolist())
        loss = loss.sum(axis=(-1, -2)) / output_mask.sum(axis=(-1, -2))
        loss_list.extend(loss.detach().cpu().tolist())
    return loss_list, old_loss_list, all_emb


writer("Start")

# %%
# load patient data from preprocessing and embeddings (train and valid)
pkl_folder = "../../preprocess1/"
writer(f'use {pkl_folder} as data folder')
fore_train_op = joblib.load(open(pkl_folder + 'fore_train_op.pkl', 'rb'))
fore_valid_op = joblib.load(open(pkl_folder + 'fore_valid_op.pkl', 'rb'))
writer(f"shapes of fore_train_op and fore_valid_op: {fore_train_op.shape}, {fore_valid_op.shape}")

# load model parameters and model
d_model, e_layers, d_layers, d_ff = 512, 2, 1, 2048  # hyperparameter training for model
V = 131  # number of different variables in dataset

batch_size, samples_per_epoch = 32, 102400
patience_scheduler, patience_break = 3, 6
d, N, he, dropout = 50, 2, 4, 0.2
lr = 0.0005
number_of_epochs = 200
seed_value = 0
model_name = 'normalised.pytorch'

# initialise new model
parser = get_parser(V, d_model, e_layers, d_layers, d_ff)
args = parser.parse_args()
importlib.reload(autoformer)
model = autoformer.Model(args).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
# Pretrain fore_model.
MSE_X_val_best = np.inf
N_train = len(fore_train_op)
N_valid = len(fore_valid_op)

writer("FRL Epoch MSE(X_train) "
       "MSE(X_val) TPR FPR TPR/FPR len(emb_privacy)")
writer2("Epoch MSE(X_train) MSE(X_val) "
        "TPR FPR TPR/FPR train_samples TP FN FP TN "
        "MSE_X_tr_old MSE_X_val_old auroc result")
# result = 0: no success, 1: only MSE, 2: success

# Ensure deterministic behaviour in Numpy and PyTorch
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If using GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for e in range(1, number_of_epochs + 1):
    old_model = copy.deepcopy(model)
    writer("epoch " + str(e) + " started.")

    # b) Train the model with advanced datasets
    e_indices = np.random.choice(range(N_train), size=samples_per_epoch, replace=True)
    pbar = tqdm(range(0, samples_per_epoch, batch_size), file=open(os.devnull, "w"))
    model.train()
    for start in pbar:
        ind = e_indices[start : start + batch_size]
        matrix = torch.tensor(fore_train_op[ind], dtype=torch.float32).cuda()
        input_matrix = matrix[:, :24, :V*2]
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        input_mask = matrix[:, :24, V:] # not used
        output_matrix = matrix[:, 24:, :V]
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        output_mask = matrix[:, 24:, V:]
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        if 'Linear' not in str(type(model)) and "Autoregressive" in str(type(model)):
            output, _, _ = model(input_matrix, input_mark, dec_inp, output_mark, tgt=output_matrix, trainn=False, backprop=True)
        elif "Linear" not in str(type(model)):
            output, _, _ = model(input_matrix, input_mark, dec_inp, output_mark)
        else:
            output, _, _ = model(input_matrix)[:, :, :V]
        loss = output_mask[:, -args.pred_len:, :]*(output - output_matrix[:, -args.pred_len:, :])**2
        loss = (output_mask[:, -args.pred_len :, :]
                * (output - output_matrix[:, -args.pred_len :, :]) ** 2)
        loss = loss.sum(axis=(-1, -2)) / output_mask.sum(axis=(-1, -2))
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # c) Recalculate MSE(X_train), MSE(X_val), TPRFPR rate for evaluation checks
    loss_listT, old_loss_list, _ = eval_set(model, fore_train_op, batch_size, args)
    MSE_X_tr, MSE_X_tr_old = np.mean(loss_listT), np.mean(old_loss_list)
    TP, FN = sum(loss_listT < MSE_X_tr), sum(loss_listT > MSE_X_tr)

    loss_listV, old_loss_list, _ = eval_set(model, fore_valid_op, batch_size, args)
    MSE_X_val, MSE_X_val_old = np.mean(loss_listV), np.mean(old_loss_list)
    FP, TN = sum(loss_listV < MSE_X_tr), sum(loss_listV > MSE_X_tr)

    TPR, FPR = TP / (FN + TP), FP / (FP + TN)
    writer(f"TPR:FPR ratio is {TPR/FPR} TP {TP} TN {TN} FP {FP} FN {FN}.")

    labels = np.concatenate([np.ones(len(loss_listT)), np.zeros(len(loss_listV))])
    mse_values = np.concatenate([loss_listT, loss_listV])

    # Compute TPR, FPR at different thresholds
    fpr, tpr, thresholds = roc_curve(labels, -mse_values)  # Negate MSE since lower = member
    auroc = auc(fpr, tpr)

    # d) check conditions, save model and update if best model
    if MSE_X_val < MSE_X_val_best:
        best_epoch = e
        writer(f"SUCCESS: all conditions are fulfilled in epoch {e}")
        torch.save(model.state_dict(), "finalmodel_from_epoch_" + str(e) + ".pytorch")
        torch.save(model.state_dict(), model_name)
        MSE_X_val_best = MSE_X_val
        res = 2
    else:
        model, res = old_model, 0

    writer(f"FRL {e} {MSE_X_tr} {MSE_X_val} {TPR} {FPR} {TPR/FPR} {N_train}")
    writer2(f"{e} {MSE_X_tr} {MSE_X_val} {TPR} {FPR} {TPR/FPR} "
           f"{N_train} {TP} {FN} {FP} {TN} {MSE_X_tr_old} {MSE_X_val_old} {auroc} {res}")
    
    if e - best_epoch > patience_scheduler:
        scheduler.step()
        writer(f'{patience_scheduler} times no improvements, so scheduler step used.')
    if e - best_epoch > patience_break:
        writer(f"{patience_break} times no improvements, so break.")
        writer(f"Best epoch was {best_epoch}")
        break
    
writer(f"Training finished after {e} epochs with best epoch {best_epoch}.")
torch.cuda.empty_cache()

# %%
# Evaluation of the best model
model.load_state_dict(torch.load(model_name))

loss_listT, old_loss_list, emb = eval_set(model, fore_train_op, batch_size, args)
joblib.dump(torch.cat([e.cpu().detach() for e in emb], dim=0).numpy(), 'emb_train.pkl')
joblib.dump(fore_train_op, 'fore_train_op.pkl')
joblib.dump(fore_train_op.astype(np.float32), 'fore_train_op_f32.pkl')
del emb, fore_train_op 

print('Train evaluation done: ', datetime.now())


loss_listV, old_loss_list, emb = eval_set(model, fore_valid_op, batch_size, args)
joblib.dump(torch.cat([e.cpu().detach() for e in emb], dim=0).numpy(), 'emb_valid.pkl')
joblib.dump(fore_valid_op, 'fore_valid_op.pkl')
joblib.dump(fore_valid_op.astype(np.float32), 'fore_valid_op_f32.pkl')

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

writer(f"FRL final{best_epoch} {MSE_X_tr} {MSE_X_val} {TPR} {FPR} {TPR/FPR} {N_train}")
writer2(f"final{best_epoch} {MSE_X_tr} {MSE_X_val} {TPR} {FPR} {TPR/FPR} "
        f"{N_train} {TP} {FN} {FP} {TN} {MSE_X_tr_old} {MSE_X_val_old} {auroc} {res}")