# %%
import os
from datetime import datetime
import importlib
import copy
import joblib  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau  # type: ignore
from utils.config import get_parser
import models.InformerAutoregressive as autoformer
from tqdm import tqdm  # type: ignore
tqdm.pandas()

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
pkl_folder = "../preprocess1/"
writer(f'use {pkl_folder} as data folder')
fore_train_op = joblib.load(open(pkl_folder + 'eicu_fore_train_op.pkl', 'rb'))
fore_valid_op = joblib.load(open(pkl_folder + 'eicu_fore_valid_op.pkl', 'rb'))
writer("shapes of fore_train_op and fore_valid_op: " + str(fore_train_op.shape) + ", " 
        + str(fore_valid_op.shape))
# 16 sec, mit f64 7 min

# %%
# load model parameters and model
d_model, e_layers, d_layers, d_ff = 256, 2, 2, 2048  # hyperparameter training for model
V = 100  # number of different variables in dataset

batch_size, samples_per_epoch, patience = 32, 102400, 6 # 6400 epochs - 26.2.2025
d, N, he, dropout, e, best_epoch = 50, 2, 4, 0.2, 0, 0
lr = 0.0005
number_of_epochs = 200
seed_value = 42
model_name = 'normalised.pytorch'

# %%
# initialise new model
parser = get_parser(V, d_model, e_layers, d_layers, d_ff)
args = parser.parse_args()
importlib.reload(autoformer)
model = autoformer.Model(args).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, "min", patience=3)
# Pretrain fore_model.
MSE_X_tr_best = np.inf
N_train = len(fore_train_op)
N_valid = len(fore_valid_op)


writer("FRL Epoche MSE(X_train) "
       "PL(X_train) MSE(X_val) TPR FPR len(emb_privacy)")
writer2("Epoche MSE(X_train) PL(X_train) "
        "MSE(X_val) TPR FPR len(emb_privacy) TP TN FP FN "
        "MSE_X_tr_old MSE_X_val_old result")
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
    e_loss = 0
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
        e_loss += loss.detach()
    writer("Epoche " + str(e) + " loss " + str(e_loss / len(e_indices)))

    # c) Recalculate MSE(X_train), MSE(X_val), TPRFPR rate for evaluation checks
    loss_list, old_loss_list, _ = eval_set(model, fore_train_op, batch_size, args)
    MSE_X_tr, MSE_X_tr_old = np.mean(loss_list), np.mean(old_loss_list)
    TP, FN = sum(loss_list < MSE_X_tr), sum(loss_list > MSE_X_tr)

    loss_list, old_loss_list, _ = eval_set(model, fore_valid_op, batch_size, args)
    MSE_X_val, MSE_X_val_old = np.mean(loss_list), np.mean(old_loss_list)
    FP, TN = sum(loss_list < MSE_X_tr), sum(loss_list > MSE_X_tr)

    TPR, FPR = TP / (FN + TP), FP / (FP + TN)
    writer(f"TPR:FPR ratio is {TPR/FPR} TP {TP} TN {TN} FP {FP} FN {FN}.")

    # d) check conditions, save model and update if best model
    if MSE_X_tr < MSE_X_tr_best:
        best_epoch = e
        writer(f"SUCCESS: all conditions are fulfilled in epoch {e}")
        torch.save(model.state_dict(), "finalmodel_from_epoch_" + str(e) + ".pytorch")
        torch.save(model.state_dict(), model_name)
        MSE_X_tr_best = MSE_X_tr
        res = 2
    else:
        model, res = old_model, 0
        if e - best_epoch > 3:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr / 10)
            writer(f'optimizer updated to {lr}.')
        if e - best_epoch > patience:
            writer(f"{patience} times no improvements, so break.")
            writer(f"Best epoch was {best_epoch}")
            break


    writer(f"FRL {e} {MSE_X_tr} {MSE_X_val} {TPR} {FPR} "
           f"{N_train} {MSE_X_tr_old} {MSE_X_val_old}")
    writer2(f"{e} {MSE_X_tr} {MSE_X_val} {TPR} {FPR} "
           f"{N_train} {TPR/FPR} {TP} {TN} {FP} {FN} {MSE_X_tr_old} {MSE_X_val_old} {res}")
    
writer(f"Training finished after {e} epochs with best epoch {best_epoch}.")
torch.cuda.empty_cache()

# %%
# Evaluation of the best model
model.load_state_dict(torch.load(model_name))

loss_list, old_loss_list, emb = eval_set(model, fore_train_op, batch_size, args)
joblib.dump(torch.cat([e.cpu().detach() for e in emb], dim=0).numpy(), 'emb_train.pkl')
joblib.dump(fore_train_op, 'fore_train_op.pkl')
joblib.dump(fore_train_op.astype(np.float32), 'fore_train_op_f32.pkl')
del emb, fore_train_op 

print('Train evaluation done: ', datetime.now())


loss_list, old_loss_list, emb = eval_set(model, fore_valid_op, batch_size, args)
joblib.dump(torch.cat([e.cpu().detach() for e in emb], dim=0).numpy(), 'emb_valid.pkl')
joblib.dump(fore_valid_op, 'fore_valid_op.pkl')
joblib.dump(fore_valid_op.astype(np.float32), 'fore_valid_op_f32.pkl')

writer(f'final{best_epoch} {MSE_X_tr} {MSE_X_val} {TP} {FN} {FP} {TN} {MSE_X_tr_old} {MSE_X_val_old}')
