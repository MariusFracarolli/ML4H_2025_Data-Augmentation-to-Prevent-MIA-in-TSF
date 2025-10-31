
import os
from datetime import datetime
import importlib
import copy
import joblib  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau  # type: ignore
from utils.config import get_parser
import models.InformerAutoregressive_emb as autoformer
# from utils.pca import pca_sklearn, noise_significant_dims
from tqdm import tqdm  # type: ignore

import pandas as pd
tqdm.pandas()


import re 

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

def eval_set(model, emb_ev, fore, batch_size, args):
  with torch.no_grad():
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

# load patient data from preprocessing and embeddings (train and valid)
pkl_folder = "../02_train/"
writer(f'use {pkl_folder} as data folder')
fore_train_op = joblib.load(open(pkl_folder + 'fore_train_op.pkl', 'rb'))
fore_valid_op = joblib.load(open(pkl_folder + 'fore_valid_op.pkl', 'rb'))
fore_test_op = joblib.load(open(pkl_folder + 'fore_test_op.pkl', 'rb'))
writer("shapes of fore_train_op and fore_valid_op: " + str(fore_train_op.shape) + ", " 
        + str(fore_valid_op.shape)+ str(fore_test_op.shape))

# # get embs from the mother folder
emb_folder = pkl_folder
embT = joblib.load(open(os.path.join(emb_folder, 'emb_train.pkl'), 'rb'))
embV = joblib.load(open(os.path.join(emb_folder, 'emb_valid.pkl'), 'rb'))
embTest = joblib.load(open(os.path.join(emb_folder, 'emb_test.pkl'), 'rb'))
writer("shapes of embeddings train, valid, test: " + str(embT.shape) + ", " 
        + str(embV.shape)+ str(embTest.shape))

# load model parameters and model
d_model, e_layers, d_layers, d_ff = 256, 2, 2, 2048  # hyperparameter training for model
V = 100  # number of different variables in dataset

batch_size, samples_per_epoch, patience = 32, 102400, 6 # 6400 epochs - 26.2.2025
d, N, he, dropout, e, best_epoch = 50, 2, 4, 0.2, 0, 0
lr = 0.0005
number_of_epochs = 200
seed_value = 0
# model_name = 'normalised.pytorch'

# %%
# initialise new model
parser = get_parser(V, d_model, e_layers, d_layers, d_ff)
args = parser.parse_args([])
importlib.reload(autoformer)
model = autoformer.Model(args).cuda()

dirs = ['../03_privacy']


def get_best_model(dir):
    # Initialize variables to track the highest epoch
    max_epoch, best_model = -1, None
    pattern = re.compile(r"finalmodel_from_epoch_(\d+)\.pytorch")

    # Iterate through files in the folder
    for file in os.listdir(dir):
        match = pattern.match(file)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                best_model = file
    return best_model, max_epoch


def save_lossLists(dir):
    print(dir)
    dir = f"../03_privacy/{dir}"
    get_best_model(dir)
    pytorch_train = f"{dir}/{get_best_model(dir)[0]}"
    print(pytorch_train)
    if get_best_model(dir)[0] is None:
        print(f"No model found in {dir}, skipping...")
    else:
        model.load_state_dict(torch.load(pytorch_train))
        loss_listTest, _ = eval_set(model, embTest, fore_test_op, batch_size, args)
        loss_listT, _ = eval_set(model, embT, fore_train_op, batch_size, args)
        loss_listV, _ = eval_set(model, embV, fore_valid_op, batch_size, args)


        pd.Series(loss_listT).to_csv(f"{dir}/loss_listT.csv", index=False)
        pd.Series(loss_listV).to_csv(f"{dir}/loss_listV.csv", index=False)
        pd.Series(loss_listTest).to_csv(f"{dir}/loss_listTest.csv", index=False)
        del loss_listT, loss_listV, loss_listTest


for dir in next(os.walk('../03_privacy/'))[1]:
    print(dir)
    save_lossLists(dir)
