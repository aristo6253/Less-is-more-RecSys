import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset, dataset_good, dataset_bad
import model

# Recmodel = model.LightGCN(world.config, dataset).to(world.device)
# bpr = utils.BPRLoss(Recmodel, world.config)

if world.config['dual']:
    Recmodel_good = model.LightGCN(world.config, dataset_good).to(world.device)
    bpr_good = utils.BPRLoss(Recmodel_good, world.config)

    Recmodel_bad = model.LightGCN(world.config, dataset_bad).to(world.device)
    bpr_bad = utils.BPRLoss(Recmodel_bad, world.config)
    
    weight_file_good, weight_file_bad = utils.getFileName()
    print(f"Load and save to {weight_file_good} and {weight_file_bad}")
    if world.LOAD:
        try:
            Recmodel_good.load_state_dict(torch.load(weight_file_good, map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file_good}")
        except FileNotFoundError:
            print(f"{weight_file_good} not exists, start from beginning")
        try:
            Recmodel_bad.load_state_dict(torch.load(weight_file_bad, map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file_bad}")
        except FileNotFoundError:
            print(f"{weight_file_bad} not exists, start from beginning")
else:
    Recmodel = model.LightGCN(world.config, dataset).to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)
    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")

Neg_k = 1


if world.config['infer']:
    if world.config['dual']:
        Procedure.Infer_Dual(dataset, Recmodel)
    else:
        Procedure.Infer(dataset, Recmodel)
else:

    if world.config['dual']:
        pass
    else:
    # init tensorboard
        if world.tensorboard:
            w : SummaryWriter = SummaryWriter(
                                            join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                            )
        else:
            w = None
            world.cprint("not enable tensorflowboard")

        try:
            for epoch in range(world.TRAIN_epochs):
                start = time.time()
                #if epoch %10 == 0:
                cprint("[TEST]")
                Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
                print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
                torch.save(Recmodel.state_dict(), weight_file)
        finally:
            if world.tensorboard:
                w.close()






