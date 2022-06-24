import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
from utils import timer
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
# bpr = utils.BPRLoss(Recmodel, world.config)
bpr = utils.CrossEntroyLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    eval_recall10 = 0
    result_topk = {}
    result_ctr = {}
    eval_auc = 0
    best_epoch = []
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if world.task == 'topk':
            if epoch %10 == 0:
                print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}]')
                cprint("[EVAL]")
                eval_result_epoch = Procedure.Test(dataset, Recmodel, epoch, 'eval', w, world.config['multicore'])
                eval_recall10_batch = eval_result_epoch['recall'][2]
                if eval_recall10_batch > eval_recall10:
                    eval_recall10 = eval_recall10_batch
                    result_topk = eval_result_epoch
                    torch.save(Recmodel.state_dict(), weight_file)
                    best_epoch.append(epoch-1)
                cprint("[TEST]")
                Procedure.Test(dataset, Recmodel, epoch, 'test', w, world.config['multicore'])
            # output_information = Procedure.BPR_train_original_fixed(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
            # print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            # torch.save(Recmodel.state_dict(), weight_file)
        elif world.task == 'ctr':
            if epoch %10 == 0:
                cprint("[EVAL]")
                eval_result_epoch = Procedure.CTRTest(dataset, Recmodel, epoch, 'eval', w, world.config['multicore'])
                eval_auc_batch = eval_result_epoch['auc']
                if eval_auc_batch > eval_auc:
                    result_ctr = eval_result_epoch
                    eval_auc = eval_auc_batch
                    torch.save(Recmodel.state_dict(), weight_file)
                    best_epoch.append(epoch - 1)
                cprint("[TEST]")
                Procedure.CTRTest(dataset, Recmodel, epoch, 'test', w, world.config['multicore'])
        output_information = Procedure.BPR_train_original_fixed(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
    print(best_epoch)
    if world.task == 'ctr':
        print(result_ctr)
    else:
        print(result_topk)
finally:
    if world.tensorboard:
        w.close()