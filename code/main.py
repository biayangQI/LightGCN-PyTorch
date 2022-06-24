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
    with timer(name="Sample"):  # sample only once for training
        S = utils.UniformSample_original(dataset)

    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if world.task == 'topk':
            if epoch %10 == 0:
                print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}]')
                cprint("[EVAL]")
                Procedure.Test(dataset, Recmodel, epoch, 'eval', w, world.config['multicore'])
                cprint("[TEST]")
                Procedure.Test(dataset, Recmodel, epoch, 'test', w, world.config['multicore'])
            output_information = Procedure.BPR_train_original(S, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
            # print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            # torch.save(Recmodel.state_dict(), weight_file)
        elif world.task == 'ctr':
            if epoch %10 == 0:
                cprint("[EVAL]")
                Procedure.CTRTest(dataset, Recmodel, epoch, 'eval', w, world.config['multicore'])
                cprint("[TEST]")
                Procedure.CTRTest(dataset, Recmodel, epoch, 'test', w, world.config['multicore'])
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
finally:
    if world.tensorboard:
        w.close()