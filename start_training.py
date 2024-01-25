# Author: ray
# Date: 1/23/24
# Description: runable start training
import time

import torch
from torch import nn

from datatype.training_dataset import Mydataset
from training_pipeline import StockPredictionModel, train_model, validate_model, create_dataloaders, \
    get_cosine_schedule_with_warmup, valid, save_model, set_seed
from util.common import get_now_time_with_time_zone


def start_training_process(seed: int = 78,
                           use_reduced_passage_vec: bool = False,
                           num_epochs: int = 10,
                           time_features: int = 3,
                           d_model: int = 1024,
                           num_training_steps: int = 10000,
                           batch_size: int = 500,
                           nhead: int = 8,
                           transformer_encoder_layer_num: int = 20,
                           device = torch.device("mps")
                           ):
    set_seed(seed=seed)

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"[Info]: Use {device} now!")

    passage_vec_size = 128 if use_reduced_passage_vec else 768
    output_size = 6

    model = StockPredictionModel(passage_vec_size=passage_vec_size,
                                 time_features=time_features,
                                 d_model=d_model,
                                 output_size=output_size,
                                 nhead=nhead,
                                 transformer_encoder_layer_num=transformer_encoder_layer_num)

    model = model.to(device).float()

    mds = Mydataset(use_reduced_passage_vec=use_reduced_passage_vec)

    train_loader, test_loader = create_dataloaders(dataset=mds,
                                                   batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps)
    print(f"[Info]: Finish creating model!", flush=True)

    training_start_time = time.time()
    train_model(model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=num_epochs,
                device=device,
                num_training_steps=num_training_steps, )
    valid(model=model, dataloader=test_loader, criterion=criterion, device=device)
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    training_duration = training_duration / 60

    description = "local run"

    now_time_str = get_now_time_with_time_zone()

    training_machine = "M1 Pro 16G"

    model_info = {
        "passage vector size": passage_vec_size,
        "num of epochs": num_epochs,
        "d_model": d_model,
        "description": description,
        "time started training": now_time_str,
        "training step num": num_training_steps,
        "batch size": batch_size,
        "seed": seed,
        "nhead": nhead,
        "transformer encoder layer num": transformer_encoder_layer_num,
        "training time (min)": training_duration,
        "training machine": training_machine,
    }

    save_model(model=model,
               info=model_info)
