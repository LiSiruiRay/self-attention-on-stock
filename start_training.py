# Author: ray
# Date: 1/23/24
# Description: runable start training
import time

import torch
from torch import nn

from datatype.csv_read_data_type import CSVDSOTR, CSVDSChunk
from datatype.training_dataset import SPDS
from training_pipeline import StockPredictionModel, train_model, validate_model, create_dataloaders, \
    get_cosine_schedule_with_warmup, valid, save_model, set_seed, visualization
from util.common import get_now_time_with_time_zone, get_hash_id_dict

data_dict = {
    'CSV one time only': CSVDSOTR,
    'CSV chunk read': CSVDSChunk,
    'self proces ds': SPDS,
}


def start_training_process(seed: int = 78,
                           use_reduced_passage_vec: bool = False,
                           num_epochs: int = 10,
                           time_features: int = 3,
                           d_model: int = 1024,
                           num_training_steps: int = 10000,
                           batch_size: int = 500,
                           nhead: int = 8,
                           transformer_encoder_layer_num: int = 20,
                           device=torch.device("cuda" if torch.cuda.is_available()
                                               else "mps" if torch.backends.mps.is_available()
                                               else "cpu"),
                           check_point_steps=-1,
                           training_machine: str = "M1 Pro 16G",
                           description: str = "local run",
                           dataset: str = 'CSV one time only',
                           dataset_path: str = "data/dataset.csv"
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

    TrainingDS = data_dict[dataset]

    if TrainingDS == CSVDSOTR:
        ds = TrainingDS(csv_file_path=dataset_path,
                        use_reduced_passage_vec=use_reduced_passage_vec,
                        device = device)
    else:
        ds = TrainingDS(use_reduced_passage_vec=use_reduced_passage_vec)

    train_loader, test_loader = create_dataloaders(dataset=ds,
                                                   batch_size=batch_size)

    print(f"[Info]: Finish creating the dataset")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps)
    print(f"[Info]: Finish creating model!", flush=True)

    training_start_time = time.time()
    losses = train_model(model=model,
                         train_loader=train_loader,
                         criterion=criterion,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         num_epochs=num_epochs,
                         device=device,
                         num_training_steps=num_training_steps,
                         check_point_steps=check_point_steps)
    valid(model=model, dataloader=test_loader, criterion=criterion, device=device)
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    training_duration = training_duration / 60

    now_time_str = get_now_time_with_time_zone()

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
    model_id = get_hash_id_dict(data_dict=model_info)
    # model_id = "test"
    # print(f"check losses: {losses}")
    for i, l in enumerate(losses):
        visualization(losses=l, model_id=f"{model_id}", epoch_index=i)
