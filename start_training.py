# Author: ray
# Date: 1/23/24
# Description: runable start training
import time

import torch
from torch import nn

from datatype.training_dataset import Mydataset
from training_pipeline import StockPredictionModel, train_model, validate_model, create_dataloaders, \
    get_cosine_schedule_with_warmup, valid, save_model
from util.common import get_now_time_with_time_zone

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"[Info]: Use {device} now!")

use_reduced_passage_vec = True

num_epochs = 500  # Set the number of epochs

time_features = 3

passage_vec_size = 128 if use_reduced_passage_vec else 768
output_size = 6

d_model = 1024

num_training_steps = 100

batch_size = 50

model = StockPredictionModel(passage_vec_size=passage_vec_size,
                             time_features=time_features,
                             d_model=d_model,
                             output_size=output_size)

model = model.to(device).float()

mds = Mydataset(use_reduced_passage_vec=use_reduced_passage_vec)

train_loader, test_loader = create_dataloaders(dataset=mds,
                                               batch_size=batch_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
print(f"[Info]: Finish creating model!", flush=True)

train_model(model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            num_training_steps=num_training_steps,)
valid(model=model, dataloader=test_loader, criterion=criterion, device=device)

description = "local run"

now_time_str = get_now_time_with_time_zone()

model_info = {
    "passage vector size": passage_vec_size,
    "num of epochs": num_epochs,
    "d_model": d_model,
    "description": description,
    "time started training": now_time_str,
    "training step num": num_training_steps,
    "batch size": batch_size,
}

save_model(model=model,
           info=model_info)

# model_id = f"{passage_vec_size}_{num_epochs}_{d_model}_{description}_{current_timestamp_millis}_reducedPassageVec" \
#     if use_reduced_passage_vec \
#     else f"{passage_vec_size}_{num_epochs}_{d_model}_{description}_{current_timestamp_millis}_fullPassageVec"
#
# torch.save(model.state_dict(), f'model_{model_id}.pth')
