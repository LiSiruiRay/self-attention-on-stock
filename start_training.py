# Author: ray
# Date: 1/23/24
# Description: runable start training
import time

import torch
from torch import nn

from datatype.training_dataset import Mydataset
from training_pipeline import StockPredictionModel, train_model, validate_model, create_dataloaders, \
    get_cosine_schedule_with_warmup, valid

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"[Info]: Use {device} now!")

use_reduced_passage_vec = False

num_epochs = 10  # Set the number of epochs

time_features = 3

passage_vec_size = 128 if use_reduced_passage_vec else 768
output_size = 6

d_model = 1024

model = StockPredictionModel(passage_vec_size=passage_vec_size,
                             time_features=time_features,
                             d_model=d_model,
                             output_size=output_size)

model = model.to(device).float()


mds = Mydataset(use_reduced_passage_vec=use_reduced_passage_vec)

train_loader, test_loader = create_dataloaders(dataset=mds)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=70000)
print(f"[Info]: Finish creating model!", flush=True)

train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device, 70000)
valid(model=model, dataloader=test_loader, criterion=criterion, device=device)

description = "changedsteps"
current_timestamp_millis = int(time.time() * 1000)

model_id = f"{passage_vec_size}_{num_epochs}_{d_model}_{description}_{current_timestamp_millis}_reducedPassageVec" \
    if use_reduced_passage_vec \
    else f"{passage_vec_size}_{num_epochs}_{d_model}_{description}_{current_timestamp_millis}_fullPassageVec"

torch.save(model.state_dict(), f'model_{model_id}.pth')
