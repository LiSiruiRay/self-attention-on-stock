# Author: ray
# Date: 1/22/24
# Description:
import json
import math
import os

import numpy as np
import torch
import random

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from util.common import get_proje_root_path, get_hash_id_dict


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Check if MPS is available for Apple's M1 chips
    if torch.backends.mps.is_available():
        # MPS-specific seed settings can be added here if available
        pass

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


# set_seed(87)


def create_dataloaders(dataset, batch_size=32, shuffle_train=True):
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size

    # Split dataset
    train_dataset = Subset(dataset, range(train_size))
    test_dataset = Subset(dataset, range(train_size, total_size))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    """
	Create a schedule with a learning rate that decreases following the values of the cosine function between the
	initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
	initial lr set in the optimizer.

	Args:
		optimizer (:class:`~torch.optim.Optimizer`):
		The optimizer for which to schedule the learning rate.
		num_warmup_steps (:obj:`int`):
		The number of steps for the warmup phase.
		num_training_steps (:obj:`int`):
		The total number of training steps.
		num_cycles (:obj:`float`, `optional`, defaults to 0.5):
		The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
		following a half-cosine).
		last_epoch (:obj:`int`, `optional`, defaults to -1):
		The index of the last epoch when resuming training.

	Return:
		:obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
	"""

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

    mels, labels = batch
    # print(f"checking mels: {mels}")
    input_data = [i.to(device) for i in mels]
    labels = labels.to(device)

    outs = model(input_data[0], input_data[1], input_data[2])

    # print(f"checking outs : {outs}, shape: {outs.shape}")
    # print(f"checking label: {labels}, shape: {labels.shape}")

    loss = criterion(outs, labels)

    # Get the speaker id with highest probability.
    # preds = outs.argmax(1)
    # # Compute accuracy.
    # accuracy = torch.mean((preds == labels).float())

    return loss


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device, num_training_steps):
    # pbar2 = tqdm(total=num_epochs, ncols=0, desc="Train", unit="epoch")
    model.to(device)
    train_iterator = iter(train_loader)

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        pbar = tqdm(total=num_training_steps, ncols=0, desc="Train", unit=" step")

        for step in range(num_training_steps):
            # Get data
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            loss = model_fn(batch, model, criterion, device)
            batch_loss = loss.item()
            # batch_accuracy = accuracy.item()

            # Updata model
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Log
            pbar.update()
            pbar.set_postfix(
                loss=f"{batch_loss:.2f}",
                # accuracy=f"{batch_accuracy:.2f}",
                step=step + 1,
            )
        pbar.close()
        # pbar.close()
        #
        # model.train()
        # total_loss = 0
        # for inputs, targets in train_loader:
        #     inputs = [input.float().to(device) for input in inputs]  # Convert inputs to float32
        #     targets = targets.float().to(device)
        #
        #     # Forward pass
        #     outputs = model(*inputs)
        #     loss = criterion(outputs, targets)
        #
        #     # Backward and optimize
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     scheduler.step()
        #
        #     total_loss += loss.item()
        # pbar.update()
        # pbar.set_postfix(
        #     loss=f"{total_loss:.2f}",
        #     # accuracy=f"{batch_accuracy:.2f}",
        #     step=epoch + 1,
        # )

        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

    print("Training complete")
    # pbar2.close()


def validate_model(model, valid_loader, criterion, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        total_loss = 0
        for inputs, targets in valid_loader:
            inputs = [input.to(device) for input in inputs]
            targets = targets.to(device)

            # Forward pass
            outputs = model(*inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(valid_loader)
    print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss


def valid(dataloader, model, criterion, device):
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    # running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            # running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i + 1):.2f}",
            # accuracy=f"{running_accuracy / (i + 1):.2f}",
        )

    pbar.close()
    model.train()

    # return running_accuracy / len(dataloader)


class StockPredictionModel(nn.Module):
    def __init__(self, passage_vec_size, time_features, d_model, output_size):
        super(StockPredictionModel, self).__init__()
        # Define layers
        self.linear1 = nn.Linear(passage_vec_size + time_features + 1, d_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        self.linear2 = nn.Linear(d_model, output_size)

    def forward(self, passage_vec, time_vec, time_of_effect):
        # Ensure all inputs are 2D tensors before concatenation
        # passage_vec is already 2D, [1, N]
        # Add an extra dimension to time_vec to make it [1, 3]
        # time_vec = time_vec.unsqueeze(0) if time_vec.dim() == 1 else time_vec
        # Turn time_of_effect into a 2D tensor [1, 1]
        # time_of_effect = time_of_effect.view(1, -1)
        # print(f"checking the datas: passage_vec: {passage_vec}, time_vec: {time_vec}, time_of_effect: {time_of_effect}")
        # Combine input features
        combined_input = torch.cat((passage_vec, time_vec, time_of_effect), dim=1)
        # Pass through layers
        x = torch.relu(self.linear1(combined_input))
        x = x.unsqueeze(0)  # Add a batch dimension
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Remove the batch dimension
        output = self.linear2(x)
        return output


def save_model(model: nn.Module, info: dict):
    project_root = get_proje_root_path()
    model_folder_path = os.path.join(project_root, "model")
    meta_data_path = os.path.join(model_folder_path, "meta_data/")
    model_id = get_hash_id_dict(data_dict=info)
    info["model_id"] = model_id

    model_path = os.path.join(model_folder_path, f"{model_id}.pth")

    meta_data_file_path = os.path.join(meta_data_path, f"{model_id}.json")

    with open(file=meta_data_file_path) as file:
        json.dump(info, file, indent=4)

    torch.save(model.state_dict(), model_path)
