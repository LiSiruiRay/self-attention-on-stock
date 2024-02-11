# Author: ray
# Date: 2/11/24
# Description:
import os

import torch
from torch.utils.data import Dataset

import pandas as pd

from util.common import get_proje_root_path


class CSVDS(Dataset):
    len: int
    csv_file_path: str
    chunk_size: int

    def __init__(self, csv_file_path: str, chunk_size=10000):
        self.csv_file_path = csv_file_path
        self.chunk_size = chunk_size
        # Initialize an iterator over the CSV file
        self.chunk_iterator = pd.read_csv(self.csv_file_path, chunksize=self.chunk_size, iterator=True)
        # Load the first chunk to get the number of columns
        try:
            self.first_chunk = next(self.chunk_iterator)
        except StopIteration:
            raise ValueError("CSV file is empty")

    def loading_data(self):
        # Reset the iterator after fetching the first chunk
        self.chunk_iterator = pd.read_csv(self.csv_file_path, chunksize=self.chunk_size, iterator=True)
        # Compute the total length
        self.len = sum(1 for row in pd.read_csv(self.csv_file_path, chunksize=1))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len or idx < 0:
            raise IndexError("Index out of range")

        chunk_idx = idx // self.chunk_size
        row_idx = idx % self.chunk_size

        # Load the chunk containing the desired row
        for i, chunk in enumerate(self.chunk_iterator):
            if i == chunk_idx:
                curr_row = chunk.iloc[row_idx]
                text_id = curr_row[1]
                time_info = curr_row[2: 5]
                effect_time = torch.tensor(curr_row[5])
                target_vec = curr_row[6: 11]

                project_root = get_proje_root_path()
                text_id = os.path.join(project_root, f"data/text_vector/{text_id}.pt")
                passage_vec = torch.load(text_id).float()

                time_vec = torch.tensor(time_info)

                return (passage_vec.float(), time_vec, effect_time), target_vec.float()

        # This point should never be reached due to the bounds check at the start of this method
        raise IndexError("Index out of range")

# Usage
# csv_file_path = 'path/to/your/large_file.csv'
# dataset = LargeCSVDataset(csv_file_path)
#
# # Get the length of the dataset
# print("Dataset length:", len(dataset))
#
# # Get a specific item (row)
# item = dataset[100]  # Get the 101st row (0-indexed)
# print(item)
