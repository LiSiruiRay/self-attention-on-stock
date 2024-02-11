# Author: ray
# Date: 2/11/24
# Description:
import os

import torch
from torch.utils.data import Dataset

import pandas as pd

from util.common import get_proje_root_path


class CSVDSOTR(Dataset):
    """
    One time read = OTR
    """
    len: int
    csv_file_path: str
    df: pd.DataFrame
    use_reduced_passage_vec: bool
    device: torch.device

    def __init__(self,  device, csv_file_path: str = "data/dataset.csv",
                 use_reduced_passage_vec: bool = False, lazy_load: bool = False):
        print(f"reading once dataset initiation started, with dataset: {csv_file_path}")
        self.csv_file_path = csv_file_path
        self.use_reduced_passage_vec = use_reduced_passage_vec
        self.device = device
        if not lazy_load:
            print(f"loading all the data at once...")
            self.load_data()
            print(f"finished loading data")

    def load_data(self):
        self.df = pd.read_csv(os.path.join(get_proje_root_path(), self.csv_file_path))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        curr_row = self.df.iloc[idx]
        text_id = curr_row.iloc[1]
        time_info = curr_row.iloc[2: 5]
        effect_time = torch.tensor([curr_row.iloc[5]], dtype=torch.float32).to(self.device)
        target_vec = curr_row.iloc[6: 12]
        project_root = get_proje_root_path()
        if self.use_reduced_passage_vec:
            text_id = os.path.join(project_root, f"data/text_vector/{text_id}_reduced.pt")
        else:
            text_id = os.path.join(project_root, f"data/text_vector/{text_id}.pt")
        passage_vec = torch.load(text_id)

        passage_vec = passage_vec.squeeze(0)

        time_vec = torch.tensor(time_info, dtype=torch.float32).to(self.device)

        time_vec[0] = time_vec[0] / 1e6
        time_vec[1] = time_vec[1] / 1e4
        time_vec[2] = time_vec[2] / 1e4

        return ((passage_vec.float(),
                time_vec,
                effect_time),
                torch.tensor(target_vec, dtype=torch.float32).to(self.device).float())


class CSVDSChunk(Dataset):
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
