# Author: ray
# Date: 1/22/24
import os.path
import unittest

from datatype.csv_read_data_type import CSVDSChunk
from datatype.training_dataset import SPDS
from util.common import get_proje_root_path


class MyTestCaseMyDataset(unittest.TestCase):
    def test_MyDataset(self):
        md = SPDS(use_reduced_passage_vec=False)
        print(f"check the first: {md[0]}")
        print(f"length: {md[0][0][0].shape}")

    def test_csv(self):
        proje_root = get_proje_root_path()
        csvds = CSVDSChunk(os.path.join(proje_root, f"data/dataset.csv"))

        print(csvds[0])



if __name__ == '__main__':
    unittest.main()
