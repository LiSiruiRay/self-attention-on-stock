# Author: ray
# Date: 1/22/24

import unittest

from datatype.training_dataset import Mydataset


class MyTestCaseMyDataset(unittest.TestCase):
    def test_MyDataset(self):
        md = Mydataset(use_reduced_passage_vec=False)
        print(f"check the first: {md[0]}")
        print(f"length: {md[0][0][0].shape}")


if __name__ == '__main__':
    unittest.main()
