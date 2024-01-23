# Author: ray
# Date: 1/22/24

import unittest

from datatype.training_dataset import Mydataset


class MyTestCaseMyDataset(unittest.TestCase):
    def test_MyDataset(self):
        md = Mydataset()
        print(f"check the first: {md[0]}")


if __name__ == '__main__':
    unittest.main()
