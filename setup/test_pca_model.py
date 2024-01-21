# Author: ray
# Date: 1/21/24
import pickle
import unittest
import torch.nn.functional as F

from vectorization.vectorization import get_passage_vector
import torch
import numpy as np


class MyTestCasePCAModel(unittest.TestCase):
    def test_PCA_model(self):
        pca_loaded = None
        with open('pca_reducer_128.pkl', 'rb') as file:
            pca_loaded = pickle.load(file)
        text = "Tech giants AMZN, META, MSFT, and NVDA saw significant gains in November, while Meta rebounded strongly in 2023 and struck an exclusive deal with Tencent for low-cost VR headsets in China."
        rewrite_text = "Technology powerhouses Amazon, Meta Platforms, Microsoft, and Nvidia experienced substantial growth in November. Concurrently, Meta made a robust recovery in 2023 and secured an exclusive agreement with Tencent to supply affordable virtual reality headsets in China."
        not_related_text = "Officers responded to the shooting on the first Thursday of the year within minutes and discovered several people at the high school suffering gunshot wounds. They then located the gunman with a self-inflicted gunshot wound, CNN previously reported."
        v_text = get_passage_vector(text=text)
        v_r = get_passage_vector(text=rewrite_text)
        v_n = get_passage_vector(text=not_related_text)

        cosine_sim_long_same = F.cosine_similarity(v_text, v_r)
        cosine_sim_long_diff = F.cosine_similarity(v_text, v_n)

        reduced_v_text = pca_loaded.transform(v_text)
        reduced_v_text = torch.from_numpy(reduced_v_text)

        reduced_v_r = pca_loaded.transform(v_r)
        reduced_v_r = torch.from_numpy(reduced_v_r)

        reduced_v_n = pca_loaded.transform(v_n)
        reduced_v_n = torch.from_numpy(reduced_v_n)

        cosine_sim_short_same = F.cosine_similarity(reduced_v_text, reduced_v_r)
        cosine_sim_short_diff = F.cosine_similarity(reduced_v_text, reduced_v_n)

        print(f"shape of short: {reduced_v_text.shape}")
        print(f"cosine_sim_long_same: {cosine_sim_long_same}, cosine_sim_long_diff: {cosine_sim_long_diff}")
        print(f"cosine_sim_short_same: {cosine_sim_short_same}, cosine_sim_short_diff: {cosine_sim_short_diff}")


if __name__ == '__main__':
    unittest.main()
