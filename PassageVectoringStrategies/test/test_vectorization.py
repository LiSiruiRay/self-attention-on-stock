# Author: ray
# Date: 1/20/24

import unittest

from PassageVectoringStrategies.bert_embedding_average_strategy import BERTEmbeddingAverageStrategy


class MyTestCase(unittest.TestCase):
    def test_BERTEmbeddingAverageStrategy(self):
        text = ("Sheryl Sandberg, former Meta Platforms COO, announces she will not seek reelection to the company's "
                "board after 12 years, opting to serve as an adviser instead. Sandberg stepped down as COO in 2022, "
                "intending to stay on the board. Meta CEO Mark Zuckerberg expresses gratitude.")
        vectorizer = BERTEmbeddingAverageStrategy()
        result = vectorizer.vectorize(text=text)
        print(f"vector result: {result}")
        print(f"check datatype: {type(result)}")



if __name__ == '__main__':
    unittest.main()
