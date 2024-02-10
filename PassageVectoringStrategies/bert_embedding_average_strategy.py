# Author: ray
# Date: 1/20/24
# Description:
from typing import Any

from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

from Interfaces.SteUpInterfaces.PreProcessSN.PassageVectorizingInterfaces.passage_vectoring_strategy import \
    PassageVectorizer

# Initialize the tokenizer and model for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


class BERTEmbeddingAverageStrategy(PassageVectorizer):
    def vectorize(self, text) -> torch.Tensor:
        """
            Average of word vectors stands for passage vector.
            Good for short passages.

            Always the same dimension

            Args:
                text: to vectorize

            Returns: vectorized text

            """
        # Tokenize the sentence and convert to tensor
        inputs = tokenizer(text, return_tensors="pt")

        # Extract hidden states
        with torch.no_grad():
            outputs = model(**inputs)

        # The last_hidden_states are the word vectors for each token
        # last_hidden_states = outputs.last_hidden_state
        # print(f"last_hidden_states: {last_hidden_states}")

        last_hidden_states = outputs.last_hidden_state
        average_embedding = torch.mean(last_hidden_states, dim=1)
        return average_embedding
