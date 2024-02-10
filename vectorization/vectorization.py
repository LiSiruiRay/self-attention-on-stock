# Author: ray
# Date: 1/20/24
# Description:

from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# Initialize the tokenizer and model for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_passage_vector(text: str):
    # Tokenize the sentence and convert to tensor
    inputs = tokenizer(text, return_tensors="pt")
    # print(f"inputs: {inputs}---------")

    # Extract hidden states
    with torch.no_grad():
        outputs = model(**inputs)

    # The last_hidden_states are the word vectors for each token
    # last_hidden_states = outputs.last_hidden_state
    # print(f"last_hidden_states: {last_hidden_states}")

    last_hidden_states = outputs.last_hidden_state
    average_embedding = torch.mean(last_hidden_states, dim=1)
    return average_embedding

