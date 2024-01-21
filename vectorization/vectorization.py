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
    last_hidden_states = outputs.last_hidden_state

    # print(f"last_hidden_states: {last_hidden_states}")

    last_hidden_states = outputs.last_hidden_state
    average_embedding = torch.mean(last_hidden_states, dim=1)
    return average_embedding


# Sample sentence
# sentence = "Hello, my name is ChatGPT."
# text = "Tech giants AMZN, META, MSFT, and NVDA saw significant gains in November, while Meta rebounded strongly in 2023 and struck an exclusive deal with Tencent for low-cost VR headsets in China."
#
# rewrite_text = "Technology powerhouses Amazon, Meta Platforms, Microsoft, and Nvidia experienced substantial growth in November. Concurrently, Meta made a robust recovery in 2023 and secured an exclusive agreement with Tencent to supply affordable virtual reality headsets in China."  # similarity: tensor([0.9559])
#
# rewrite_text_2 = "Major technology firms Amazon, Meta, Microsoft, and Nvidia witnessed considerable increases in November. In the same vein, Meta achieved a significant turnaround in 2023 and entered into an exclusive partnership with Tencent to provide low-priced VR headsets in the Chinese market."
#
# not_related_text = "The Iowa caucus is set to take place on Monday, marking the official start of the primary season for the 2024 presidential election. As the first major contest for the Republican primary process, it is often considered a bellwether test for candidates with a shot at winning the nomination, though success in the Hawkeye State has not always indicated the eventual party nominee."
#
# not_related_text_2 = "Officers responded to the shooting on the first Thursday of the year within minutes and discovered several people at the high school suffering gunshot wounds. They then located the gunman with a self-inflicted gunshot wound, CNN previously reported."
#
# average_embedding_1 = get_passage_vector(text)
# average_embedding_rewrite_2 = get_passage_vector(rewrite_text)
# average_embedding_rewrite_3 = get_passage_vector(rewrite_text_2)
# average_embedding_3 = get_passage_vector(not_related_text)
# average_embedding_not_related_4 = get_passage_vector(not_related_text_2)
#
# cosine_sim = F.cosine_similarity(average_embedding_1, average_embedding_rewrite_2, dim=1)
# print(f"similarity, rewrite: {cosine_sim}")
#
# cosine_sim = F.cosine_similarity(average_embedding_1, average_embedding_rewrite_3, dim=1)
# print(f"similarity, rewrite_2: {cosine_sim}")
#
# cosine_sim_2 = F.cosine_similarity(average_embedding_1, average_embedding_3, dim=1)
# print(f"similarity, not related: {cosine_sim_2}")
#
# cosine_sim_2 = F.cosine_similarity(average_embedding_1, average_embedding_not_related_4, dim=1)
# print(f"similarity, not related 4: {cosine_sim_2}")
#
# print(f"passage vector: {average_embedding_1}")
