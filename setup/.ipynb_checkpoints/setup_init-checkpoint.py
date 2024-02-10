# Author: ray
# Date: 1/20/24
# Description:

import json
import pickle

import torch

from util.common import text_to_md5_hash, get_dim_reducer
from PassageVectoringStrategies.bert_embedding_average_strategy import BERTEmbeddingAverageStrategy

# Replace 'your_file.json' with your JSON file's name
with open('../data/news_data.json', 'r') as file:
    data = json.load(file)

embedding_list = []
# print(f"len: {len(data)}")
vectorizer = BERTEmbeddingAverageStrategy()

for index, (i, each) in enumerate(data.items()):
    each_text = each['data']
    vectorized_text = vectorizer.vectorize(text=each_text)
    embedding_list.append(vectorized_text)
    print(f"finished text_generating : {index}/{len(data)}")

all_embeddings = torch.cat(embedding_list, dim=0)
reducer = get_dim_reducer(to_size=128, bert_embeddings=all_embeddings)

for index, (i, each) in enumerate(data.items()):
    each_text = each['data']
    each_id = text_to_md5_hash(each_text)
    vector = embedding_list[index]
    reduced_vector = torch.from_numpy(reducer.transform(vector))
    torch.save(reduced_vector, f'../data/{each_id}_reduced.pt')
    print(f"finished storing : {index}/{len(data)}")

with open('pca_reducer_128.pkl', 'wb') as file:
    pickle.dump(reducer, file)
