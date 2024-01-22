# Author: ray
# Date: 1/20/24
# Description:

import hashlib
import os

import torch
from sklearn.decomposition import PCA
import json


def text_to_md5_hash(text: str):
    # Create an MD5 hash object
    hash_object = hashlib.md5()

    # Update the hash object with the bytes of the text
    hash_object.update(text.encode())

    # Get the hexadecimal representation of the hash
    hash_code = hash_object.hexdigest()

    return hash_code


def get_dim_reducer(to_size: int, bert_embeddings):
    pca = PCA(n_components=to_size)
    pca.fit(bert_embeddings)
    return pca


def get_proje_root_path() -> str:
    """
        This function gets absolute path for the root of the project.
        Whichever program called this function under wherever, it will always return the correct absolute path
    """

    # If not running in the root folder, return the absolute folder.
    # get the directory that the current script is in
    current_script_directory = os.path.dirname(os.path.realpath(__file__))

    # get the path of the resource directory relative to the current script
    proje_root_path = os.path.join(current_script_directory, '../')
    return proje_root_path

# Example usage
text = "Your text here"
hash_code = text_to_md5_hash(text)
print("Hash Code:", hash_code)
