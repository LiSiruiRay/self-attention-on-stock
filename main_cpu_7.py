# This is a sample Python script.
from start_training import start_training_process
import torch

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cpu")
    start_training_process(num_training_steps=70000,
                           batch_size=100,
                           nhead=2,
                           transformer_encoder_layer_num=50,
                           num_epochs=2,
                           device=device)

    print(f"-------=-============= finished first one, started the second one... ============-----------")
    start_training_process(num_training_steps=10000,
                           batch_size=100,
                           nhead=2,
                           transformer_encoder_layer_num=70,
                           num_epochs=2,
                           device=device)

    print(f"-------=-============= finished first one, started the second one... ============-----------")
    start_training_process(num_training_steps=70000,
                           batch_size=100,
                           nhead=2,
                           transformer_encoder_layer_num=90,
                           num_epochs=2,
                           device=device)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
