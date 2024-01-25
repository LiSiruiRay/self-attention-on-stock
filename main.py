# This is a sample Python script.
from start_training import start_training_process

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_training_process(num_training_steps=10000,
                           batch_size=500,
                           nhead=8,
                           transformer_encoder_layer_num=20,
                           num_epochs=10)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
