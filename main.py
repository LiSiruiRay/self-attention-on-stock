# This is a sample Python script.
from start_training import start_training_process
if __name__ == '__main__':
    start_training_process(num_training_steps=10000,
                           batch_size=500,
                           nhead=4,
                           transformer_encoder_layer_num=15,
                           num_epochs=10,
                           check_point_steps=1000)
    
    start_training_process(num_training_steps=10000,
                           batch_size=500,
                           nhead=8,
                           transformer_encoder_layer_num=10,
                           num_epochs=15,
                           check_point_steps=2000)

