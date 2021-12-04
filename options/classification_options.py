from options.options import Options


class ClassificationOptions(Options):
    def __init__(self):
        super().__init__()
        # dataset related
        self.batch_size_test = 32*2
        self.batch_size_train = 128*2

        # hyperparameters
        self.lr = 0.01
        self.num_epochs = 50
        self.hidden_sizes = {
            'layer1_input_shape': 3 * 13 * 26,  # 28*28
            'layer1_output_shape': 128 * 2,
            'layer2_input_shape': 128 * 2,
            'layer2_output_shape': 64 * 2,
            'layer3_input_shape': 64 * 2,
            'layer3_output_shape': 10
        }


"""
Accuracy: 96.62%
epoch [10/10]: Running loss = 1.4880483250882326
Accuracy: 96.68%
The Accuracy of the model is: 
Accuracy: 96.68%
"""
