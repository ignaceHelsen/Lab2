from options.options import Options


class LinearRegressionOptions(Options):
    def __init__(self):
        super().__init__()
        # dataset related
        self.batch_size_train = 10000
        self.batch_size_test = 2000
        self.train_dataset_size = 1000000
        self.test_dataset_size = 200000
        self.min_house_size = 30
        self.max_house_size = 700
        self.noise_house_data = 50000

        # hyperparameters
        self.lr = 0.05
        self.num_epochs = 1


"""
Inreasing the max house size seems to have a positive effect on the training, using different optimizers shows that each optimizer does the learning in a different way
In the end, we reach a steady loss of ~5 that does not seem to lower any more.

A first run (max_house_size of 700) with RmsProp gives us a test loss of 31.38, when lowering the max_house_size to 70 again, we observe a lower test loss of 26.36.
Nadam does not show great results, test loss is 142.14. Perhaps the lr is too low? (test loss of 27.06).
"""
