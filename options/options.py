class Options:
    def __init__(self):
        # runtime related
        self.random_seed = 1
        self.device = "cuda"  # Run linear regression with 'cpu', classification can be done with 'cuda'

        # model related
        self.save_path = "./models/"
        self.load_path = "./models/"
        self.model_name = "lin_reg.pth"
