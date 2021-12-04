import torch
from torch.utils.data import DataLoader

from utilities import utils
from datasets.houses_dataset import HousesDataset
from models.models import LinearRegression
from options.linear_regression_options import LinearRegressionOptions

if __name__ == "__main__":
    options = LinearRegressionOptions()
    utils.init_pytorch(options)

    # create and visualize datasets
    train_dataset = HousesDataset(options, train=True)
    test_dataset = HousesDataset(options, train=False)
    train_dataset.plot_data()
    test_dataset.plot_data()

    # create dataloaders for easy access
    train_dataloader = DataLoader(train_dataset, options.batch_size_train)
    test_dataloader = DataLoader(test_dataset, options.batch_size_test)

    """START TODO: fill in the missing parts as mentioned by the comments."""
    # create a LinearRegression instance named model
    model = LinearRegression()
    # define the opimizer
    # (visit https://pytorch.org/docs/stable/optim.html?highlight=torch%20optim#module-torch.optim for more info)
    """ With momentum, our loss lowers more drastically. Combining this with a dampening of 0.1, we see a much faster decrease of loss.
    Old loss: ~252, new loss: ~244
    Adding L2 regularizing does not seem to make any difference.
    
    Increasing the batch size to ~10000 results in a much lower loss of around ~2.8 on the training set and 28.24 on the test set.
    
    Using Adam seems to immediately converge to a loss of ~179.18.
    
    When incraesing the batch_size and dataset in a whole, we see a gentle decrease from 178.8 to 177 and below. Adding L2 penalty is no avail.
    
    Using RMSProp with 0.01 lr .9 momentum, 0.1 L2 penalty and centered, we observe a loss of 13.84 and it's still decreasing steadily.
    After decreasing the mas house size back to 70, I get a train loss of 3.54.
    """

    # optimizer = torch.optim.SGD(model.parameters(), options.lr, momentum=0.9, dampening=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), options.lr, amsgrad=True, weight_decay=0.9)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.1, centered=True) # Test loss of 31.38, higher lr seems to lower the loss quicker
    # optimizer = torch.optim.NAdam(model.parameters(), lr=options.lr, momentum_decay=0.9, weight_decay=0.1)
    # optimizer = torch.optim.Rprop(model.parameters(), lr=options.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.9, patience=10, threshold=0.0001,
                                                           threshold_mode='abs')
    # train the model
    utils.train_lin_model(model, optimizer, train_dataloader, options, scheduler)
    """END TODO"""

    # test the model
    print("Testing the model...\n")

    print("On the train set:")
    utils.test_lin_reg_model(model, train_dataloader)
    utils.test_lin_reg_plot(model, train_dataloader, options)

    print("On the test set:")
    utils.test_lin_reg_model(model, test_dataloader)
    utils.test_lin_reg_plot(model, test_dataloader, options)
    utils.print_lin_reg(model, options)

    # save the model
    utils.save(model, options)
