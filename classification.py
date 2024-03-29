import torch

from datasets.mnist_dataset import MNISTDataset
from models.models import Classifier, ClassifierVariableLayers
from options.classification_options import ClassificationOptions
from utilities import utils
from utilities.utils import init_pytorch, test_classification_model, train_classification_model, classify_images

if __name__ == "__main__":
    options = ClassificationOptions()
    init_pytorch(options)

    # create and visualize the MNIST dataset
    dataset = MNISTDataset(options)
    dataset.show_examples()

    """START TODO: fill in the missing parts"""
    # create a Classifier instance named model
    model = Classifier(options)
    model.to(options.device)
    # define the opimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.9, patience=10, threshold=0.0001,
                                                           threshold_mode='abs')

    # optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    # train the model
    train_classification_model(model, optimizer, dataset, options, scheduler)
    """END TODO"""

    # Test the model
    print("The Accuracy of the model is: ")
    test_classification_model(model, dataset, options)
    classify_images(model, dataset, options)

    # save the model
    utils.save(model, options)
