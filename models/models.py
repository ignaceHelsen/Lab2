import torch
import torch.nn as nn

from options.classification_options import ClassificationOptions


class Print(nn.Module):
    """"
    This model is for debugging purposes (place it in nn.Sequential to see tensor dimensions).
    """

    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        """START TODO: replace None with a Linear layer"""
        self.linear_layer = torch.nn.Linear(in_features=1, out_features=1)
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward the tensor x through the linear layer and return the outcome (replace None)"""
        x = self.linear_layer(x)
        """END TODO"""
        return x


class Classifier(nn.Module):
    def __init__(self, options: ClassificationOptions):
        super().__init__()
        """ START TODO: fill in all three layers. 
            Remember that each layer should contain 2 parts, a linear layer and a nonlinear activation function.
            Use options.hidden_sizes to store all hidden sizes, (for simplicity, you might want to 
            include the input and output as well).
        """
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), stride=(2, 1)),
            nn.Flatten(),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=options.hidden_sizes['layer1_input_shape'],
                      out_features=options.hidden_sizes['layer1_output_shape']),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=options.hidden_sizes['layer2_input_shape'],
                      out_features=options.hidden_sizes['layer2_output_shape']),
            nn.ReLU()
        )

        self.dropout = nn.Sequential(
            nn.Dropout(0.2)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=options.hidden_sizes['layer3_input_shape'],
                      out_features=options.hidden_sizes['layer3_output_shape']),
            nn.Softmax(dim=1)
        )

        self.layers = nn.Sequential(self.conv2d, self.layer1, self.layer2, self.dropout, self.layer3)
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward tensor x through all layers."""
        x = self.conv2d.forward(x)
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.dropout.forward(x)
        x = self.layer3.forward(x)
        # x = self.layers.forward(x)
        """END TODO"""
        return x


class ClassifierVariableLayers(nn.Module):
    def __init__(self, options: ClassificationOptions):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(options.hidden_sizes) - 1):
            self.layers.add_module(
                f"lin_layer_{i + 1}",
                nn.Linear(options.hidden_sizes[i], options.hidden_sizes[i + 1])
            )
            if i < len(options.hidden_sizes) - 2:
                self.layers.add_module(
                    f"relu_layer_{i + 1}",
                    nn.ReLU()
                )
            else:
                self.layers.add_module(
                    f"softmax_layer",
                    nn.Softmax(dim=1)
                )
        print(self)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return x
