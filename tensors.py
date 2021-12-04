import torch

from options.options import Options
from utilities.utils import plot_tensor, mse, init_pytorch, not_implemented


def create_image(options: Options) -> torch.Tensor:
    """
    use options to put the tensor to the correct device.
    """
    data = [[[0.5021, 0.1138, 0.9047], [0.2843, 0.0684, 0.6829], [0.1935, 0.5483, 0.3117]],
            [[0.8017, 0.8733, 0.6258], [0.5914, 0.6004, 0.2893], [0.7038, 0.5983, 0.9914]]]
    tensor = torch.FloatTensor(data, device=options.device)
    tensor = torch.permute(tensor, (2, 0, 1))

    return tensor


def lin_layer_forward(weights: torch.Tensor, random_image: torch.Tensor) -> torch.Tensor:
    linear = torch.dot(random_image, weights)

    return linear


def tensor_network():
    target = torch.FloatTensor([0.5], device=options.device)
    print(f"The target is: {target.item():.2f}")
    plot_tensor(target, "Target")

    input_tensor = torch.FloatTensor([0.4, 0.8, 0.5, 0.3], device=options.device)
    weights = torch.FloatTensor([0.1, -0.5, 0.9, -1], device=options.device)
    """START TODO:  ensure that the tensor 'weights' saves the computational graph and the gradients after backprop"""
    weights.requires_grad = True
    """END TODO"""

    # remember the activation a of a unit is calculated as follows:
    #      T
    # a = W * x, with W the weights and x the inputs of that unit
    output = lin_layer_forward(weights, input_tensor)
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Initial Output")

    # We want a measure of how close we are according to our target
    loss = mse(output, target)
    print(f"The initial loss is: {loss.item():.2f}\n")

    # Lets update the weights now using our loss..
    print(f"The current weights are: {weights}")

    """START TODO: the loss needs to be backpropagated"""
    weights.grad = torch.zeros_like(weights)
    loss.backward(retain_graph=False)
    """END TODO"""

    print(f"The gradients are: {weights.grad}")
    """START TODO: implement the update step with a learning rate of 0.5"""
    # use tensor operations, recall the following formula we've seen during class: x <- x - alpha * x
    a = 0.01  # Learning rate
    weights = weights - a * weights.grad

    """
            At first I thought it was necessary to manually create a loop in which the weights would be updated dependant on the distance (loss) from our target. I started off with a while loop that
            check if the loss is smaller than our target loss. If it isn't, we keep updating our weights.

            After some advise from the docents, I replaced the while loop with a for loop that works the same way as epochs do.

            With each iteration, we check our distance to our target and update the weights accordingly.

            Apparently, this whole function wasn't necessary as we only had to change the learning rate to reach our target in just one line of code. But hey, we learned something.

            After some hyperparameters tuning I can conclude that a learning rate combined with 250 epochs the optimal result yields.
    """

    for i in range(250):
        print(f'Epoch {i}/100')
        weights_data = list(weights.data)
        print(weights_data)
        del weights, output, loss  # we want to rebuild the whole graph to have a fresh start and calculate the grads correctly (not give [0,0,0,0] as weights.grad)

        weights = torch.FloatTensor(weights_data, device=options.device)
        weights.requires_grad = True

        output = lin_layer_forward(weights, input_tensor)
        loss = mse(output, target)

        weights.grad = torch.zeros_like(weights)
        loss.backward(retain_graph=False)  # by default 1
        # grads = weights.grad

        weights = weights - a * weights.grad

        print(f"Intermediate loss: {loss: .5f}\n")
        if a > 0.11:
            a -= 0.002

    print(f"Current loss: {loss: .2f}\n")

    """END TODO"""
    print(f"The new weights are: {weights}\n")

    # What happens if we forward through our layer again?
    output = lin_layer_forward(weights, input_tensor)
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Improved Output")


if __name__ == "__main__":
    options = Options()
    init_pytorch(options)
    tensor_network()
