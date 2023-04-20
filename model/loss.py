import torch


def loss_func(a, b, alpha=0.1):
    r"""
    Loss in paper. loss_traj + `alpha` * loss_node
    loss_traj:
        Paper's description:
            - loss_traj is the negative Gaussian log-likelihood for the groundtruth future trajectories
        The commonly NLLLoss is for classification problem, but this problem doesn't belong to it.
        Using the literal comprehension, -log(Gaussion(a-b)) is equals MSE, so we use MSE loss.
    loss_node:
        Relative to node completion, now we set it to zero.
    Args:
        a: [batch_size, len, dim]
        b:
        alpha: blend factor
    Returns:
        A value.
    """
    loss_traj = torch.nn.MSELoss()
    loss_node = 0
    return loss_traj(a, b) + alpha * loss_node
