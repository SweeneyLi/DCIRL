"""
@File  :optimizer_factory.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 9:38 AM
@Desc  :build the optimizer
"""

from torch import optim


def get_optim_and_scheduler(model, lr, optimizer_config):
    params = filter(lambda x: x.requires_grad, model.parameters())

    if optimizer_config["optimizer_type"] == 'sgd':
        optimizer = optim.SGD(params,
                              weight_decay=optimizer_config["weight_decay"],
                              momentum=optimizer_config["momentum"],
                              nesterov=optimizer_config["nesterov"],
                              lr=lr)
    elif optimizer_config["optimizer_type"] == 'adam':
        optimizer = optim.Adam(params,
                               weight_decay=optimizer_config["weight_decay"],
                               lr=optimizer_config["lr"])
    else:
        raise ValueError("Optimizer not implemented")

    if optimizer_config["scheduler_type"] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=optimizer_config["lr_decay_step"],
                                              gamma=optimizer_config["lr_decay_rate"])
    elif optimizer_config["scheduler_type"] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=optimizer_config["lr_decay_step"],
                                                   gamma=optimizer_config["lr_decay_rate"])
    elif optimizer_config["scheduler_type"] == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=optimizer_config["lr_decay_rate"])
    else:
        raise ValueError("Scheduler not implemented")

    return optimizer, scheduler
