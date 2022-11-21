import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import tune
from ray.air import session

from .convnet import ConvNet
from .utils import parse_config
# from .ray_tune import AimCallback
from .aim import AimCallback


from ray.tune.logger import tensorboardx

# from aim.hugging_face import AimCallback


# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 4096
TEST_SIZE = 4096

# get the available device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# we don't want tensorboard logging
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"


def train(model, optimizer, train_dataloader, epoch=None):
    """
    Args:
        model:
        optimizer:
        train_dataloader:

    Returns:

    """
    model.train()
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx * len(batch) > EPOCH_SIZE:
            break
        data, target = batch
        optimizer.zero_grad()
        outputs = model(data)
        loss = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    acc = correct / total
    session.report(metrics={"acc": acc, "context": {"subset": "train"}, "epoch":epoch})


def validate():
    pass


def test(model, data_loader, epoch=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # We set this just for the example to run quickly.
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = correct / total
    session.report(metrics={"acc": acc, "context": {"subset": "test"}, "epoch": epoch})


def trainable(config: dict, train_dataloader, validation_dataloader):
    """
    Args:
        config:
        model:
        train_dataloader:
        validation_dataloader:
        test_dataloader:
    """

    model = ConvNet()
    model.to(DEVICE)

    optimizer = optim.SGD(
        model.parameters(), lr=config["max_lr"], momentum=config["momentum"])

    for epoch in range(config["num_epochs"]):
        train(model, optimizer, train_dataloader, epoch)
        test(model, validation_dataloader, epoch)


def run_ray_experiment(config, search_space):
    config = parse_config(config)

    # Uncomment this to enable distributed execution
    # `ray.init(address="auto")`

    # Download the dataset first
    datasets.MNIST("./data", train=True, download=True)

    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataloader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=mnist_transforms),
        batch_size=32,
        shuffle=True
    )

    validation_dataloader = DataLoader(
        datasets.MNIST("./data", train=False, transform=mnist_transforms),
        batch_size=32,
        shuffle=False
    )

    tune_config = None
    if config["tune_algo"] is not None:
        try:
            searcher = config["searcher"]()
            algo = config["algo"](searcher, max_concurrent=4)
        except KeyError:
            raise ModuleNotFoundError("seems like the algo or searcher in the ""config file is not a valid tune "
                                      "module or not imported")

        tune_config = tune.TuneConfig(
            metric=config["metric"],
            mode="min",
            search_algo=algo,
            num_smaples=config["num_hyperparam_samples"],
        )

    reporter = CLIReporter(
        parameter_columns=None,
        metric_columns=config["metric"]
    )

    tune_run_config = air.RunConfig(progress_reporter=reporter,
                                    local_dir=config["output_dir"],
                                    name=config["experiment_name"],
                                    checkpoint_config=air.CheckpointConfig(
                                        num_to_keep=3,
                                        checkpoint_score_order="max",
                                    ),
                                    callbacks=[AimCallback(repo=config["output_dir"],
                                                           experiment=config["experiment_name"],
                                                           metrics=config["metric"])]
                                    )

    tuner = tune.Tuner(
        tune.with_parameters(trainable, train_dataloader=train_dataloader,
                             validation_dataloader=validation_dataloader),
        param_space=search_space,
        run_config=tune_run_config,
        tune_config=tune_config,
    )
    results = tuner.fit()
