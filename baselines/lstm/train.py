import json
import argparse
import logging
import os
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from ogrit.core.base import get_lstm_dir
from baselines.lstm.dataset_base import GRITDataset
from baselines.lstm.model import LSTMModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# todo: code adapted from ...
def save_checkpoint(path, epoch, model, optimizer, losses, accs):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'accs': accs
    }, path)


def run_evaluation(model, loss_fn, data_loader, device, use_encoding=False):
    val_data = [_ for _ in data_loader][0]
    logger.info(f"Running validation")
    trajectories = val_data[0].to(device)
    target = val_data[1].to(device)
    lengths = val_data[2]
    input = pack_padded_sequence(trajectories, lengths, batch_first=True, enforce_sorted=False)

    output, (encoding, lengths) = model(input, use_encoding=use_encoding)

    val_loss = 0.0
    if use_encoding:
        for h_t in encoding.transpose(0, 1):
            val_loss += loss_fn(h_t, target)
    val_loss += loss_fn(output, target)
    val_loss /= encoding.shape[1] + 1

    accuracy = sum(output.argmax(axis=1) == target) / target.shape[0]
    if not use_encoding:
        return val_loss, accuracy, None
    else:
        t = encoding.shape[1]
        encoding_losses = nn.CrossEntropyLoss(reduction="none")(
            encoding.transpose(2, 1), target.repeat(t, 1).T)
        return val_loss, accuracy, encoding_losses


def load_save_dataset(config, split_type="train"):
    dataset_path = get_lstm_dir() + f"/datasets/{config.scenario}_{config.dataset}_{split_type}.pt"
    if not os.path.exists(dataset_path):
        from baselines.lstm.dataset import DATASET_MAP
        dataset_cls = DATASET_MAP[config.dataset]
        dataset = dataset_cls(config.scenario, split_type)
        torch.save({"dataset": dataset.dataset,
                    "labels": dataset.labels,
                    "lengths": dataset.lengths},
                   dataset_path)
    else:
        dataset_dict = torch.load(dataset_path)
        dataset = GRITDataset(config.scenario, split_type)
        dataset.dataset = dataset_dict["dataset"]
        dataset.labels = dataset_dict["labels"]
        dataset.lengths = dataset_dict["lengths"]
    return dataset


def train_epoch(model, loss_fn, data_loader, device, optim, epoch, use_encoding=False):
    running_loss = 0.0

    for i_batch, sample_batched in enumerate(data_loader):
        trajectories = sample_batched[0].to(device)
        target = sample_batched[1].to(device)
        lengths = sample_batched[2]
        input = pack_padded_sequence(trajectories, lengths, batch_first=True, enforce_sorted=False)

        optim.zero_grad()
        output, (encoding, lengths) = model(input, use_encoding=use_encoding)
        loss = 0.0
        if use_encoding:
            # todo: for each time step t with t_max = len(longest trajectory), compute the loss for the predicted
            #  goal of the trajectories in the batch
            for h_t in encoding.transpose(0, 1):
                loss += loss_fn(h_t, target)
        # todo: originally was loss += loss_fn(output, target) + 1
        loss += loss_fn(output, target)

        if use_encoding:
            loss /= encoding.shape[1] + 1  # todo: added +1
        loss.backward()
        optim.step()

        running_loss += loss.item()
        logger.info(f"Epoch: {epoch}; Step: {len(data_loader) * epoch + i_batch}; Loss: {loss.item()}")
    return running_loss / len(data_loader)


def train(config):
    torch.random.manual_seed(42)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('baselines/lstm/runs/train_lstm_{}'.format(timestamp)) # todo: add folder automatically

    if hasattr(config, "config"):
        config = argparse.Namespace(**json.load(open(config.config)))
    logger.info(config)

    # Process and save/load the datasets
    dataset = load_save_dataset(config, "train")
    data_loader = DataLoader(dataset, shuffle=config.shuffle, batch_size=min(config.batch_size, len(dataset)))
    logger.info(f"Dataset loaded: {str(dataset)}")

    val_dataset = load_save_dataset(config, "valid")
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=len(val_dataset))

    # Create model and send to device
    model = LSTMModel(dataset.dataset.shape[-1],
                      config.lstm_hidden_dim,
                      config.fc_hidden_dim,
                      dataset.labels.unique().shape[-1],
                      num_layers=config.lstm_layers,
                      dropout=config.dropout)
    logger.info(f"Model created: {str(model)}")
    if torch.cuda.is_available():
        # todo: install pytorch with cuda: conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    logger.info(f"Training on {device} (CUDA: {torch.cuda.device_count()}).")
    model.to(device)

    # Create loss function -- negative log likelihood loss which requires log probabilities as input
    loss_fn = nn.NLLLoss()
    loss_fn.to(device)

    # Create optimizer and learning rate scheduler
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5)

    losses = []
    accs = []

    logger.info("Starting gradient descent:")
    for epoch in range(config.max_epoch):
        model.train()

        train_loss = train_epoch(model, loss_fn, data_loader, device, optim, epoch, use_encoding=config.use_encoding)

        torch.cuda.empty_cache()
        model.eval()

        val_loss, accuracy, _ = run_evaluation(model, loss_fn, val_loader, device, use_encoding=config.use_encoding)
        schedule.step(val_loss)
        logger.info(f"Validation Loss: {val_loss.item()}; Accuracy {accuracy.item()} "
                    f"LR: {optim.param_groups[0]['lr']}")

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': train_loss, 'Validation': val_loss},
                           epoch + 1)
        writer.flush()

        if config.save_latest:
            save_checkpoint(get_lstm_dir() + config.save_path + ("_" if config.save_path[-1] != "/" else "") +
                            f"{config.scenario}_{config.dataset}_latest.pt",
                            epoch, model, optim, np.array(losses), np.array(accs))
        if len(losses) < 1 or val_loss < min(losses):
            logger.info("Saving best model")
            save_checkpoint(get_lstm_dir() + config.save_path + ("_" if config.save_path[-1] != "/" else "") +
                            f"{config.scenario}_{config.dataset}_best.pt",
                            epoch, model, optim, np.array(losses), np.array(accs))
        losses.append(val_loss.item())
        accs.append(accuracy.item())

    return np.array(losses), np.array(accs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Location of config file ending with *.json. Specifying this will"
                                                   "overwrite all other arguments.")
    parser.add_argument("--scenario", type=str, help="Scenario to train on")
    parser.add_argument("--dataset", type=str, help="Whether to use trajectory or features dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size of a sample.")
    parser.add_argument("--shuffle", "-s", action="store_true", help="Shuffle dataset before sampling")
    parser.add_argument("--max_epoch", type=int, help="The maximum number of epochs to train.")
    parser.add_argument("--lr", type=float, help="Starting learning rate.")
    parser.add_argument("--dropout", type=float, help="Dropout regularisation for the LSTM,.")
    parser.add_argument("--lstm_hidden_dim", type=int, help="Dimensions of the LSTM hidden layer.")
    parser.add_argument("--fc_hidden_dim", type=int, help="Dimensions of the FC hidden layer.")
    parser.add_argument("--save_path", type=str, help="Save path for model checkpoints.")

    args = parser.parse_args()
    train(args)
