import argparse
import json
import time
import torch
from ogrit.core.base import get_base_dir
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from baselines.lstm.model import LSTMModel
from baselines.lstm.train import load_save_dataset, logger


def main(config):
    torch.random.manual_seed(42)

    if hasattr(config, "config"):
        config = argparse.Namespace(**json.load(open(config.config)))
    logger.info(config)

    test_dataset = load_save_dataset(config, "test")
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=len(test_dataset))
    test_data = [_ for _ in test_loader][0]
    logger.info(f"Running testing")

    model_dict = torch.load(get_base_dir() + "/lstm/" + config.model_path)
    model = LSTMModel(test_dataset.dataset.shape[-1],
                      config.lstm_hidden_dim,
                      config.fc_hidden_dim,
                      test_dataset.labels.unique().shape[-1],
                      num_layers=config.lstm_layers,
                      dropout=0.0)
    model.load_state_dict(model_dict["model_state_dict"])

    start = time.time()

    trajectories = test_data[0]
    target = test_data[1]
    lengths = test_data[2]
    input = pack_padded_sequence(trajectories, lengths, batch_first=True, enforce_sorted=False)

    model.eval()

    output, (encoding, lengths) = model(input, use_encoding=True)

    matches = (encoding.argmax(axis=-1) == target.unsqueeze(-1)).to(float)
    mask = (torch.arange(encoding.shape[1])[None, :] >= lengths[:, None])
    matches = matches.masked_fill(mask, 0)
    goal_probs = torch.exp(encoding)

    step = config.step
    count = int(1 / step + 1)
    if encoding.shape[1] > count:
        step_mask = torch.arange(encoding.shape[1])[None, :] % (lengths[:, None] * step - 1).ceil() == 0
        step_mask = step_mask.masked_fill(mask, 0)
        steps = step_mask.to(float).cumsum(1)
        mask = steps > count
        step_mask[mask] = False
        steps = step_mask.to(float).sum(1)
        assert (steps == count).all()
        corrects = matches.masked_select(step_mask).view((matches.shape[0], count))
        goal_probs = goal_probs.masked_select(step_mask.unsqueeze(-1)).view(
            (matches.shape[0], count, goal_probs.shape[-1]))
    else:
        corrects = (encoding.argmax(axis=-1) == target.unsqueeze(-1)).to(float)

    dur = time.time() - start

    return corrects.detach().numpy(), goal_probs.detach().numpy(), dur


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Location of config file ending with *.json. Specifying this will"
                                                   "overwrite all other arguments.")
    parser.add_argument("--scenario", type=str, help="Scenario to train on")
    parser.add_argument("--dataset", type=str, help="Whether to use trajectory or features dataset")
    parser.add_argument("--shuffle", "-s", action="store_true", help="Shuffle dataset before sampling")
    parser.add_argument("--save_path", type=str, help="Save path for model checkpoints.")

    args = parser.parse_args()

    main(args)