import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.lstm.datasets.dataset import OGRITFeatureDataset
from baselines.lstm.lstm_logger import Logger
from baselines.lstm.model.model import LSTMModel
from baselines.lstm.runs.lstm_writer import LSTMWriter
from ogrit.core.base import get_lstm_dir, get_results_dir

"""
The LSTM takes in a list of features at each timestep, the same as those that OGRIT gets to evaluate the 
probability that the agent will take a certain action-type. 
"""


class FeaturesLSTM:
    def __init__(self, configs, mode="train"):
        """
        Args:
            configs: dict containing the following keys:
                        for mode=="train":
                            batch_size, lr, input_size, lstm_hidden_size, fc_hidden_shape, out_shape, lstm_layers,
                            dropout, seed, max_epochs, shuffle, train_scenarios.
                        for mode=="test":
                            batch_size, input_size, lstm_hidden_size, fc_hidden_shape, out_shape, lstm_layers,
                            dropout, seed, shuffle, train_scenarios, test_scenarios.
            mode: "train" or "test" to either train or test the model.
        Notes:
            See the /OGRIT/baselines/lstm/get_results.py file for the default values of the hyper-parameters.
        """

        self.logger = Logger()
        self.writer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train the model on these scenarios, or use the model trained on these scenarios to test on the test scenarios
        training_scenarios = configs["train_scenarios"].split(",")
        self.training_scenarios_names = "_".join(training_scenarios)
        self.batch_size = configs["batch_size"]

        # Load the datasets
        if mode == "train":
            train_dataset = OGRITFeatureDataset(training_scenarios, split_type="train")
            self.logger.info(f"Train dataset: {train_dataset}")
            val_dataset = OGRITFeatureDataset(training_scenarios, split_type="valid")
            self.logger.info(f"Validation dataset: {val_dataset}")

            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=configs["shuffle"])
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=configs["shuffle"])

            input_size = train_dataset.get_num_features()
            output_size = train_dataset.get_num_classes()

        elif mode == "test":
            test_scenarios = configs["test_scenarios"].split(",")
            self.test_scenarios_names = "_".join(test_scenarios)
            test_dataset = OGRITFeatureDataset(test_scenarios, split_type="test")
            self.logger.info(f"Test dataset: {test_dataset}")

            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                          shuffle=configs["shuffle"])
            input_size = test_dataset.get_num_features()
            output_size = test_dataset.get_num_classes()
        else:
            raise ValueError(f"Mode {mode} not supported.")

        dropout = configs["dropout"] if mode != "test" else 0.0
        # Use the model defined in baselines/lstm/model/model.py
        self.model = LSTMModel(in_shape=input_size,
                               lstm_hidden_shape=configs["lstm_hidden_size"],
                               fc_hidden_shape=configs["fc_hidden_shape"],
                               out_shape=output_size,
                               num_layers=configs["lstm_layers"],
                               dropout=dropout)

        self.logger.info(f"Model created: {str(self.model)}")

        self.dataset = "features"
        self.model_path = self.get_model_path()

        if mode == "train":
            # Use multiple GPUs if available
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.logger.info(f"Training on {self.device} (CUDA: {torch.cuda.device_count()}).")
            self.model.to(self.device)

            self.max_epochs = configs["max_epochs"]

            # Define the loss function and optimizer
            self.loss_fn = nn.NLLLoss()
            self.loss_fn.to(self.device)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configs["lr"])
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5)

            self.writer = LSTMWriter(scheduler=self.scheduler)
        torch.random.manual_seed(configs["seed"])

    def train(self):

        min_loss = np.inf
        val_loss_avg_ls = []
        f1_score_avg_ls = []

        self.logger.info("Starting gradient descent:")
        for epoch_nr in range(self.max_epochs):

            train_loss_avg, train_accuracy = self.train_epoch(epoch_nr)

            val_loss_avg, val_acc = self.evaluation(epoch_nr)  # TODO: clarify if accuracy or F1 score...
            self.logger.info(f"Epoch {epoch_nr + 1} - Validation Loss: {val_loss_avg} - Accuracy: {val_acc}")

            # Update the learning rate according to the validation loss
            self.scheduler.step(val_loss_avg)

            self.logger.info(f"Validation Loss: {val_loss_avg}; Accuracy: {val_acc} "
                             f"LR: {self.optimizer.param_groups[0]['lr']}")

            self.writer.write(epoch_nr, train_loss_avg, val_loss_avg, val_acc, 0)  # TODO: last one should be val_f1
            self.writer.flush()

            val_loss_avg_ls.append(val_loss_avg)
            f1_score_avg_ls.append(val_acc)

            # Save the model if the validation loss is the best so far
            if val_loss_avg < min_loss:
                min_loss = val_loss_avg
                self.save_model(epoch_nr, np.array(val_loss_avg_ls), np.array(f1_score_avg_ls))

        self.writer.close()
        return np.array(val_loss_avg_ls), np.array(f1_score_avg_ls)

    def train_epoch(self, epoch_nr):
        """
        Train the model for one epoch.
        Args:
            epoch_nr: nr of the current epoch
        """

        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for i_batch, sample_batched in enumerate(self.train_loader):
            trajectories = sample_batched[0].to(self.device)
            target = sample_batched[1].to(self.device)

            predictions, _ = self.model(trajectories, device=self.device)

            self.optimizer.zero_grad()

            loss = self.loss_fn(predictions, target)
            loss.backward()  # Compute the gradients

            self.optimizer.step()  # Update the weights

            running_loss += loss.item()

            _, predicted = torch.max(predictions, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            self.logger.info(
                f"Epoch: {epoch_nr}; Step: {len(self.train_loader) * epoch_nr + i_batch}; Loss: {loss.item()}")

        return running_loss / len(self.train_loader), correct / total

    def evaluation(self, epoch_nr):
        """
        Evaluate the model on the validation set.
        Args:
            epoch_nr: nr of the current epoch
        """

        self.model.eval()
        self.logger.info(f"Running validation")

        running_loss = 0.0
        f1_accuracy = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.val_loader):
                trajectories = sample_batched[0].to(self.device)
                target = sample_batched[1].to(self.device)

                predictions, _ = self.model(trajectories, device=self.device)
                _, predicted = torch.max(predictions, 1)

                loss = self.loss_fn(predictions, target)

                running_loss += loss.item()

                # Compute F1 score as the data may be imbalanced TODO
                # f1_accuracy += f1_score(target.cpu().data, predicted.cpu())

                _, predicted = torch.max(predictions, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return running_loss / len(self.val_loader), correct / total

    def test(self):

        model_dict = torch.load(self.model_path, map_location=torch.device('cpu'))

        self.model.load_state_dict(model_dict['model_state_dict'])

        self.model.eval()
        self.logger.info(f"Running test")

        goal_probs_df = {"true_goal_prob": [], "fraction_observed": []}

        # Compute the goal probabilities for each trajectory
        for i_batch, sample_batched in enumerate(self.test_loader):
            trajectories = sample_batched[0]
            targets = sample_batched[1].detach().numpy()
            fraction_observed = sample_batched[2].detach().numpy()

            # For each trajectory, compute the sub-trajectories and their goal probabilities
            # predictions will be a (batch_size, n_goals) tensor with the goal probabilities for each trajectory
            predictions, _ = self.model(trajectories, device=self.device)
            goal_probs = torch.exp(predictions).detach().numpy()

            # Get the probability assigned by the LSTM to the true goal.
            true_goal_prob = goal_probs[np.arange(self.batch_size), targets]

            goal_probs_df["true_goal_prob"].extend(true_goal_prob)
            goal_probs_df["fraction_observed"].extend(np.round(fraction_observed, 1))

        goal_probs_df = pd.DataFrame(goal_probs_df)

        # save true goal probability
        fraction_observed_grouped = goal_probs_df.groupby('fraction_observed')
        true_goal_prob = fraction_observed_grouped.mean()
        true_goal_prob_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())

        true_goal_prob_sem.to_csv(
            get_results_dir() + f'/{self.test_scenarios_names}_lstm_on_{self.training_scenarios_names}_true_goal_prob_sem.csv')
        true_goal_prob.to_csv(
            get_results_dir() + f'/{self.test_scenarios_names}_lstm_on_{self.training_scenarios_names}_true_goal_prob.csv')

    def get_model_path(self):

        model_dir = os.path.join(get_lstm_dir(), f"checkpoint/")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        return os.path.join(model_dir, f"{'_'.join(self.training_scenarios_names)}_{self.dataset}.pt")

    def save_model(self, epoch, losses, accs):

        self.logger.info(f"Saving model to {self.model_path}")

        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': model_state_dict,
            'losses': losses,
            'accs': accs
        }, self.model_path)


if __name__ == "__main__":
    raise Exception("WARNING - This file is not meant to be run directly. Please run the get_results.py file in "
                    "/OGRIT/get_results.py.")
