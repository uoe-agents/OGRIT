import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.lstm.datasets.lstm_dataset import LSTMDataset
from baselines.lstm.model.model import LSTMModel
from baselines.lstm.runs.lstm_writer import LSTMWriter
from ogrit.core.base import get_lstm_dir, get_results_dir
from ogrit.core.logger import logger

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

        self.logger = logger
        self.writer = None
        self.VALIDATION_STEP = 2  # How often to validate the model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train the model on these scenarios, or use the model trained on these scenarios to test on the test scenarios
        training_scenarios = configs["train_scenarios"].split(",")
        self.training_scenarios_names = "_".join(training_scenarios)
        self.batch_size = configs["batch_size"]
        self.update_hz = configs["update_hz"]

        self.input_type = configs["input_type"]

        # Load the datasets
        if mode == "train":
            train_dataset = LSTMDataset(training_scenarios, input_type=self.input_type, split_type="train",
                                        update_hz=self.update_hz)
            val_dataset = LSTMDataset(training_scenarios, input_type=self.input_type, split_type="valid",
                                      update_hz=self.update_hz)

            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=configs["shuffle"])
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=configs["shuffle"])

            input_size = train_dataset.get_num_features()
            output_size = train_dataset.get_num_classes()

        elif mode == "test":
            test_scenarios = configs["test_scenarios"].split(",")
            self.test_scenarios_names = "_".join(test_scenarios)
            test_dataset = LSTMDataset(test_scenarios, input_type=self.input_type, split_type="test",
                                       update_hz=self.update_hz)
            self.logger.info(f"Test dataset: {test_dataset}")

            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=configs["shuffle"])
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

    def _compute_loss(self, intermediate_predictions, lengths, targets):
        """
        Args:
            intermediate_predictions: tensor of shape (batch_size, seq_len, num_classes)
            lengths: list of shape (batch_size)
            targets: tensor of shape (batch_size)
        Returns:
            loss: the loss value
        """
        # Compute the loss
        loss = 0
        nr_trajectories = intermediate_predictions.shape[0]
        for i in range(nr_trajectories):
            # The loss function requires the target for each timestep, so we repeat it for all the steps in the trajectory
            targets_i = torch.tensor([targets[i]] * lengths[i])
            loss += self.loss_fn(intermediate_predictions[i, :lengths[i], :].to(self.device), targets_i.to(self.device))
        loss /= nr_trajectories
        return loss

    def train(self):

        min_loss = np.inf
        val_loss_avg_ls = []
        val_acc_ls = []
        f1_score_avg_ls = []

        self.logger.info("Starting gradient descent:")
        for epoch_nr in range(self.max_epochs):

            train_loss_avg, train_accuracy, f1_train = self.train_epoch(epoch_nr)

            if epoch_nr % self.VALIDATION_STEP == 0:
                val_loss_avg, val_acc, val_f1 = self.evaluation(epoch_nr)
                self.logger.info(
                    f"Epoch {epoch_nr + 1} - Validation Loss: {val_loss_avg} - Val. Accuracy: {val_acc} - Val. F1: {val_f1}")

                # Update the learning rate according to the validation loss
                self.scheduler.step(val_loss_avg)

                self.writer.write(epoch_nr, train_loss_avg, train_accuracy, f1_train, val_loss_avg, val_acc, val_f1)
                self.writer.flush()

                val_loss_avg_ls.append(val_loss_avg)
                val_acc_ls.append(val_acc)
                f1_score_avg_ls.append(val_f1)

                # Save the model if the validation loss is the best so far
                if val_loss_avg < min_loss:
                    min_loss = val_loss_avg
                    self.save_model(epoch_nr, np.array(val_loss_avg_ls), np.array(val_acc_ls),
                                    np.array(f1_score_avg_ls))
                # TODO: Early stopping uncomment
                # # Early stopping if the validation loss has not improved for the last 5 validation steps.
                # # 0.4*np.ceil(self.max_epochs / self.VALIDATION_STEP) is an arbitrary number
                # if len(val_loss_avg_ls) > 0.4 * np.ceil(
                #         self.max_epochs / self.VALIDATION_STEP) and val_loss_avg > np.mean(val_loss_avg_ls[-5:]):
                #     self.logger.info("Early stopping.")
                #     break

        self.writer.close()
        return np.array(val_loss_avg_ls), np.array(val_acc_ls), np.array(f1_score_avg_ls)

    def forward_pass(self, trajectories, lengths, targets, test=False):
        """
        Perform a forward pass through the model. Compute the predictions and the loss.

        Args:
            trajectories: tensor of shape (batch_size, seq_len, num_features)
            lengths: list of shape (batch_size)
            targets: tensor of shape (batch_size)
            test: if True, the loss is not computed
        Returns:
            loss: the loss value
            final_prediction: the final prediction of the model (after the last timestep)
            intermediate_predictions: the intermediate predictions of the model (after each timestep)
        """
        final_prediction, intermediate_predictions = self.model(trajectories, lengths, device=self.device)

        if test:
            return final_prediction, intermediate_predictions

        loss = self._compute_loss(intermediate_predictions, lengths, targets)
        return loss, final_prediction, intermediate_predictions

    def compute_accuracy(self, final_prediction, intermediate_predictions, lengths, targets):
        """
        Compute the accuracy of the model.
        Args:
            final_prediction: the final prediction of the model (after the last timestep)
            intermediate_predictions: the intermediate predictions of the model (after each timestep)
            targets: tensor of shape (batch_size)
        Returns:
            correct: nr of correct predictions
            total: total nr of predictions
        """
        _, predicted = torch.max(final_prediction.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()

        # compute f1 score
        f1 = f1_score(targets.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        return correct, total, f1

    def train_epoch(self, epoch_nr):
        """
        Train the model for one epoch.
        Args:
            epoch_nr: nr of the current epoch
        """

        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        f1_accuracy = 0

        for i_batch, sample_batched in enumerate(tqdm(self.train_loader)):
            trajectories = sample_batched[0].to(self.device)
            targets = sample_batched[1].to(self.device)
            lengths = sample_batched[2].to(self.device)

            loss, final_prediction, intermediate_predictions = self.forward_pass(trajectories=trajectories,
                                                                                 lengths=lengths, targets=targets)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()  # Compute the gradients

            # Update the weights
            self.optimizer.step()

            new_correct, new_total, f1 = self.compute_accuracy(final_prediction, intermediate_predictions, lengths,
                                                               targets)
            f1_accuracy += f1
            correct += new_correct
            total += new_total

            running_loss += loss.item()

        return running_loss / len(self.train_loader), correct / total, f1_accuracy / len(self.train_loader)

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
                lengths = sample_batched[2].to(self.device)

                loss, final_prediction, intermediate_predictions = self.forward_pass(trajectories=trajectories,
                                                                                     lengths=lengths, targets=target)

                new_correct, new_total, f1 = self.compute_accuracy(final_prediction, target)

                f1_accuracy += f1
                correct += new_correct
                total += new_total

                running_loss += loss.item()

        return running_loss / len(self.val_loader), correct / total, f1_accuracy / len(self.val_loader)

    def test(self):

        model_dict = torch.load(self.model_path, map_location=torch.device('cpu'))

        self.model.load_state_dict(model_dict['model_state_dict'])
        self.model.to(self.device)

        self.model.eval()
        self.logger.info(f"Running test")

        goal_probs_df = {"true_goal_prob": [], "fraction_observed": []}

        # Compute the goal probabilities for each trajectory
        for i_batch, sample_batched in enumerate(tqdm(self.test_loader)):
            trajectories = sample_batched[0].to(self.device)
            targets = sample_batched[1].to(self.device)
            lengths = sample_batched[2].to(self.device)
            fraction_observed = sample_batched[3].to(self.device)

            final_prediction, intermediate_predictions = self.forward_pass(trajectories=trajectories,
                                                                           lengths=lengths, targets=targets,
                                                                           test=True)
            goal_probs = torch.exp(intermediate_predictions).cpu().detach().numpy()
            lengths = lengths.cpu().detach().numpy()

            for i in range(len(trajectories)):
                # Get the probability assigned by the LSTM to the true goal by the i-th trajectory
                # It is a list of length lengths[i]
                true_goal_prob_timestep = goal_probs[i, :lengths[i], targets[i]]

                if i == 0:
                    assert true_goal_prob_timestep[-1] == goal_probs[i, lengths[i] - 1, targets[i]]
                    assert true_goal_prob_timestep[0] == goal_probs[i, 0, targets[i]]

                goal_probs_df["true_goal_prob"].extend(true_goal_prob_timestep)
                goal_probs_df["fraction_observed"].extend(np.round(fraction_observed[i, :lengths[i]], 1))

        goal_probs_df = pd.DataFrame(goal_probs_df)

        # save true goal probability
        fraction_observed_grouped = goal_probs_df.groupby('fraction_observed')
        true_goal_prob = fraction_observed_grouped.mean()
        true_goal_prob_sem = fraction_observed_grouped.std() / np.sqrt(fraction_observed_grouped.count())

        true_goal_prob_sem.to_csv(
            get_results_dir() + f'/{self.test_scenarios_names}_lstm_{self.input_type}_on_{self.training_scenarios_names}_true_goal_prob_sem.csv')
        true_goal_prob.to_csv(
            get_results_dir() + f'/{self.test_scenarios_names}_lstm_{self.input_type}_on_{self.training_scenarios_names}_true_goal_prob.csv')

    def get_model_path(self):

        model_dir = os.path.join(get_lstm_dir(), f"checkpoint/")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        return os.path.join(model_dir, f"{self.training_scenarios_names}_{self.input_type}.pt")

    def save_model(self, epoch, losses, accs, f1_scores):

        self.logger.info(f"Saving model to {self.model_path}")

        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': losses,
            'accs': accs,
            'f1_scores': f1_scores
        }, self.model_path)


if __name__ == "__main__":
    raise Exception("WARNING - This file is not meant to be run directly. Please run the get_results.py file in "
                    "/OGRIT/get_results.py.")
