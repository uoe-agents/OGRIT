import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from ogrit.core.base import get_lstm_dir


class LSTMWriter:

    def __init__(self, scheduler):
        path_for_training_logs = os.path.join(get_lstm_dir(), 'runs/')
        if not os.path.exists(path_for_training_logs):
            os.mkdir(path_for_training_logs)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(path_for_training_logs, f'train_lstm_{self.timestamp}'))

        self.scheduler = scheduler

    def write(self, epoch, train_loss, train_accuracy, f1_train, val_loss, val_acc, val_f1):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        self.writer.add_scalar('F1/train', f1_train, epoch)
        self.writer.add_scalar('F1/val', val_f1, epoch)

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()
