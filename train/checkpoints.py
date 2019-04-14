import torch
from pathlib import Path
import sys


class CheckpointManager:  # Don't know if this works in graph mode...
    """
    A checkpoint manager for pytorch models and optimizers loosely based on Keras/Tensorflow Checkpointers.
    I should note that I am not sure whether this works in Pytorch graph mode.
    Giving up on saving as HDF5 files like in Keras. Just too annoying.
    """
    def __init__(self, model, optimizer, ckpt_dir='./checkpoints', max_to_keep=5):

        # Type checking.
        assert isinstance(model, torch.nn.Module), 'Not a Pytorch Model'
        assert isinstance(optimizer, torch.optim.Optimizer), 'Not a Pytorch Optimizer'
        assert isinstance(max_to_keep, int) and (max_to_keep >= 0), 'Not a non-negative integer'
        ckpt_path = Path(ckpt_dir)
        assert ckpt_path.exists(), 'Not a valid, existing path'

        record_path = ckpt_path / 'Checkpoints.txt'

        try:
            record_file = open(record_path, mode='x')
        except FileExistsError:
            print('WARNING: It is recommended to have a separate checkpoint directory for each run.', file=sys.stderr)
            print('Appending to previous Checkpoint record file!')
            record_file = open(record_path, mode='a')

        print(f'Checkpoint List for {ckpt_path}', file=self.record_file)

        self.model = model
        self.optimizer = optimizer
        self.ckpt_path = ckpt_path
        self.max_to_keep = max_to_keep
        self.save_counter = 0
        self.record_file = record_file  # No way to close this until program finishes execution
        self.record_dict = dict()

    def save(self, ckpt_name=None):
        save_dict = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
        save_path = self.ckpt_path / (f'{ckpt_name}.tar' if ckpt_name else f'ckpt_{self.save_counter:03d}.tar')

        self.save_counter += 1
        torch.save(save_dict, save_path)
        print(f'Saved Checkpoint to {save_path}')
        print(f'Checkpoint {self.save_counter:04d}: {save_path}')
        self.record_dict[self.save_counter] = save_path

        if self.save_counter > self.max_to_keep:
            for count, path in self.record_dict.items():
                if count <= (self.save_counter - self.max_to_keep):
                    path.unlink()  # Delete existing checkpoint

    def load(self, load_dir):
        pass

    def load_latest(self, load_root):
        pass
