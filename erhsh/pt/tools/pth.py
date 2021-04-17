import torch

from erhsh.common.checkpoint import CheckpointLoader


class PtPthLoader(CheckpointLoader):
    def _load_checkpoint(self):
        param_dict = torch.load(self.checkpoint_path)
        return {k: v.detach().numpy() for k, v in param_dict.items()}
