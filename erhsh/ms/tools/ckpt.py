from erhsh.common.checkpoint import CheckpointLoader


class MsCkptLoader(CheckpointLoader):
    def _load_checkpoint(self):
        from mindspore.train.serialization import load_checkpoint
        param_dict = load_checkpoint(self.checkpoint_path)
        return {k: v.asnumpy() for k, v in param_dict.items()}
