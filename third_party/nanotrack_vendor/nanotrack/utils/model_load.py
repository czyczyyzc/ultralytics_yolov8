from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch

logger = logging.getLogger("nanotrack")


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    missing_keys = [x for x in (model_keys - ckpt_keys) if not x.endswith("num_batches_tracked")]
    if missing_keys:
        logger.info("missing keys: %s", len(missing_keys))
    assert used_pretrained_keys, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    return {key.split(prefix, 1)[-1] if key.startswith(prefix) else key: value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    logger.info("load pretrained model from %s", pretrained_path)
    pretrained_dict = torch.load(pretrained_path, map_location="cpu")
    if "state_dict" in pretrained_dict:
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    try:
        check_keys(model, pretrained_dict)
    except Exception:
        new_dict = {"features." + key: value for key, value in pretrained_dict.items()}
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
