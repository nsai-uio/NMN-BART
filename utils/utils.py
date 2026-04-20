import argparse
import io
import json
import os
import pickle
import time
import types
import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from tokenizers import Encoding as EncodingFast
from transformers.utils import TensorType, PaddingStrategy, logging, is_torch_available, is_numpy_array, is_tf_tensor, is_torch_tensor, to_py_obj
from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, BartTokenizerFast
from collections.abc import Mapping, Sized
logger = logging.get_logger(__name__)

EncodedInput = List[int]


class NmnBatchEncoding(BatchEncoding):
    def __init__(
        self, 
        data: Optional[Dict[str, Any]] = None, 
        encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None, 
        tensor_type: Union[None, str, TensorType] = None, 
        prepend_batch_axis: bool = False, 
        n_sequences: Optional[int] = None
    ):
        super().__init__(data)

        if isinstance(encoding, EncodingFast):
            encoding = [encoding]

        self._encodings = encoding

        if n_sequences is None and encoding is not None and len(encoding):
            n_sequences = encoding[0].n_sequences

        self._n_sequences = n_sequences

        self.nmn_convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    def nmn_convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
    ):
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            as_tensor = torch.tensor
            is_tensor = torch.is_tensor

        else:

            def as_tensor(value, dtype=None):
                if isinstance(value, (list, tuple)) and isinstance(value[0], (list, tuple, np.ndarray)):
                    value_lens = [len(val) for val in value]
                    if len(set(value_lens)) > 1 and dtype is None:
                        # we have a ragged list so handle explicitly
                        value = as_tensor([np.asarray(val) for val in value], dtype=object)
                return np.asarray(value, dtype=dtype)

            is_tensor = is_numpy_array

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    if key in ['input_ids', 'attention_mask', 'labels', 'vision_features', 'relation_masks']:
                        tensor = as_tensor(value)

                        # Removing this for now in favor of controlling the shape with `prepend_batch_axis`
                        # # at-least2d
                        # if tensor.ndim > 2:
                        #     tensor = tensor.squeeze(0)
                        # elif tensor.ndim < 2:
                        #     tensor = tensor[None, :]

                        self[key] = tensor
                    else:
                        tensor_list = [as_tensor(m) for m in value]
                        self[key] = tensor_list

            except Exception as e:
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    ) from e
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding with"
                    " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"
                    f" features (`{key}` in this case) have excessive nesting (inputs type `list` where type `int` is"
                    " expected)."
                ) from e

        return self

@dataclass
class DataCollatorForNmnBart(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

            
            features = self.nmn_pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

    def nmn_pad(self,
        encoded_inputs: Union[
            NmnBatchEncoding,
            List[NmnBatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> NmnBatchEncoding:
        self.model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]
        if self.__class__.__name__.endswith("Fast"):
            if not self.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False):
                logger.warning_advice(
                    f"You're using a {self.__class__.__name__} tokenizer. Please note that with a fast tokenizer,"
                    " using the `__call__` method is faster than using a method to encode the text followed by a call"
                    " to the `pad` method to get a padded encoding."
                )
                self.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if required_input is None or (isinstance(required_input, Sized) and len(required_input) == 0):
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_tensor(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self.tokenizer._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = PreTrainedTokenizerBase._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return NmnBatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items()}
            outputs = self.tokenizer._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return NmnBatchEncoding(batch_outputs, tensor_type=return_tensors)



def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def bool_str_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v


def int_flag(v):
    return int(float(v))


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def check_file(file):
    return os.path.isfile(file)


def export_config(config, path):
    param_dict = vars(config)
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)


def import_config(imported_args, existing_args):
    existing_param_dict = vars(existing_args)
    existing_param_dict.update(vars(imported_args))
    config = types.SimpleNamespace(**existing_param_dict)
    return config


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


def test_data_loader_ms_per_batch(data_loader, max_steps=10000):
    start = time.time()
    n_batch = sum(1 for batch, _ in zip(data_loader, range(max_steps)))
    return (time.time() - start) * 1000 / n_batch


def print_cuda_info():
    print('torch version:', torch.__version__)
    print('torch cuda version:', torch.version.cuda)
    print('cuda is available:', torch.cuda.is_available())
    print('cuda device count:', torch.cuda.device_count())
    print("cudnn version:", torch.backends.cudnn.version())


def move_tensor(t, device):
    if type(t) == torch.Tensor:
        return t.to(device)
    elif type(t) == list:
        return [move_tensor(x, device) for x in t]
    else:
        return t


def freeze_params(params):
    for p in params:
        p.requires_grad = False


def unfreeze_params(params):
    for p in params:
        p.requires_grad = True


def save_pickle(data, data_path):
    check_path(data_path)
    with open(data_path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_json(data, file_path):
    check_path(file_path)
    with open(file_path, "w") as f:
        json.dump(data, f, default=set_default)


def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    check_path(file_path)
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True, default=set_default))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def append_filename(filename, appendix):
    name, ext = os.path.splitext(filename)
    return "{name}_{uid}{ext}".format(name=name, uid=appendix, ext=ext)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: 
            return super().find_class(module, name)


def map_wrapper(*args, **kwargs):
    return map(*args)


def sort_dict(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}


def sort_and_normalize_dict(d):
    s = sum(d.values())
    return {k: v / s for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}