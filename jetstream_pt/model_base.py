import torch
import itertools
from typing import Dict, Any


def _get_property_name(module, key):
  if hasattr(module, '_attr_to_property') and key in module._attr_to_property:
    return module._attr_to_property[key]
  return None


def _gather_names(module, myprefix, hf_prefix, result):
  for key, _ in itertools.chain(
      module.named_parameters(recurse=False),
      module.named_buffers(recurse=False)):
    hf_name = _get_property_name(module, key) or key
    result[hf_prefix + hf_name] = myprefix + key

  for name, child in module.named_children():
    hf_name = _get_property_name(module, name) or name
    _gather_names(child, myprefix + name + '.', hf_prefix + hf_name + '.', result)


class ModuleBase(torch.nn.Module):
  
  _attr_to_property: Dict[str, Any]

  def __init__(self):
      super().__init__()
      self._attr_to_property = {}

  def get_hf_names_to_real_name(self):
    result = {}
    _gather_names(self, '', '', result)
    return result

  def hf_name(self, orig_name, hf_name):
    self._attr_to_property[orig_name] = hf_name

  def drop_weights(self, key):
    return False