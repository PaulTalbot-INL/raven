# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 Base class for signal processing tools.
 @author: talbpw
"""
import abc
from utils import InputData, InputTypes, utils

class SignalProcessor(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    Base class for signal processors.
    Each signal processor should know how to do a couple things:
    - Train the signal process on a provided signal
      - Return characteristic attributes of the signal
      - Return the signal with trained portion removed
    - Given training parameters from first part, produce signal
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Define the acceptable input for this class.
      @ In, None
      @ Out, spec, InputData.ParameterInput, specs class with acceptable options for this object
    """
    spec = InputData.parameterInputFactory(cls.__name__, ordered=False)
    spec.addParam('name', InputTypes.StringType, required=True)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = None
    self.type = self.__class__.__name__
    self._isStochastic = False

  def handleInput(self, specs):
    """
      Read from input to set the required settings
      @ In, specs, InputData.ParameterInput, defined specifications
      @ Out, None
    """
    self.name = specs.parameterValues['name']

  @abc.abstractmethod
  def train(self, signal, **options):
    """
      @ In, signal, np.array, 1d array of floats to train on
      @ In, options, dict, settings provided to train with
      @ Out, signalParams, dict, information about the training of this signal
      @ Out, residual, np.array, 1d array of floats after training
    """

  @abc.abstractmethod
  def sample(self, signalParams, **options):
    """
      @ In, signalParams, dict, info about this training of this signal (as from "train")
      @ In, options, dict, additional keyword-based arguments
      @ Out, signal, np.array, 1d array of floats as a sample of data
    """