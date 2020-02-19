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
from utils import InputData, InputTypes
from .SignalProcessor import SignalProcessor

class Fourier(SignalProcessor):
  """
    Trains predetermined Fourier periods on signal.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Define the acceptable input for this class.
      @ In, None
      @ Out, spec, InputData.ParameterInput, specs class with acceptable options for this object
    """
    spec = SignalProcessor.getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('periods', contentType=InputTypes.FloatListType))
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    SignalProcessor.__init__(self)
    self._bases = []

  def handleInput(self, specs):
    """
      Read from input to set the required settings
      @ In, specs, InputData.ParameterInput, defined specifications
      @ Out, None
    """
    SignalProcessor.handleInput(self, specs)
    self._bases = specs.findFirst('bases').value

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