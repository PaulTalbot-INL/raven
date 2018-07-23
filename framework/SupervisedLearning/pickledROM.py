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
  Created on May 8, 2018

  @author: talbpaul
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  Specific ROM implementation for pickledROM
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------


#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .SupervisedLearning import supervisedLearning
#Internal Modules End--------------------------------------------------------------------------------



class pickledROM(supervisedLearning):
  """
    Placeholder for ROMs that will be generated by unpickling from file.
  """
  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    self.printTag          = 'pickledROM'
    self.messageHandler    = messageHandler
    self._dynamicHandling  = False
    self.initOptionDict    = {}
    self.features          = ['PlaceHolder']
    self.target            = 'PlaceHolder'

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, confidence, float, the confidence
    """
    pass

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    pass

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    pass

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = {}
    return params

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, list, of values at which to evaluate the ROM
      @ Out, returnDict, dict, the evaluated point for each target
    """
    self.raiseAnError(RuntimeError, 'PickledROM has not been loaded from file yet!  An IO step is required to perform this action.')

  def __trainLocal__(self,featureVals,targetVals):
    """
      Trains ROM.
      @ In, featureVals, np.ndarray, feature values
      @ In, targetVals, np.ndarray, target values
    """
    self.raiseAnError(RuntimeError, 'PickledROM has not been loaded from file yet!  An IO step is required to perform this action.')
