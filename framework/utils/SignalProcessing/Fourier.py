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
import copy
import collections
import numpy as np
from sklearn import linear_model
from utils import InputData, InputTypes, mathUtils
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

  def train(self, signal, dependent, **options):
    """
      @ In, signal, np.array, 1d array of floats to train on (e.g. Signal)
      @ In, dependent, np.array, 1d array of floats as dependent (e.g. Time)
      @ In, options, dict, settings provided to train with
      @ Out, signalParams, dict, information about the training of this signal
      @ Out, residual, np.array, 1d array of floats after training
    """
    periods = self._bases # TODO as an input so this is more functional?
    # TODO masks/filters for splitting fits?
    # generate Fourier signals to fit
    fouriers = self._generateFourierSignals(dependent, self._bases)

    # collinearity check -> see if we can do all coeffs at once
    condition = np.linalg.cond(fouriers)
    if condition > 30:
      self.raiseADebug('Fourier fitting condition number is {:1.1e}!'.format(condition) +
                       ' Calculating iteratively instead of simultaneously.')
      # fouriers has shape (H, 2F) where H is dependents length and F is number of bases.
      ## They're also sorted in order of period from largets to smallest, with sine, cosine for each.
      H, F2 = fouriers.shape
      residual = copy.deepcopy(signal[:]) # residual that needs fitting
      intercept = 0
      coeffs = np.zeros(F2)
      for fn in range(F2):
        fSignal = fouriers[:, fn]
        eng = linear_model.LinearRegression(normalize=False)
        eng.fit(fSignal.reshape(H, 1), residual)
        thisIntercept = eng.intercept_
        thisCoeff = eng.coef_[0]
        coeffs[fn] = thisCoeff
        intercept += thisIntercept
        # update residual signal for further training
        fitted = thisIntercept + thisCoeff * fSignal
        residual -= fitted
    else:
      self.raiseADebug('Fourier fitting condition number is {:1.1e}.'.format(condition) +
                      ' Calculating simultaneously.')
      # build regressor
      fitter = linear_model.LinearRegression(normalize=False)
      fitter.fit(fouriers, signal)
      intercept = fitter.intercept_
      coeffs = fitter.coef_

      # we have A * sin(f t) + B * cos(f t), want C sin(f t + p)
      ## collect all sine-cosine coefficients
      waveCoeffMap = collections.defaultdict(dict) # {period: {sin:#, cos:#}}
      for c, coef in enumerate(coeffs):
        period = periods[c//2]
        waveform = 'cos' if c % 2 else 'sin' # c % 2 is 0 for sin
        waveCoeffMap[period][waveform] = coef
      ## convert to magnitude-phase coefficients
      coefMap = {}
      fit = np.ones(dependent.size) * intercept
      for period, coefs in waveCoeffMap.items():
        A = coefs['sin']
        B = coefs['cos']
        C, s = mathUtils.convertSinCosToSinPhase(A, B)
        coefMap[period] = {'amplitude': C, 'phase': s}
        fit += mathUtils.evalFourier(period, C, s, dependent)
      # store results
      results = {'regression': {'intercept': intercept,
                                'coeffs': coefMap,
                                'periods': periods},
                 'predict': fit} # TODO should we be storing this?
      return results, signal - fit

  def sample(self, signalParams, **options):
    """
      @ In, signalParams, dict, info about this training of this signal (as from "train")
      @ In, options, dict, additional keyword-based arguments
      @ Out, signal, np.array, 1d array of floats as a sample of data
    """
    return signalParams['predict']

  def _generateFourierSignals(self, dependent, periods):
    """
      Generates superposition of Fourier signals
      @ In, dependent, np.array, 1d array of dependent variables (e.g. Time)
      @ In, periods, list, list of float periods to generate Fourier for
      @ Out, signal, np.array, matrix representing signal, shape is [n_timesteps, 2*n_periods]
    """
    fourier = np.zeros((dependent.size, 2*len(periods))) # sin, cos for each period
    for p, period in enumerate(periods):
      hist = 2. * np.pi / period * dependent
      fourier[:, 2 * p] = np.sin(hist)
      fourier[:, 2 * p + 1] = np.cos(hist)
    return fourier