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
import numpy as np

def run(raven, info):
  """
    A dummy model used to check the connections for the EnsembleModel
    @ In, raven, object, object to store members on
    @ In, info, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  raven.d = raven.c.sum()
