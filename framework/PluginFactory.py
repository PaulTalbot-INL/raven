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
Created on October 23, 2019

@author: talbpaul

comment: Superseding the ModelPluginFactory, this factory collects the entities from plugins
         and makes them available to the various entity factories in RAVEN.
"""
import os
import sys
import types
import pprint
import inspect
import warnings
import importlib
from collections import defaultdict

from PluginsBaseClasses import PluginBase
from utils import utils, xmlUtils

warnings.simplefilter('default', DeprecationWarning)

# Design Note: This module is meant to be loaded BEFORE any other entities are loaded!
#              It is critical that factories can inquire the plugin list to populate their modules.

## custom errors
class PluginError(RuntimeError):
  pass

## Method definitions
def loadPlugins(path, catalogue):
  """
    Loads plugins and their entities into factory storage.
    @ In, path, str, location of plugins folder
    @ In, catalogue, str, location of plugins directory xml file
    @ Out, None
  """
  pluginsRoot, _ = xmlUtils.loadToTree(catalogue)
  for pluginNode in pluginsRoot:
    name = pluginNode.find('name').text
    location = pluginNode.find('location').text
    if location is None:
      raise PluginError('Installation is corrupted for plugin "{}". Check raven/plugins/plugin_directory.xml and try reinstalling using raven/scripts/intall_plugin')
    if name is None:
      name = os.path.basename(location)
    print('Loading plugin "{}" at {}'.format(name, location))
    module = loadPluginModule(name, location)
    loadEntities(name, module)

def loadPluginModule(name, location):
  """
    Loads the plugin as a package
    @ In, name, str, name of plugin package
    @ In, location, str, path to plugin main directory
    @ Out, plugin, module instance, root module of package
  """
  # TODO for now assume all entities are named in the package
  # load the package loading specs
  spec = importlib.util.spec_from_file_location(name, os.path.join(location, '__init__.py'))
  if spec is None:
    raise PluginError('Plugin "{}" does not appear to be set up as a package!'.format(name))
  # instance of the module
  plugin = importlib.util.module_from_spec(spec)
  # add it to the namespace TODO is this a good idea?
  sys.modules[spec.name] = plugin
  # load the module
  spec.loader.exec_module(plugin)
  print(' ... successfully imported "{}" ...'.format(name))
  return plugin

def loadEntities(name, plugin):
  """
    Loads the RAVEN entities in a package.
    Uses inheritance to determine what plugin entities belong where
    @ In, name, str, name of plugin package
    @ In, plugin, module instance, root module of package
    @ Out, None
  """
  # get the modules that are part of this package
  for pluginMemberName, pluginMember in inspect.getmembers(plugin):
    if inspect.ismodule(pluginMember):
      # get the classes that are part of the module
      ## only get classes that inherit from the PluginBase class
      for candidateName, candidate in inspect.getmembers(pluginMember):
        if inspect.isclass(candidate) and inspect.getmodule(candidate) == pluginMember \
                                      and issubclass(candidate, PluginBase.PluginBase):
          # find the first parent class that is a PluginBase class
          ## NOTE [0] is the class itself
          for parent in inspect.getmro(candidate)[1:]:
            if issubclass(parent, PluginBase.PluginBase):
              break
            # guaranteed to be found, I think, so no else
          # check validity
          print('DEBUGG candidate:', candidate)
          if not candidate.isAValidPlugin():
            raise PluginError('Invalid plugin entity: "{}" from plugin "{}"'.format(candidateName, name))
          registerName = '{}.{}'.format(name, candidateName)
          pluginEntities[candidate.entityType][registerName] = candidate
          print(' ... registered "{}" as a "{}" RAVEN entity.'.format(registerName, candidate.entityType))

def getEntities(entityType):
  """
    Provides loaded entities.
    @ In, entityType, str, class of entity to load (e.g. ExternalModel)
    @ Out, pluginEntities, dict, name: class of plugin entities
  """
  return pluginEntities.get(entityType, {})

## factory loading

# storage for available entities
# structure is pluginEntities[raven entity name] = {name: import command} (TODO check this is good)
pluginEntities = defaultdict(dict)

# load plugins directory and collect plugins
pluginsPath = os.path.join(os.path.dirname(__file__), '..', 'plugins')
# use "catalogue" to differentiate between "path" and "directory"
pluginsCatalogue = os.path.abspath(os.path.join(pluginsPath, 'plugin_directory.xml'))
# if no installed plugins, report and finish; otherwise, load plugins
if os.path.isfile(pluginsCatalogue):
  loadPlugins(pluginsPath, pluginsCatalogue)
else:
  print('PluginFactory: No installed plugins detected.')
  print('DEBUGG file:', pluginsCatalogue)




#### OLD ####
__moduleInterfaceList = []
startDir = os.path.join(os.path.dirname(__file__),'../../plugins')
for dirr,_,_ in os.walk(startDir):
  __moduleInterfaceList.extend(glob(os.path.join(dirr,"*.py")))
  utils.add_path(dirr)
__moduleImportedList = []
__basePluginClasses  = {'ExternalModel':'ExternalModelPluginBase'}

"""
 Interface Dictionary (factory) (private)
"""
__base                          = 'ModelPlugins'
__interFaceDict                 = defaultdict(dict)
for moduleIndex in range(len(__moduleInterfaceList)):
  if 'class' in open(__moduleInterfaceList[moduleIndex]).read():
    __moduleImportedList.append(utils.importFromPath(__moduleInterfaceList[moduleIndex],False))
    for key,modClass in inspect.getmembers(__moduleImportedList[-1], inspect.isclass):
      for base in modClass.__bases__:
        for ravenEntityName, baseClassName in __basePluginClasses.items():
          if base.__name__ == baseClassName:
            __interFaceDict[ravenEntityName][key] = modClass
            # check the validity of the plugin
            if not modClass.isAvalidPlugin():
              raise IOError("The plugin based on the class "+ravenEntityName.strip()+" is not valid. Please check with the Plugin developer!")
__knownTypes = [item for sublist in __interFaceDict.values() for item in sublist]

def knownTypes():
  """
    Method to return the list of known model plugins
    @ In, None
    @ Out, __knownTypes, list, the list of known types
  """
  return __knownTypes

def returnPlugin(Type,subType,caller):
  """
    this allows the code(model) class to interact with a specific
     code for which the interface is present in the CodeInterfaces module
    @ In, Type, string, the type of plugin main class (e.g. ExternalModel)
    @ In, subType, string, the subType of the plugin specialized class (e.g. CashFlow)
    @ In, caller, instance, instance of the caller
  """
  if subType not in knownTypes():
    caller.raiseAnError(NameError,'not known '+__base+' type '+Type+' subType '+Type)
  return __interFaceDict[Type][subType]()
