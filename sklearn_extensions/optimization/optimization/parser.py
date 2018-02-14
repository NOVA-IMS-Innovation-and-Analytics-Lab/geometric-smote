"""
This module various parsers
for configuration files.
"""

from importlib import import_module
from inspect import getmembers, isclass
from configparser import ConfigParser
from ast import literal_eval
from .pipelines import Pipelines
from .split import TimeSeriesSplit


MODULES = ['sklearn', 'imblearn', 'category_encoders']
DEPRECATED_MODULES = ['sklearn.cross_validation', 'sklearn.grid_search', 'sklearn.learning_curve']
DEFAULT_OPTIONS = {'Scoring': ['neg_mean_squared_error'], 'Splits': 3, 'Span': 10, 'CPUs': 1}

def _parse_params_values(params_values):
    params_values = params_values.split('(')
    params_values[1] = params_values[1].replace(')', '').split(',')
    try:
        params_values = {params_values[0]:[literal_eval(value) for value in params_values[1]]}
    except ValueError:
        params_values = {params_values[0]:params_values[1]}
    return params_values

def _parse_params(params):
    params = params.replace(' ', '')
    params = params.split('+')
    params = [grid.split('*') for grid in params]
    params = [[_parse_params_values(params_values) for params_values in param] for param in params]
    param_grids = []
    for param in params:
        param_grid = {}
        for param_values in param:
            param_grid.update(param_values)
        param_grids.append(param_grid)
    return param_grids

def _parse_imported_classes(classes):
    classes = classes.copy()
    all_modules = []
    class_map = {}
    for module_name in MODULES:
        module = import_module(module_name)
        members = getmembers(module)
        module_classes = {name:value for name, value in members\
                          if isclass(value) and name in classes}
        class_map.update(module_classes)
        modules = [[module_name + '.' + module for module in value]\
                   for key, value in members if key == '__all__'][0]
        all_modules += modules
    all_modules = [module for module in all_modules if module not in DEPRECATED_MODULES]
    for module in all_modules:
        try:
            mod = import_module(module)
        except ModuleNotFoundError:
            pass
        for cl in classes:
            try:
                class_map.update({cl:getattr(mod, cl)})
            except AttributeError:
                pass
        if set(class_map.keys()) == classes:
            break
    return class_map

def parse_pipelines(config_file):
    """Parses the pipelines from a
    configuration file."""

    # Read configuration file
    config = ConfigParser(allow_no_value=True, strict=False)
    config.optionxform = str
    config.read(config_file)

    # Parse pipelines
    pipelines_names = [sec for sec in config.sections() if sec != 'OPTIONS']
    pipelines = [list(config[section].items()) for section in pipelines_names]

    # Parse classes to import
    classes = set()
    for section in pipelines_names:
        classes = classes.union(set(config[section].keys()))
    class_map = _parse_imported_classes(classes)

    # Parse estimators
    estimators = []
    for pipeline in pipelines:
        estimator = []
        for est_name, params in pipeline:
            if params is not None:
                estimator.append((est_name, class_map[est_name](), _parse_params(params)))
            else:
                estimator.append((est_name, class_map[est_name]()))
        estimators.append(estimator)
    estimators = list(zip(pipelines_names, estimators))

    # Parse options section
    options = {}
    for option_key, default_option_value in DEFAULT_OPTIONS.items():
        options[option_key] = config.get('OPTIONS', option_key, fallback=None)
        if options[option_key] is None:
            options[option_key] = default_option_value
        else:
            if option_key != 'Scoring':
                options[option_key] = literal_eval(options[option_key])
            else:
                options['Scoring'] = options['Scoring'].replace(' ', '').split(',')

    # Create pipelines
    pipelines = Pipelines(estimators,
                          options['Scoring'],
                          TimeSeriesSplit(n_splits=options['Splits'], time_span=options['Span']),
                          options['CPUs'])

    return pipelines
