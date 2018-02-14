"""
This module contains functions to compare the
results from various pipelines.
"""

import re
from IPython.display import display, HTML


MAX_PARAMETER_LENGTH = 15

def _param_grid_to_string(param_grid):
    params_string = []
    for param, value in param_grid.items():
        param = param.split('__')
        if len(param[0]) < MAX_PARAMETER_LENGTH:
            param[0] = ' '.join(re.sub(r'([A-Z])', r' \1', param[0]).split())
        else:
            param[0] = re.sub(r'([a-z])', '', param[0])
        param[1] = param[1].replace('_', ' ').title()
        param = ': '.join(param)
        params_string.append(param + ' = ' + str(value))
    if params_string:
        params_string = '\\n'.join(params_string)
    else:
        params_string = 'Default parameters'
    return params_string

def print_results(pipelines):
    """Pretty prints the results."""
    results = pipelines.extract_results()
    results.PARAMETERS = results.apply(lambda row: _param_grid_to_string(row.PARAMETERS), axis=1)
    display(HTML(results.to_html().replace("\\n", "<br>")))
