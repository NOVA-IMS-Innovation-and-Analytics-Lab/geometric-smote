"""
The :mod:`sklearnext.tools.model_analysis` module
contains functions to evaluate the results of model search.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import pandas as pd
from ..model_selection import ModelSearchCV
from sklearn.utils.validation import check_is_fitted


def report_model_search_results(model_search_cv, sort_results=None):
    """Generate a basic model search report of results."""
    if not isinstance(model_search_cv, ModelSearchCV):
        raise ValueError('Parameter `model_search_cv` should be a ModelSearchCV instance.')
    check_is_fitted(model_search_cv, 'cv_results_')
    columns = ['models', 'params'] + [results_param for results_param in model_search_cv.cv_results_.keys()
                                      if 'mean_test' in results_param or results_param == 'mean_fit_time']
    results = {results_param:values for results_param, values in model_search_cv.cv_results_.items()
               if results_param in columns}
    report = pd.DataFrame(results, columns=columns)
    if sort_results:
        try:
            report = report.sort_values(sort_results, ascending = (sort_results == 'mean_fit_time')).reset_index(drop=True)
        except KeyError:
            error_msg = 'Parameter `sort_results` should be one of mean_fit_score, {}. Instead {} found.'
            if isinstance(model_search_cv.scoring, list):
                options = ', '.join(['mean_test_%s' % sc for sc in model_search_cv.scoring])

            else:
                options = 'mean_test_score'
            error_msg = error_msg.format(options, sort_results)
            raise KeyError(error_msg)
    return report
