# Configuration
import os
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    import sphinx_theme
    html_theme = 'stanford_theme'
    html_theme_path = [sphinx_theme.get_html_theme_path('stanford_theme')]
master_doc = 'index'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_static_path = ['_static']
extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx'
]

# Project information
project = 'geometric-smote'
copyright = '2019, Georgios Douzas'
author = 'Georgios Douzas'
release = '0.0.1'
