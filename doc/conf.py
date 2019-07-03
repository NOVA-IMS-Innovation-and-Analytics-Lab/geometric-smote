import os
import sys
import sphinx_rtd_theme


sys.path.insert(0, os.path.abspath('sphinxext'))
from github_link import make_linkcode_resolve
import sphinx_gallery


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'numpydoc',
    'sphinx_issues',
    'sphinx_gallery.gen_gallery',
]

numpydoc_show_class_members = False
import sphinx
from distutils.version import LooseVersion
if LooseVersion(sphinx.__version__) < LooseVersion('1.4'):
    extensions.append('sphinx.ext.pngmath')
else:
    extensions.append('sphinx.ext.imgmath')
autodoc_default_flags = ['members', 'inherited-members']
source_suffix = '.rst'
plot_gallery = True
master_doc = 'index'
project = 'geometric-smote'
copyright = '2019, G. Douzas'
author = 'Georgios Douzas'
version = '0.1'
release = '0.1'
add_function_parentheses = False
pygments_style = 'sphinx'

html_theme = 'sphinx_rtd_theme'

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']

htmlhelp_basename = 'geometric-smote-doc'



latex_elements = {}
latex_documents = [
    ('index', 'geometric-smote.tex', 'geometric-smote Documentation',
     'G. Douzas', 'manual'),
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'sklearn': ('http://scikit-learn.org/stable', None)
}
sphinx_gallery_conf = {
    'doc_module': 'gsmote',
    'backreferences_dir': os.path.join('generated'),
    'reference_url': {
        'gsmote': None}
}


man_pages = [('index', 'geometric-smote', 'geometric-smote Documentation',
              ['G. Douzas'], 1)]

texinfo_documents = [
    ('index', 'geometric-smote', 'geometric-smote Documentation',
     'G. Douzas', 'geometric-smote',
     'Geometric SMOTE algorithm.', 'Miscellaneous'),
]




issues_uri = 'https://github.com/AlgoWit/geometric-smote/issues/{issue}'
issues_github_path = 'AlgoWit/geometric-smote'
issues_user_uri = 'https://github.com/{user}'


linkcode_resolve = make_linkcode_resolve('imblearn',
                                         'https://github.com/scikit-learn-contrib/'
                                         'imbalanced-learn/blob/{revision}/'
                                         '{package}/{path}#L{lineno}')

# on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
# if not on_rtd:
#     import sphinx_theme
#     html_theme = 'stanford_theme'
#     html_theme_path = [sphinx_theme.get_html_theme_path('stanford_theme')]


autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', '_templates']




