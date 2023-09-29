# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import pathlib
import sys
import warnings

import xibs

# ignore numpy warnings, see:
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

TOPLEVEL_DIR = pathlib.Path(__file__).parent.parent.absolute()

if str(TOPLEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(TOPLEVEL_DIR))

# This is to tell Sphinx how to print some specific type annotations
# See: https://stackoverflow.com/a/67483317
# See: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_type_aliases
autodoc_type_aliases = {"ArrayLike": "ArrayLike"}


# -- Project information -----------------------------------------------------

project = "xibs"
copyright = f"2023, {xibs.__author__}"
author = xibs.__author__

rst_prolog = f""":github_url: {xibs.__url__}"""

# The full version, including alpha/beta/rc tags
version = xibs.__version__
release = xibs.__version__

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "7.0"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames. You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation for a list of supported languages.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "docs", "docker", "tests", ".github", ".vscode"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "obj"

# -- Extensions Configuration ---------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    "sphinx.ext.autosectionlabel",  # Allow reference sections using its title
    "sphinx.ext.autosummary",  # Generate autodoc summaries
    "sphinx.ext.coverage",  # Collect doc coverage stats
    "sphinx.ext.doctest",  # Test snippets in the documentation
    "sphinx.ext.githubpages",  # Publish HTML docs in GitHub Pages
    "sphinx.ext.intersphinx",  # Link to other projects’ documentation
    "sphinx.ext.mathjax",  # Render math via JavaScript
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.todo",  # Support for todo items
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinxcontrib.bibtex",  # Insert BibTeX citations into Sphinx documentation
    "sphinx_copybutton",  # Add a "copy" button to code blocks
    "sphinx_issues",  # Link to project's issue tracker
    "sphinx-prompt",  # prompt symbols will not be copy-pastable
    "sphinx_codeautolink",  # Automatically link example code to documentation source
]

# Config for autosectionlabel extension
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Config for the napoleon extension
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# Configuration for sphinx.ext.todo
todo_include_todos = True

# Config for the sphinx_issues extension
issues_github_path = "fsoubelet/xibs"

# Config for the sphinxcontrib.bibtex extension
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "navigation_depth": 3,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# A dictionary of values to pass into the template engine’s context for all
# pages. Single values can also be put in this dictionary using the
# -A command-line option of sphinx-build.
html_context = {
    "display_github": True,
    # the following are only needed if :github_url: is not set
    "github_user": author,
    "github_repo": project,
    "github_version": "master/doc/",
}

# A list of CSS files. The entry must be a filename string or a tuple containing the
# filename string and the attributes dictionary. The filename must be relative to the
# html_static_path, or a full URI with scheme like https://example.org/style.css.
# The attributes is used for attributes of <link> tag. It defaults to an empty list.
html_css_files = ["css/custom.css"]

smartquotes_action = "qe"  # renders only quotes and ellipses (...) but not dashes (option: D)

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    "**": [
        "relations.html",  # needs 'show_related': True theme option to display
        "searchbox.html",
    ]
}

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "xibs_doc"

# -- Options for LaTeX output ---------------------------------------------

# The paper size ('letter' or 'a4').
latex_paper_size = "letter"

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "xibs.tex", "xibs Documentation", "F. Soubelet", "manual"),
]

# Use Unicode aware LaTeX engine
latex_engine = "xelatex"  # or 'lualatex'

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = None

latex_elements = {}

# Keep babel usage also with xelatex (Sphinx default is polyglossia)
# If this key is removed or changed, latex build directory must be cleaned
latex_elements["babel"] = r"\usepackage{babel}"

# Font configuration
# Fix fontspec converting " into right curly quotes in PDF
# cf https://github.com/sphinx-doc/sphinx/pull/6888/
latex_elements[
    "fontenc"
] = r"""
\usepackage{fontspec}
\defaultfontfeatures[\rmfamily,\sffamily,\ttfamily]{}
"""

# Sphinx 2.0 adopts GNU FreeFont by default, but it does not have all
# the Unicode codepoints needed for the section about Mathtext
# "Writing mathematical expressions"
fontpkg = r"""
\IfFontExistsTF{XITS}{
 \setmainfont{XITS}
}{
 \setmainfont{XITS}[
  Extension      = .otf,
  UprightFont    = *-Regular,
  ItalicFont     = *-Italic,
  BoldFont       = *-Bold,
  BoldItalicFont = *-BoldItalic,
]}
\IfFontExistsTF{FreeSans}{
 \setsansfont{FreeSans}
}{
 \setsansfont{FreeSans}[
  Extension      = .otf,
  UprightFont    = *,
  ItalicFont     = *Oblique,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldOblique,
]}
\IfFontExistsTF{FreeMono}{
 \setmonofont{FreeMono}
}{
 \setmonofont{FreeMono}[
  Extension      = .otf,
  UprightFont    = *,
  ItalicFont     = *Oblique,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldOblique,
]}
% needed for \mathbb (blackboard alphabet) to actually work
\usepackage{unicode-math}
\IfFontExistsTF{XITS Math}{
 \setmathfont{XITS Math}
}{
 \setmathfont{XITSMath-Regular}[
  Extension      = .otf,
]}
"""
latex_elements["fontpkg"] = fontpkg


# Additional stuff for the LaTeX preamble.
latex_elements[
    "preamble"
] = r"""
   % Show Parts and Chapters in Table of Contents
   \setcounter{tocdepth}{0}
   % One line per author on title page
   \DeclareRobustCommand{\and}%
     {\end{tabular}\kern-\tabcolsep\\\begin{tabular}[t]{c}}%
   \usepackage{etoolbox}
   \AtBeginEnvironment{sphinxthebibliography}{\appendix\part{Appendices}}
   \usepackage{expdlist}
   \let\latexdescription=\description
   \def\description{\latexdescription{}{} \breaklabel}
   % But expdlist old LaTeX package requires fixes:
   % 1) remove extra space
   \makeatletter
   \patchcmd\@item{{\@breaklabel} }{{\@breaklabel}}{}{}
   \makeatother
   % 2) fix bug in expdlist's way of breaking the line after long item label
   \makeatletter
   \def\breaklabel{%
       \def\@breaklabel{%
           \leavevmode\par
           % now a hack because Sphinx inserts \leavevmode after term node
           \def\leavevmode{\def\leavevmode{\unhbox\voidb@x}}%
      }%
   }
   \makeatother
"""

# Sphinx 1.5 provides this to avoid "too deeply nested" LaTeX error
# and usage of "enumitem" LaTeX package is unneeded.
# Value can be increased but do not set it to something such as 2048
# which needlessly would trigger creation of thousands of TeX macros
latex_elements["maxlistdepth"] = "10"
latex_elements["pointsize"] = "11pt"

# Better looking general index in PDF
latex_elements["printindex"] = r"\footnotesize\raggedright\printindex"

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = True

latex_toplevel_sectioning = "part"

# If false, no module index is generated.
# latex_domain_indices = True

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "xibs", "xibs Documentation", [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "xibs",
        "xibs Documentation",
        author,
        "F. Soubelet",
        "Intra-Beam Scattering modeling prototype.",
        "Miscellaneous",
    ),
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False

# -- Autodoc Configuration ---------------------------------------------------

# Add here all modules to be mocked up. When the dependencies are not met at building time. Here used to have PyQT mocked.
autodoc_mock_imports = [
    "matplotlib.backends.backend_qt5agg",
]

# -- Instersphinx Configuration ----------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
# use in refs e.g:
# :ref:`comparison manual <python:comparisons>`
intersphinx_mapping = {
    "cpymad": ("https://hibtc.github.io/cpymad/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "tfs": ("https://pylhc.github.io/tfs/", None),
    "tfs-pandas": ("https://pylhc.github.io/tfs/", None),
}
