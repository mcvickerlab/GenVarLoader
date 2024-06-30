# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import genvarloader

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GenVarLoader"
copyright = "2024, David Laub"
author = "David Laub"
release = genvarloader.__version__
# short X.Y verison
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

napoleon_type_aliases = {"ArrayLike": ":term:`array_like`", "NDArray": "ndarray"}
napoleon_use_rtype = True

autodoc_typehints = "both"
autodoc_type_aliases = {"ArrayLike": "ArrayLike"}
autodoc_default_options = {"private-members": False}
autodoc_member_order = "bysource"

myst_enable_extensions = ["colon_fence", "html_image"]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "polars": ("https://docs.pola.rs/py-polars/html", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/mcvickerlab/GenVarLoader",
    "use_repository_button": True,
}
html_logo = "_static/gvl_logo.png"
html_title = f"GenVarLoader v{version}"
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
    ]
}
