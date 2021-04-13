.. _contributing:

*********************
Contributing to Loupe
*********************


Reporting bugs and requesting new features
==========================================
Bug reports and enhancement requests should be filed using Loupe's
`issue tracker <https://github.com/andykee/loupe/issues>`__

.. _contributing.source:

Working with the source code
============================

Version control, Git, and GitHub
--------------------------------
Loupe's source code is hosted on `GitHub <https://github.com/andykee/loupe>`_.
To contribute you'll need `an account <https://github.com/signup/free>`_. Local
version control is handled by `Git <https://git-scm.com/>`_.

`GitHub has instructions <https://help.github.com/set-up-git-redirect>`_ for
installing Git, setting up your SSH key, and configuring Git. All these steps
need to be completed before you can work seamlessly between your local repository
and GitHub.

Some useful resources for learning Git:

* `Github Docs <https://docs.github.com/en>`_
* Matthew Brett's `Pydagouge <http://matthew-brett.github.io/pydagogue/git.html>`_
  notes on Git
* `Oh Shit, Git? <https://ohshitgit.com>`_ for when things go horribly wrong

.. _contributing.forking:

Forking
-------
You will need your own fork to work on the code. Go to the Loupe GitHub page and
hit the ``Fork`` button. You will want to clone your fork to your machine::

    git clone https://github.com/your-user-name/loupe.git loupe-yourname
    cd loupe-yourname
    git remote add upstream https://github.com/andykee/loupe.git

This creates the directory `loupe-yourname` and connects your repository to the
upstream (main project) Loupe repository.

Creating a Python environment
-----------------------------
To test out code changes, you’ll need to build install from source, which
requires a suitable Python environment. To create an isolated Loupe development
environment:

* Install either `Anaconda <https://www.anaconda.com/download/>`_ or `miniconda
  <https://conda.io/miniconda.html>`_
* Make sure your conda is up to date (``conda update conda``)
* Make sure that you have :ref:`cloned the repository <contributing.forking>`
* ``cd`` to the Loupe source directory

We can now create a development environment and install Loupe::

    # Create and activate the build environment:
    conda env create -f environment.yml
    conda activate loupe-dev

    # or with older versions of Anaconda:
    source activate loupe-dev

    # Install Loupe and its dependencies
    python -m pip install -e . --no-build-isolation --no-use-pep517

You should now be able to import Loupe in your development environment::

    $ python
    >>> import loupe

To view your environments::

    conda info -e

To return to your root environment::

    conda deactivate

See the full conda docs `here <https://conda.pydata.org/docs>`_.


Contribution workflow
=====================
The general contribution workflow should look something like `GitHub flow 
<https://guides.github.com/introduction/flow/>`_ but we're not particularly
picky about it.

.. _contributing.tests:

Running the tests
=================
.. note::

  Running the tests requires `pytest <https://docs.pytest.org/en/latest/>`_.

The tests can then be run directly inside your Git clone by typing::

    pytest tests


Building the docs
=================

.. note::

  Building the documentation requires `Sphinx <https://www.sphinx-doc.org/en/master/>`_
  and the `PyData Sphinx Theme
  <https://pydata-sphinx-theme.readthedocs.io/en/latest/index.html>`_.

To build the documentation, navigate to your local ``docs/`` directory and run::

    make html

The HTML documentation will be written to ``docs/_build/html``.

If you want to do a full clean build, do::

    make clean && make html