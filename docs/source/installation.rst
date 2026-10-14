Installation
============

SMPy needs Python 3.10 or newer. Install it from a clone of the repository:

.. code-block:: bash

   git clone https://github.com/GeorgeVassilakis/SMPy.git
   cd SMPy
   pip install .

pip also installs SMPy's dependencies: NumPy, SciPy, Astropy, pandas,
Matplotlib and PyYAML. To edit the code in place, use ``pip install -e .``
instead.

To check the installation, print the version and the command-line help:

.. code-block:: bash

   python -c "import smpy; print(smpy.__version__)"
   smpy --help

Running the tests
-----------------

.. code-block:: bash

   pip install ".[test]"
   pytest

Building the documentation
--------------------------

Building the documentation also runs the tutorials, which needs
`pandoc <https://pandoc.org/installing.html>`_:

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs/source docs/_build/html
