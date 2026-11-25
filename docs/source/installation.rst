Installation
============

SMPy needs Python 3.10 or newer. Install it from PyPI, where it is named
``smpy-lensing``; it is imported and run as ``smpy``:

.. code-block:: bash

   pip install smpy-lensing

pip also installs the CPU build of JAX and the dependencies for catalog
handling, plotting and FITS output.

On a compatible NVIDIA Linux host, install the JAX CUDA build for your
driver and Python version. For example, with a CUDA 13 compatible driver,
run the following command.

.. code-block:: bash

   pip install "jax[cuda13]"
   python -c "import jax; print(jax.devices())"

See the `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
for supported platforms and accelerator builds. Use ``JAX_PLATFORMS=cpu``
to run on CPU or ``CUDA_VISIBLE_DEVICES`` to select a GPU.

The tutorials use the example catalogs in the repository. To run them, or to
work on the code, install SMPy from a clone of the repository instead:

.. code-block:: bash

   git clone https://github.com/GeorgeVassilakis/SMPy.git
   cd SMPy
   pip install .

To edit the code in place, use ``pip install -e .``.

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
