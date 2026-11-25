Installation
============

SMPy needs Python 3.10 or newer. Install it from PyPI, where it is named
``smpy-lensing``; it is imported and run as ``smpy``:

.. code-block:: bash

   pip install smpy-lensing

pip also installs SMPy's dependencies: NumPy, SciPy, Astropy, pandas,
Matplotlib and PyYAML.

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

Optional JAX backend
--------------------

NumPy is the default backend. Importing ``smpy`` leaves JAX unloaded.
Install the ``jax`` extra to add the CPU build of JAX (version 0.4.38 or newer).

.. code-block:: bash

   pip install "smpy-lensing[jax]"

On a compatible NVIDIA Linux host, install the JAX CUDA build for your
driver and Python version. For example, with a CUDA 13 compatible driver,
install the build and check that JAX lists your GPU.

.. code-block:: bash

   pip install "jax[cuda13]"
   python -c "import jax; print(jax.devices())"

See the `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
for supported platforms and accelerator builds. Use ``JAX_PLATFORMS=cpu``
to run on CPU or ``CUDA_VISIBLE_DEVICES`` to select a GPU.

Running the tests
-----------------

.. code-block:: bash

   pip install ".[test]"
   pytest

The tests of the JAX backend are skipped when JAX is not installed. Install
``".[test,jax]"`` to run them too.

Building the documentation
--------------------------

Building the documentation also runs the tutorials, which needs
`pandoc <https://pandoc.org/installing.html>`_. The API pages import
``smpy.jax``, so the build needs the ``jax`` extra:

.. code-block:: bash

   pip install ".[jax]" -r docs/requirements.txt
   sphinx-build -b html docs/source docs/_build/html
