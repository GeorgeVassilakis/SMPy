from setuptools import setup, find_packages

setup(
    name='SMPy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'astropy', 'pandas', 'matplotlib'
    ],
    author='Georgios N. Vassilakis',
    author_email='vassilakis.g@northeastern.edu',
    description='From Shear to Map: A Python-based approach to constructing convergence maps.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/GeorgeVassilakis/SMPy.git',
)
