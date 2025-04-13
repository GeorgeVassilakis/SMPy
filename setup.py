from setuptools import setup, find_packages

# Read the long description from the README file
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='SMPy',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'astropy>=5.0.0',
        'pandas>=1.2.0',
        'matplotlib>=3.4.0',
        'scipy>=1.6.0',
        'pyyaml>=5.1'
    ],
    python_requires='>=3.8',
    author='Georgios N. Vassilakis',
    author_email='vassilakis.g@northeastern.edu',
    description='Shear Mapping in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GeorgeVassilakis/SMPy.git',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    license='MIT',
)