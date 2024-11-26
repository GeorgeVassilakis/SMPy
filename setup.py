from setuptools import setup, find_packages

# Read the long description from the README file
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='SMPy',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'astropy',
        'pandas',
        'matplotlib',
        'scipy',
        'pyyaml'
    ],
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    license='MIT',
)
