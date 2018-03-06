How to build and upload to PyPI
===============================

A small memo about packaging and deployment.

Some resources:
 - https://packaging.python.org/tutorials/distributing-packages/
 - https://packaging.python.org/guides/using-testpypi/
 - https://packaging.python.org/guides/migrating-to-pypi-org/
 


Preparation
-----------
First clean output directories (if any):

    rm build/ dist/ *.egg-info/ -rf


Create a virtual environment for packaging, with appropriate tools:

   mkvirtualenv python_packaging
   pip install wheel twine



Build the package
-----------------
Package the sources:

    python setup.py sdist


Build a universal (Python 2 & 3) wheel:

    python setup.py bdist_wheel --universal


There now should be a new `dist/` directory (among others) with the following files
(version information may change):
 - `smartdoc15_ch1-0.1-py2.py3-none-any.whl`: wheel binary
 - `smartdoc15_ch1-0.1.tar.gz`: sources



Test upload to testpypi
-----------------------

 - Make sure you have created an account on https://test.pypi.org/.
 - Make sure you have activated your account: logged in, email ok, etc.
 - Opt. configure a `$HOME/.pypirc` file:
    [distutils]
    index-servers=
       pypi
       pypitest

    [pypitest]
    repository = https://test.pypi.org/legacy/
    username = jchazalon
    password = ****

    [pypi]
    repository = https://upload.pypi.org/legacy/
    username = jchazalon
    password = ****

 - Upload (the project will be automatically registered):
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    # or, using the ~/.pypirc file:
    # twine upload --repository pypitest dist/*
 - Check the webpage: https://test.pypi.org/project/smartdoc15-ch1/
 - Test install:
    deactivate
    rmvirtualenv test_install_sd15ch1
    mkvirtualenv test_install_sd15ch1
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple smartdoc15-ch1



