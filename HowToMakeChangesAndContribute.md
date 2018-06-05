How to make changes and contribute
==================================

A small memo about Python package development.

The recommended process described here is based on two things:
1. the use of virtual environments;
2. the installation of packages in a *editable* mode.

Setting up virtual environments
-------------------------------
We recommend to use [Virtualenv wrapper](http://virtualenvwrapper.readthedocs.org/) to set up a virtual environment you can safely work on.
Once Virtualenv wrapper is properly set up, you can create a test environment using:
    mkvirtualenv --system-site-packages sd15ch1_dev

Installing smartdoc15_ch1 in editable mode
------------------------------------------
You should clone the code repository and install the package from its sources.
    cd MY_WORKSPACE_DIRECTORY
    git clone https://github.com/jchazalon/smartdoc15-ch1-pywrapper.git 
    cd smartdoc15-ch1-pywrapper
    pip install -e .

The changes you will make on the current sources will be available immediately to new Python program using the `sd15ch1_dev` virtual environment.


Contributing changes
--------------------

You should fork the source repository and work on your personal branch, then produce pull requests with well isolated features or fixes.
Well isolated features or fixes have a clear purpose, contain only the strictly necessary code, and should be easy to test.
You should provide the tests showing that your pull request is working.

By contributing, you accept that the code you produce is licensed under the same terms as the existing package.
