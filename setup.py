"""
uploading to pypi is delayed until API is stable
https://packaging.python.org/tutorials/distributing-packages/#setup-py
#python setup.py bdist_wheel --universal
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'Readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='FenicsSolver',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.0',

    description='A multi-physics FEA solver based on Fenics',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/qingfengxia/FenicsSolver',

    # Author details
    author='Qingfeng Xia',
    author_email='qingfeng.xia@eng-ox-ac-uk',

    # Choose your license
    license='LGPL',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science\Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Environment :: Console'

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords=['Dolfin', 'Fenics', 'FEA', 'FEM', 'CFD'],

    # You can just specify the packages manually here if your project is simple. Or you can use find_packages().
    # there must be a subfolder in the git repo root
    packages=find_packages(where='.', exclude=['contrib', 'docs', 'tests']),
    # Alternatively, if you want to distribute just a my_module.py, uncomment this:

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    #install_requires=['fenics', 'numpy', 'matplotlib'],  # let user to install a proper Fenics version
    install_requires=[],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': [],  # dolfin requires a lot of packages, this package does not introduce any new dependencies
        'test': ['unittest'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        #'sample': ['package_data.dat'],     # todo later for testing data
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[('readme', ['Readme.md', 'FenicsSolver_FreeCAD.png'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'main=main.py',
        ],
    },
)
