from setuptools import setup, find_packages
import os

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="OpenPtmFinder",
    version = get_version('version.py'),
    author = "Annyshka",
    author_email = 'kopeykina.00@bk.ru',
    python_requires = ">=3.7",
    description="PTM Annotation Tool Based on Open Strategy Search",
    long_description = read('README.MD'),
    long_description_content_type="text/markdown",
    license = 'License :: OSI Approved :: Apache Software License',
    packages = find_packages(),
    install_requires = [
        "deeplc==2.2.38",
        "tensorflow==2.11.0",
        "numpy==1.24.4",
        "pandas",
        "scipy",
        "pyteomics",
        "plotly",
        "statsmodels",
        "flask"
    ],
    entry_points = {"console_scripts": ["OpenPtmFinder=openptmfinder.main:main"]},
    include_package_data = True,
    package_data = {"openptmfinder": ["templates/*.html", "data/*.tsv","config.ini"]},
    
    classifiers = ['Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   'Topic :: Scientific/Engineering :: Chemistry',
                   'Topic :: Scientific/Engineering :: Physics'],
)

