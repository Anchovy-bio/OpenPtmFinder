from setuptools import setup, find_packages

setup(
    name="OpenPtmFinder",
    version="1.0",
    author="Annyshka",
    description="PTM Annotation Tool Based on Open Strategy Search",
    packages=find_packages(),
    install_requires=[
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
    entry_points={
        "console_scripts": [
            "OpenPtmFinder=openptmfinder.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "openptmfinder": ["templates/*.html", "data/*.tsv","config.ini"],  # Путь к вложенным данным
    },
    python_requires=">=3.7",
)

