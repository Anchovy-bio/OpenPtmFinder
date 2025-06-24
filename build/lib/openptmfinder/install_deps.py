def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import subprocess
    import sys
    import csv
    import statistics
    import numpy as np
    import pandas as pd
    import argparse
    import logging
    import os
    import re
    import json
    import plotly
    import configparser
    from xml.etree import ElementTree as ET
    from pyteomics import pepxml, mzml, fasta
    from deeplc import DeepLC, FeatExtractor
    from scipy import stats as scipy_stats
except ImportError as e:
    print(f"Необходимые библиотеки не установлены.  Устанавливаю из requirements.txt...")
    print("Ошибка импорта:", e)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "/home/kopeykina/search_modification_project/requirements.txt"])
    print("Библиотеки установлены. Пожалуйста, перезапустите программу.")
    sys.exit(1)