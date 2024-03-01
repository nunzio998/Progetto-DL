import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


dataset = pd.read_excel("Dataset.xlsx", sheet_name='Weather')

print(dataset.head())



