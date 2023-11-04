# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 01:04:09 2023

@author: Mathilde & Diane
"""

import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as prep
import numpy as np
import dataViz as dv
import seaborn as sns


# Import
FILE_PATH = "data/Hotel_Reservations.csv"
DATASET = pd.read_csv(FILE_PATH, sep=',')

description = DATASET.describe()
description.to_markdown('description.md')

dv.canceledPopulation()
dv.canceledPercentageByDate()
