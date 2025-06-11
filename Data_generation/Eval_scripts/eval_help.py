import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import os
import math

def load_df(name,tag):
    parent_dir = os.path.dirname(os.getcwd())
    # Path to the Density_profiles directory
    density_profiles_dir = os.path.join(parent_dir, "Density_profiles")

    name = os.path.join(parent_dir, "Density_profiles",tag, name)
    print("Reading file: ", name)
    df = pd.read_csv(name,delimiter = " ")

    extra=[]
    try:
        nperiod = (list(df.columns)[4])
        df.drop(columns=[nperiod],inplace=True)
        nperiod = int(nperiod)
        extra.append(nperiod)
    except:
        print("No nperiod")
    try:
        mu = (list(df.columns)[4])

        df.drop(columns=[mu],inplace=True)
        mu = float(mu)
        extra.append(mu)
    except:
        print("No mu")
    try:
        packing = (list(df.columns)[4])
        df.drop(columns=[packing],inplace=True)
        packing = float(packing)
        extra.append(packing)
    except:
        print("No packing")
    try:
        packing = (list(df.columns)[4])
        df.drop(columns=[packing],inplace=True)
        packing = float(packing)
        extra.append(packing)
    except:
        print("No amp")

    return df, extra
