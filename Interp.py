import os
import pandas as pd
import torch

def interp1d(x, y):
    # Ensure x and y are 1-D tensors
    x = x.view(-1)
    y = y.view(-1)

    # Define a function that takes a value v as input and returns the
    # linearly interpolated value of y at x=v
    def f(v):
        # Find the index i such that x[i] <= v < x[i+1]
        i = torch.searchsorted(x, v)
        i = torch.clamp(i, 1, len(x)-1)
        
        # Compute the slope and intercept of the line connecting the
        # points (x[i], y[i]) and (x[i+1], y[i+1])
        slope = (y[i] - y[i-1]) / (x[i] - x[i-1])
        intercept = y[i-1] - slope * x[i-1]

        # Use the slope and intercept to compute the interpolated value
        return slope * v + intercept

    # Return the function f
    return f

BASE_PATH = "/root/autodl-tmp/kaggle/"
PATH = os.path.join(BASE_PATH, "icecube-neutrinos-in-deep-ice")
FILE_ICE_TRANS = os.path.join(PATH, "ice_transparency_info.csv")

def ice_transparency():
    df = pd.read_csv(FILE_ICE_TRANS)
    df["z"] = df["depth"] - 1950 # origin
    # df["scattering_len"] = (df["scattering_len"] - 32.4) / (46.575 - 19.4)
    # df["absorption_len"] = (df["absorption_len"] - 111.8) / (157.325 - 68.0)
    f_scattering = interp1d(
        torch.from_numpy(df["z"].values).float(), 
        torch.from_numpy(df["scattering_len"].values).float())
    f_absorption = interp1d(
        torch.from_numpy(df["z"].values).float(), 
        torch.from_numpy(df["absorption_len"].values).float())
    return f_scattering, f_absorption

f_sca, f_abs = ice_transparency()
print(f_sca(torch.tensor([1448.4 - 1950, 1448.4 - 1950, -1000, 2488.4 - 1950, 2480 - 1950, 1000])))
