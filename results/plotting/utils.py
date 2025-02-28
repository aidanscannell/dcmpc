import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 400
plt.rcParams["font.size"] = 28
plt.rcParams["legend.fontsize"] = 29
plt.rcParams["legend.loc"] = "lower right"
plt.rcParams["text.usetex"] = True

AGENTS = {
    "DCWM": "DC-MPC (ours)",
    "TD-MPC2": "TD-MPC2",
    "DreamerV3": "DreamerV3",
    "SAC": "SAC",
    "TD-MPC": "TD-MPC",
}
PALETTE = {
    "DCWM (ours)": "#984ea3",
    "DC-MPC (ours)": "#984ea3",
    "DCWM-ref (ours)": "black",
    "TD-MPC2": "#377eb8",
    "DreamerV3": "#e41a1c",
    "SAC": "#ff7f00",
    "TD-MPC": "green",
}
YLABELS = {
    "episode_reward": "Episode Return",
    "episode_success": "Success Rate ($\%$)",
    "active_percent": "Codebook Active Percent (\%)",
    "rank1": "Matrix Rank",
    "rank_percent_1": "Matrix Rank (\% of full)",
}