#!/usr/bin/env python3
from os import listdir
from os.path import join

import pandas as pd


data = []
path = "./dreamerv3/"
for filename in listdir(path):  # iterates over all the files in 'path'
    print(f"{filename}")
    # breakpoint()
    env_name = filename.split(".csv")[0]
    # if "mw-" in env_name:
    # env_name = env_name.split("mw-")[-1]
    full_path = join(path, filename)  # joins the path with the filename
    env_csv = pd.read_csv(full_path)
    env_csv["env"] = env_name
    env_csv["agent"] = "DreamerV3"
    try:
        env_csv["episode_reward"] = env_csv["reward"]
    except KeyError:
        print(f"No reward for {filename}")
    try:
        env_csv["episode_success"] = env_csv["success"]
    except KeyError:
        print(f"No success for {filename}")
    env_csv["env_step"] = env_csv["step"]
    data.append(env_csv)

data = pd.concat(data)
data.to_csv("./dreamerv3.csv")
