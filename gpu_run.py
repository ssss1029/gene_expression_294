# -*- coding: utf-8 -*-

"""
Given a bunch of commands to run, check the available GPUs and run them on the GPUs in separate tmux sessions.
Usage: Just modify the settings in the Config class and then run python3 gpu_run.py
"""

import GPUtil
import subprocess
import sys
import time

class Config:
    """
    Global class that houses all configurations
    """

    # Shared args to put onto all of the JOBS
    SHARED_ARGS = "--lr=0.01"

    # Specifies tasks to run. It maps tmux session name to the command to run in that session.
    JOBS = {
        "E003_FC" : "python main.py --globstr-train='./dataset/E003/classification/train.csv' --globstr-val='./dataset/E003/classification/valid.csv' --globstr-test='./dataset/E003/classification/test.csv' --save=checkpoints/original/E003",
        "E004_FC" : "python main.py --globstr-train='./dataset/E004/classification/train.csv' --globstr-val='./dataset/E004/classification/valid.csv' --globstr-test='./dataset/E004/classification/test.csv' --save=checkpoints/original/E004",
        "E005_FC" : "python main.py --globstr-train='./dataset/E005/classification/train.csv' --globstr-val='./dataset/E005/classification/valid.csv' --globstr-test='./dataset/E005/classification/test.csv' --save=checkpoints/original/E005",
        "E006_FC" : "python main.py --globstr-train='./dataset/E006/classification/train.csv' --globstr-val='./dataset/E006/classification/valid.csv' --globstr-test='./dataset/E006/classification/test.csv' --save=checkpoints/original/E006",
        "E007_FC" : "python main.py --globstr-train='./dataset/E007/classification/train.csv' --globstr-val='./dataset/E007/classification/valid.csv' --globstr-test='./dataset/E007/classification/test.csv' --save=checkpoints/original/E007",
        "E011_FC" : "python main.py --globstr-train='./dataset/E011/classification/train.csv' --globstr-val='./dataset/E011/classification/valid.csv' --globstr-test='./dataset/E011/classification/test.csv' --save=checkpoints/original/E011",
        "E012_FC" : "python main.py --globstr-train='./dataset/E012/classification/train.csv' --globstr-val='./dataset/E012/classification/valid.csv' --globstr-test='./dataset/E012/classification/test.csv' --save=checkpoints/original/E012",
        "E013_FC" : "python main.py --globstr-train='./dataset/E013/classification/train.csv' --globstr-val='./dataset/E013/classification/valid.csv' --globstr-test='./dataset/E013/classification/test.csv' --save=checkpoints/original/E013",
        "E016_FC" : "python main.py --globstr-train='./dataset/E016/classification/train.csv' --globstr-val='./dataset/E016/classification/valid.csv' --globstr-test='./dataset/E016/classification/test.csv' --save=checkpoints/original/E016",
        "E024_FC" : "python main.py --globstr-train='./dataset/E024/classification/train.csv' --globstr-val='./dataset/E024/classification/valid.csv' --globstr-test='./dataset/E024/classification/test.csv' --save=checkpoints/original/E024",
        "E027_FC" : "python main.py --globstr-train='./dataset/E027/classification/train.csv' --globstr-val='./dataset/E027/classification/valid.csv' --globstr-test='./dataset/E027/classification/test.csv' --save=checkpoints/original/E027",
        "E028_FC" : "python main.py --globstr-train='./dataset/E028/classification/train.csv' --globstr-val='./dataset/E028/classification/valid.csv' --globstr-test='./dataset/E028/classification/test.csv' --save=checkpoints/original/E028",
        "E037_FC" : "python main.py --globstr-train='./dataset/E037/classification/train.csv' --globstr-val='./dataset/E037/classification/valid.csv' --globstr-test='./dataset/E037/classification/test.csv' --save=checkpoints/original/E037",
        "E038_FC" : "python main.py --globstr-train='./dataset/E038/classification/train.csv' --globstr-val='./dataset/E038/classification/valid.csv' --globstr-test='./dataset/E038/classification/test.csv' --save=checkpoints/original/E038",
        "E047_FC" : "python main.py --globstr-train='./dataset/E047/classification/train.csv' --globstr-val='./dataset/E047/classification/valid.csv' --globstr-test='./dataset/E047/classification/test.csv' --save=checkpoints/original/E047",
        "E050_FC" : "python main.py --globstr-train='./dataset/E050/classification/train.csv' --globstr-val='./dataset/E050/classification/valid.csv' --globstr-test='./dataset/E050/classification/test.csv' --save=checkpoints/original/E050",
        "E053_FC" : "python main.py --globstr-train='./dataset/E053/classification/train.csv' --globstr-val='./dataset/E053/classification/valid.csv' --globstr-test='./dataset/E053/classification/test.csv' --save=checkpoints/original/E053",
        "E054_FC" : "python main.py --globstr-train='./dataset/E054/classification/train.csv' --globstr-val='./dataset/E054/classification/valid.csv' --globstr-test='./dataset/E054/classification/test.csv' --save=checkpoints/original/E054",
        "E055_FC" : "python main.py --globstr-train='./dataset/E055/classification/train.csv' --globstr-val='./dataset/E055/classification/valid.csv' --globstr-test='./dataset/E055/classification/test.csv' --save=checkpoints/original/E055",
        "E056_FC" : "python main.py --globstr-train='./dataset/E056/classification/train.csv' --globstr-val='./dataset/E056/classification/valid.csv' --globstr-test='./dataset/E056/classification/test.csv' --save=checkpoints/original/E056",
        "E057_FC" : "python main.py --globstr-train='./dataset/E057/classification/train.csv' --globstr-val='./dataset/E057/classification/valid.csv' --globstr-test='./dataset/E057/classification/test.csv' --save=checkpoints/original/E057",
        "E058_FC" : "python main.py --globstr-train='./dataset/E058/classification/train.csv' --globstr-val='./dataset/E058/classification/valid.csv' --globstr-test='./dataset/E058/classification/test.csv' --save=checkpoints/original/E058",
        "E059_FC" : "python main.py --globstr-train='./dataset/E059/classification/train.csv' --globstr-val='./dataset/E059/classification/valid.csv' --globstr-test='./dataset/E059/classification/test.csv' --save=checkpoints/original/E059",
        "E061_FC" : "python main.py --globstr-train='./dataset/E061/classification/train.csv' --globstr-val='./dataset/E061/classification/valid.csv' --globstr-test='./dataset/E061/classification/test.csv' --save=checkpoints/original/E061",
        "E062_FC" : "python main.py --globstr-train='./dataset/E062/classification/train.csv' --globstr-val='./dataset/E062/classification/valid.csv' --globstr-test='./dataset/E062/classification/test.csv' --save=checkpoints/original/E062",
        "E065_FC" : "python main.py --globstr-train='./dataset/E065/classification/train.csv' --globstr-val='./dataset/E065/classification/valid.csv' --globstr-test='./dataset/E065/classification/test.csv' --save=checkpoints/original/E065",
        "E066_FC" : "python main.py --globstr-train='./dataset/E066/classification/train.csv' --globstr-val='./dataset/E066/classification/valid.csv' --globstr-test='./dataset/E066/classification/test.csv' --save=checkpoints/original/E066",
        "E070_FC" : "python main.py --globstr-train='./dataset/E070/classification/train.csv' --globstr-val='./dataset/E070/classification/valid.csv' --globstr-test='./dataset/E070/classification/test.csv' --save=checkpoints/original/E070",
        "E071_FC" : "python main.py --globstr-train='./dataset/E071/classification/train.csv' --globstr-val='./dataset/E071/classification/valid.csv' --globstr-test='./dataset/E071/classification/test.csv' --save=checkpoints/original/E071",
        "E079_FC" : "python main.py --globstr-train='./dataset/E079/classification/train.csv' --globstr-val='./dataset/E079/classification/valid.csv' --globstr-test='./dataset/E079/classification/test.csv' --save=checkpoints/original/E079",
        "E082_FC" : "python main.py --globstr-train='./dataset/E082/classification/train.csv' --globstr-val='./dataset/E082/classification/valid.csv' --globstr-test='./dataset/E082/classification/test.csv' --save=checkpoints/original/E082",
        "E084_FC" : "python main.py --globstr-train='./dataset/E084/classification/train.csv' --globstr-val='./dataset/E084/classification/valid.csv' --globstr-test='./dataset/E084/classification/test.csv' --save=checkpoints/original/E084",
        "E085_FC" : "python main.py --globstr-train='./dataset/E085/classification/train.csv' --globstr-val='./dataset/E085/classification/valid.csv' --globstr-test='./dataset/E085/classification/test.csv' --save=checkpoints/original/E085",
        "E087_FC" : "python main.py --globstr-train='./dataset/E087/classification/train.csv' --globstr-val='./dataset/E087/classification/valid.csv' --globstr-test='./dataset/E087/classification/test.csv' --save=checkpoints/original/E087",
        "E094_FC" : "python main.py --globstr-train='./dataset/E094/classification/train.csv' --globstr-val='./dataset/E094/classification/valid.csv' --globstr-test='./dataset/E094/classification/test.csv' --save=checkpoints/original/E094",
        "E095_FC" : "python main.py --globstr-train='./dataset/E095/classification/train.csv' --globstr-val='./dataset/E095/classification/valid.csv' --globstr-test='./dataset/E095/classification/test.csv' --save=checkpoints/original/E095",
        "E096_FC" : "python main.py --globstr-train='./dataset/E096/classification/train.csv' --globstr-val='./dataset/E096/classification/valid.csv' --globstr-test='./dataset/E096/classification/test.csv' --save=checkpoints/original/E096",
        "E097_FC" : "python main.py --globstr-train='./dataset/E097/classification/train.csv' --globstr-val='./dataset/E097/classification/valid.csv' --globstr-test='./dataset/E097/classification/test.csv' --save=checkpoints/original/E097",
        "E098_FC" : "python main.py --globstr-train='./dataset/E098/classification/train.csv' --globstr-val='./dataset/E098/classification/valid.csv' --globstr-test='./dataset/E098/classification/test.csv' --save=checkpoints/original/E098",
        "E100_FC" : "python main.py --globstr-train='./dataset/E100/classification/train.csv' --globstr-val='./dataset/E100/classification/valid.csv' --globstr-test='./dataset/E100/classification/test.csv' --save=checkpoints/original/E100",
        "E104_FC" : "python main.py --globstr-train='./dataset/E104/classification/train.csv' --globstr-val='./dataset/E104/classification/valid.csv' --globstr-test='./dataset/E104/classification/test.csv' --save=checkpoints/original/E104",
        "E105_FC" : "python main.py --globstr-train='./dataset/E105/classification/train.csv' --globstr-val='./dataset/E105/classification/valid.csv' --globstr-test='./dataset/E105/classification/test.csv' --save=checkpoints/original/E105",
        "E106_FC" : "python main.py --globstr-train='./dataset/E106/classification/train.csv' --globstr-val='./dataset/E106/classification/valid.csv' --globstr-test='./dataset/E106/classification/test.csv' --save=checkpoints/original/E106",
        "E109_FC" : "python main.py --globstr-train='./dataset/E109/classification/train.csv' --globstr-val='./dataset/E109/classification/valid.csv' --globstr-test='./dataset/E109/classification/test.csv' --save=checkpoints/original/E109",
        "E112_FC" : "python main.py --globstr-train='./dataset/E112/classification/train.csv' --globstr-val='./dataset/E112/classification/valid.csv' --globstr-test='./dataset/E112/classification/test.csv' --save=checkpoints/original/E112",
        "E113_FC" : "python main.py --globstr-train='./dataset/E113/classification/train.csv' --globstr-val='./dataset/E113/classification/valid.csv' --globstr-test='./dataset/E113/classification/test.csv' --save=checkpoints/original/E113",
        "E114_FC" : "python main.py --globstr-train='./dataset/E114/classification/train.csv' --globstr-val='./dataset/E114/classification/valid.csv' --globstr-test='./dataset/E114/classification/test.csv' --save=checkpoints/original/E114",
        "E116_FC" : "python main.py --globstr-train='./dataset/E116/classification/train.csv' --globstr-val='./dataset/E116/classification/valid.csv' --globstr-test='./dataset/E116/classification/test.csv' --save=checkpoints/original/E116",
        "E117_FC" : "python main.py --globstr-train='./dataset/E117/classification/train.csv' --globstr-val='./dataset/E117/classification/valid.csv' --globstr-test='./dataset/E117/classification/test.csv' --save=checkpoints/original/E117",
        "E118_FC" : "python main.py --globstr-train='./dataset/E118/classification/train.csv' --globstr-val='./dataset/E118/classification/valid.csv' --globstr-test='./dataset/E118/classification/test.csv' --save=checkpoints/original/E118",
        "E119_FC" : "python main.py --globstr-train='./dataset/E119/classification/train.csv' --globstr-val='./dataset/E119/classification/valid.csv' --globstr-test='./dataset/E119/classification/test.csv' --save=checkpoints/original/E119",
        "E120_FC" : "python main.py --globstr-train='./dataset/E120/classification/train.csv' --globstr-val='./dataset/E120/classification/valid.csv' --globstr-test='./dataset/E120/classification/test.csv' --save=checkpoints/original/E120",
        "E122_FC" : "python main.py --globstr-train='./dataset/E122/classification/train.csv' --globstr-val='./dataset/E122/classification/valid.csv' --globstr-test='./dataset/E122/classification/test.csv' --save=checkpoints/original/E122",
        "E123_FC" : "python main.py --globstr-train='./dataset/E123/classification/train.csv' --globstr-val='./dataset/E123/classification/valid.csv' --globstr-test='./dataset/E123/classification/test.csv' --save=checkpoints/original/E123",
        "E127_FC" : "python main.py --globstr-train='./dataset/E127/classification/train.csv' --globstr-val='./dataset/E127/classification/valid.csv' --globstr-test='./dataset/E127/classification/test.csv' --save=checkpoints/original/E127",
        "E128_FC" : "python main.py --globstr-train='./dataset/E128/classification/train.csv' --globstr-val='./dataset/E128/classification/valid.csv' --globstr-test='./dataset/E128/classification/test.csv' --save=checkpoints/original/E128",
    }

    # tmux kill-session -t E003_original
    # tmux kill-session -t E004_original
    # tmux kill-session -t E005_original
    # tmux kill-session -t E006_original
    # tmux kill-session -t E007_original
    # tmux kill-session -t E011_original
    # tmux kill-session -t E012_original
    # tmux kill-session -t E013_original
    # tmux kill-session -t E016_original
    # tmux kill-session -t E024_original
    # tmux kill-session -t E027_original
    # tmux kill-session -t E028_original
    # tmux kill-session -t E037_original
    # tmux kill-session -t E038_original
    # tmux kill-session -t E047_original
    # tmux kill-session -t E050_original
    # tmux kill-session -t E053_original
    # tmux kill-session -t E054_original
    # tmux kill-session -t E055_original
    # tmux kill-session -t E056_original
    # tmux kill-session -t E057_original
    # tmux kill-session -t E058_original
    # tmux kill-session -t E059_original
    # tmux kill-session -t E061_original
    # tmux kill-session -t E062_original
    # tmux kill-session -t E065_original
    # tmux kill-session -t E066_original
    # tmux kill-session -t E070_original
    # tmux kill-session -t E071_original
    # tmux kill-session -t E079_original
    # tmux kill-session -t E082_original
    # tmux kill-session -t E084_original
    # tmux kill-session -t E085_original
    # tmux kill-session -t E087_original
    # tmux kill-session -t E094_original
    # tmux kill-session -t E095_original
    # tmux kill-session -t E096_original
    # tmux kill-session -t E097_original
    # tmux kill-session -t E098_original
    # tmux kill-session -t E100_original
    # tmux kill-session -t E104_original
    # tmux kill-session -t E105_original
    # tmux kill-session -t E106_original
    # tmux kill-session -t E109_original
    # tmux kill-session -t E112_original
    # tmux kill-session -t E113_original
    # tmux kill-session -t E114_original
    # tmux kill-session -t E116_original
    # tmux kill-session -t E117_original
    # tmux kill-session -t E118_original
    # tmux kill-session -t E119_original
    # tmux kill-session -t E120_original
    # tmux kill-session -t E122_original
    # tmux kill-session -t E123_original
    # tmux kill-session -t E127_original
    # tmux kill-session -t E128_original

    # Time to wait between putting jobs on GPUs (in seconds). This is useful because it might take time
    # for a process to actually load the network onto the GPU, so we wait until that is done before
    # selecting the GPU for the next process.
    SLEEP_TIME = 20

    # Minimum memory required on a GPU to consider putting a job on it (MiB).
    MIN_MEMORY_REQUIRED = 2000


# Stick the shared args onto each JOB
for key, value in Config.JOBS.items():
    new_value = value + " " + Config.SHARED_ARGS
    Config.JOBS[key] = new_value

def select_gpu(GPUs):
    """
    Select the next best available GPU to run on. If nothing exists, return None
    """
    GPUs = list(filter(lambda gpu: gpu.memoryFree > Config.MIN_MEMORY_REQUIRED, GPUs))
    if len(GPUs) == 0:
        return None
    GPUs = sorted(GPUs, key=lambda gpu: gpu.memoryFree)
    return GPUs[-1]

for index, (tmux_session_name, command) in enumerate(Config.JOBS.items()):
    # Get the best available GPU
    print("Finding GPU for command \"{0}\"".format(command))
    curr_gpu = select_gpu(GPUtil.getGPUs())

    if curr_gpu == None:
        print("No available GPUs found. Exiting.")
        sys.exit(1)

    print("SUCCESS! Found GPU id = {0} which has {1} MiB free memory".format(curr_gpu.id, curr_gpu.memoryFree))

    result = subprocess.run("tmux new-session -d -s {0}".format(tmux_session_name), shell=True)
    if result.returncode != 0:
        print("Failed to create new tmux session called {0}".format(tmux_session_name))
        sys.exit(result.returncode)

    result = subprocess.run("tmux send-keys 'CUDA_VISIBLE_DEVICES={0} {1}' C-m".format(
        curr_gpu.id, command
    ), shell=True)
    if result.returncode != 0:
        print("Failed to run {0} in tmux session".format(command, tmux_session_name))
        sys.exit(result.returncode)

    print("---------------------------------------------------------------")

    if index < len(Config.JOBS) - 1:
        time.sleep(Config.SLEEP_TIME)