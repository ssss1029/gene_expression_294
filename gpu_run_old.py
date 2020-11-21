# -*- coding: utf-8 -*-

"""
Given a bunch of commands to run, check the available GPUs and run them on the GPUs in separate tmux sessions.
Usage: Just modify the settings in the Config class and then run python3 gpu_run.py
"""

import GPUtil
import subprocess
import sys
import time
import os

class Config:
    """
    Global class that houses all configurations
    """

    ALLOWED_GPUIDS = [1]

    # Shared args to put onto all of the JOBS
    SHARED_ARGS = ""

    # Specifies tasks to run. It maps tmux session name to the command to run in that session.
    JOBS = {
        "294_deepchrome_E104" : "python3 main.py  --globstr-train='./dataset/E104/classification/train.csv'  --globstr-val='./dataset/E104/classification/valid.csv' --globstr-test='./dataset/E104/classification/test.csv' --save=./checkpoints/deepchrome/E104",
        "294_deepchrome_E094" : "python3 main.py  --globstr-train='./dataset/E094/classification/train.csv'  --globstr-val='./dataset/E094/classification/valid.csv' --globstr-test='./dataset/E094/classification/test.csv' --save=./checkpoints/deepchrome/E094",
        "294_deepchrome_E098" : "python3 main.py  --globstr-train='./dataset/E098/classification/train.csv'  --globstr-val='./dataset/E098/classification/valid.csv' --globstr-test='./dataset/E098/classification/test.csv' --save=./checkpoints/deepchrome/E098",
        "294_deepchrome_E100" : "python3 main.py  --globstr-train='./dataset/E100/classification/train.csv'  --globstr-val='./dataset/E100/classification/valid.csv' --globstr-test='./dataset/E100/classification/test.csv' --save=./checkpoints/deepchrome/E100",
        "294_deepchrome_E085" : "python3 main.py  --globstr-train='./dataset/E085/classification/train.csv'  --globstr-val='./dataset/E085/classification/valid.csv' --globstr-test='./dataset/E085/classification/test.csv' --save=./checkpoints/deepchrome/E085",
        "294_deepchrome_E087" : "python3 main.py  --globstr-train='./dataset/E087/classification/train.csv'  --globstr-val='./dataset/E087/classification/valid.csv' --globstr-test='./dataset/E087/classification/test.csv' --save=./checkpoints/deepchrome/E087",
        "294_deepchrome_E097" : "python3 main.py  --globstr-train='./dataset/E097/classification/train.csv'  --globstr-val='./dataset/E097/classification/valid.csv' --globstr-test='./dataset/E097/classification/test.csv' --save=./checkpoints/deepchrome/E097",
        "294_deepchrome_E106" : "python3 main.py  --globstr-train='./dataset/E106/classification/train.csv'  --globstr-val='./dataset/E106/classification/valid.csv' --globstr-test='./dataset/E106/classification/test.csv' --save=./checkpoints/deepchrome/E106",
        "294_deepchrome_E065" : "python3 main.py  --globstr-train='./dataset/E065/classification/train.csv'  --globstr-val='./dataset/E065/classification/valid.csv' --globstr-test='./dataset/E065/classification/test.csv' --save=./checkpoints/deepchrome/E065",
        "294_deepchrome_E105" : "python3 main.py  --globstr-train='./dataset/E105/classification/train.csv'  --globstr-val='./dataset/E105/classification/valid.csv' --globstr-test='./dataset/E105/classification/test.csv' --save=./checkpoints/deepchrome/E105",
        "294_deepchrome_E066" : "python3 main.py  --globstr-train='./dataset/E066/classification/train.csv'  --globstr-val='./dataset/E066/classification/valid.csv' --globstr-test='./dataset/E066/classification/test.csv' --save=./checkpoints/deepchrome/E066",
        "294_deepchrome_E114" : "python3 main.py  --globstr-train='./dataset/E114/classification/train.csv'  --globstr-val='./dataset/E114/classification/valid.csv' --globstr-test='./dataset/E114/classification/test.csv' --save=./checkpoints/deepchrome/E114",
        "294_deepchrome_E116" : "python3 main.py  --globstr-train='./dataset/E116/classification/train.csv'  --globstr-val='./dataset/E116/classification/valid.csv' --globstr-test='./dataset/E116/classification/test.csv' --save=./checkpoints/deepchrome/E116",
        "294_deepchrome_E118" : "python3 main.py  --globstr-train='./dataset/E118/classification/train.csv'  --globstr-val='./dataset/E118/classification/valid.csv' --globstr-test='./dataset/E118/classification/test.csv' --save=./checkpoints/deepchrome/E118",
        "294_deepchrome_E119" : "python3 main.py  --globstr-train='./dataset/E119/classification/train.csv'  --globstr-val='./dataset/E119/classification/valid.csv' --globstr-test='./dataset/E119/classification/test.csv' --save=./checkpoints/deepchrome/E119",
        "294_deepchrome_E120" : "python3 main.py  --globstr-train='./dataset/E120/classification/train.csv'  --globstr-val='./dataset/E120/classification/valid.csv' --globstr-test='./dataset/E120/classification/test.csv' --save=./checkpoints/deepchrome/E120",
        "294_deepchrome_E122" : "python3 main.py  --globstr-train='./dataset/E122/classification/train.csv'  --globstr-val='./dataset/E122/classification/valid.csv' --globstr-test='./dataset/E122/classification/test.csv' --save=./checkpoints/deepchrome/E122",
        "294_deepchrome_E079" : "python3 main.py  --globstr-train='./dataset/E079/classification/train.csv'  --globstr-val='./dataset/E079/classification/valid.csv' --globstr-test='./dataset/E079/classification/test.csv' --save=./checkpoints/deepchrome/E079",
        "294_deepchrome_E123" : "python3 main.py  --globstr-train='./dataset/E123/classification/train.csv'  --globstr-val='./dataset/E123/classification/valid.csv' --globstr-test='./dataset/E123/classification/test.csv' --save=./checkpoints/deepchrome/E123",
        "294_deepchrome_E113" : "python3 main.py  --globstr-train='./dataset/E113/classification/train.csv'  --globstr-val='./dataset/E113/classification/valid.csv' --globstr-test='./dataset/E113/classification/test.csv' --save=./checkpoints/deepchrome/E113",
        "294_deepchrome_E003" : "python3 main.py  --globstr-train='./dataset/E003/classification/train.csv'  --globstr-val='./dataset/E003/classification/valid.csv' --globstr-test='./dataset/E003/classification/test.csv' --save=./checkpoints/deepchrome/E003",
        "294_deepchrome_E127" : "python3 main.py  --globstr-train='./dataset/E127/classification/train.csv'  --globstr-val='./dataset/E127/classification/valid.csv' --globstr-test='./dataset/E127/classification/test.csv' --save=./checkpoints/deepchrome/E127",
        "294_deepchrome_E027" : "python3 main.py  --globstr-train='./dataset/E027/classification/train.csv'  --globstr-val='./dataset/E027/classification/valid.csv' --globstr-test='./dataset/E027/classification/test.csv' --save=./checkpoints/deepchrome/E027",
        "294_deepchrome_E004" : "python3 main.py  --globstr-train='./dataset/E004/classification/train.csv'  --globstr-val='./dataset/E004/classification/valid.csv' --globstr-test='./dataset/E004/classification/test.csv' --save=./checkpoints/deepchrome/E004",
        "294_deepchrome_E006" : "python3 main.py  --globstr-train='./dataset/E006/classification/train.csv'  --globstr-val='./dataset/E006/classification/valid.csv' --globstr-test='./dataset/E006/classification/test.csv' --save=./checkpoints/deepchrome/E006",
        "294_deepchrome_E057" : "python3 main.py  --globstr-train='./dataset/E057/classification/train.csv'  --globstr-val='./dataset/E057/classification/valid.csv' --globstr-test='./dataset/E057/classification/test.csv' --save=./checkpoints/deepchrome/E057",
        "294_deepchrome_E056" : "python3 main.py  --globstr-train='./dataset/E056/classification/train.csv'  --globstr-val='./dataset/E056/classification/valid.csv' --globstr-test='./dataset/E056/classification/test.csv' --save=./checkpoints/deepchrome/E056",
        "294_deepchrome_E055" : "python3 main.py  --globstr-train='./dataset/E055/classification/train.csv'  --globstr-val='./dataset/E055/classification/valid.csv' --globstr-test='./dataset/E055/classification/test.csv' --save=./checkpoints/deepchrome/E055",
        "294_deepchrome_E007" : "python3 main.py  --globstr-train='./dataset/E007/classification/train.csv'  --globstr-val='./dataset/E007/classification/valid.csv' --globstr-test='./dataset/E007/classification/test.csv' --save=./checkpoints/deepchrome/E007",
        "294_deepchrome_E054" : "python3 main.py  --globstr-train='./dataset/E054/classification/train.csv'  --globstr-val='./dataset/E054/classification/valid.csv' --globstr-test='./dataset/E054/classification/test.csv' --save=./checkpoints/deepchrome/E054",
        "294_deepchrome_E058" : "python3 main.py  --globstr-train='./dataset/E058/classification/train.csv'  --globstr-val='./dataset/E058/classification/valid.csv' --globstr-test='./dataset/E058/classification/test.csv' --save=./checkpoints/deepchrome/E058",
        "294_deepchrome_E011" : "python3 main.py  --globstr-train='./dataset/E011/classification/train.csv'  --globstr-val='./dataset/E011/classification/valid.csv' --globstr-test='./dataset/E011/classification/test.csv' --save=./checkpoints/deepchrome/E011",
        "294_deepchrome_E053" : "python3 main.py  --globstr-train='./dataset/E053/classification/train.csv'  --globstr-val='./dataset/E053/classification/valid.csv' --globstr-test='./dataset/E053/classification/test.csv' --save=./checkpoints/deepchrome/E053",
        "294_deepchrome_E050" : "python3 main.py  --globstr-train='./dataset/E050/classification/train.csv'  --globstr-val='./dataset/E050/classification/valid.csv' --globstr-test='./dataset/E050/classification/test.csv' --save=./checkpoints/deepchrome/E050",
        "294_deepchrome_E012" : "python3 main.py  --globstr-train='./dataset/E012/classification/train.csv'  --globstr-val='./dataset/E012/classification/valid.csv' --globstr-test='./dataset/E012/classification/test.csv' --save=./checkpoints/deepchrome/E012",
        "294_deepchrome_E013" : "python3 main.py  --globstr-train='./dataset/E013/classification/train.csv'  --globstr-val='./dataset/E013/classification/valid.csv' --globstr-test='./dataset/E013/classification/test.csv' --save=./checkpoints/deepchrome/E013",
        "294_deepchrome_E016" : "python3 main.py  --globstr-train='./dataset/E016/classification/train.csv'  --globstr-val='./dataset/E016/classification/valid.csv' --globstr-test='./dataset/E016/classification/test.csv' --save=./checkpoints/deepchrome/E016",
        "294_deepchrome_E059" : "python3 main.py  --globstr-train='./dataset/E059/classification/train.csv'  --globstr-val='./dataset/E059/classification/valid.csv' --globstr-test='./dataset/E059/classification/test.csv' --save=./checkpoints/deepchrome/E059",
        "294_deepchrome_E024" : "python3 main.py  --globstr-train='./dataset/E024/classification/train.csv'  --globstr-val='./dataset/E024/classification/valid.csv' --globstr-test='./dataset/E024/classification/test.csv' --save=./checkpoints/deepchrome/E024",
        "294_deepchrome_E117" : "python3 main.py  --globstr-train='./dataset/E117/classification/train.csv'  --globstr-val='./dataset/E117/classification/valid.csv' --globstr-test='./dataset/E117/classification/test.csv' --save=./checkpoints/deepchrome/E117",
        "294_deepchrome_E070" : "python3 main.py  --globstr-train='./dataset/E070/classification/train.csv'  --globstr-val='./dataset/E070/classification/valid.csv' --globstr-test='./dataset/E070/classification/test.csv' --save=./checkpoints/deepchrome/E070",
        "294_deepchrome_E071" : "python3 main.py  --globstr-train='./dataset/E071/classification/train.csv'  --globstr-val='./dataset/E071/classification/valid.csv' --globstr-test='./dataset/E071/classification/test.csv' --save=./checkpoints/deepchrome/E071",
        "294_deepchrome_E038" : "python3 main.py  --globstr-train='./dataset/E038/classification/train.csv'  --globstr-val='./dataset/E038/classification/valid.csv' --globstr-test='./dataset/E038/classification/test.csv' --save=./checkpoints/deepchrome/E038",
        "294_deepchrome_E096" : "python3 main.py  --globstr-train='./dataset/E096/classification/train.csv'  --globstr-val='./dataset/E096/classification/valid.csv' --globstr-test='./dataset/E096/classification/test.csv' --save=./checkpoints/deepchrome/E096",
        "294_deepchrome_E109" : "python3 main.py  --globstr-train='./dataset/E109/classification/train.csv'  --globstr-val='./dataset/E109/classification/valid.csv' --globstr-test='./dataset/E109/classification/test.csv' --save=./checkpoints/deepchrome/E109",
        "294_deepchrome_E112" : "python3 main.py  --globstr-train='./dataset/E112/classification/train.csv'  --globstr-val='./dataset/E112/classification/valid.csv' --globstr-test='./dataset/E112/classification/test.csv' --save=./checkpoints/deepchrome/E112",
        "294_deepchrome_E095" : "python3 main.py  --globstr-train='./dataset/E095/classification/train.csv'  --globstr-val='./dataset/E095/classification/valid.csv' --globstr-test='./dataset/E095/classification/test.csv' --save=./checkpoints/deepchrome/E095",
        "294_deepchrome_E047" : "python3 main.py  --globstr-train='./dataset/E047/classification/train.csv'  --globstr-val='./dataset/E047/classification/valid.csv' --globstr-test='./dataset/E047/classification/test.csv' --save=./checkpoints/deepchrome/E047",
        "294_deepchrome_E084" : "python3 main.py  --globstr-train='./dataset/E084/classification/train.csv'  --globstr-val='./dataset/E084/classification/valid.csv' --globstr-test='./dataset/E084/classification/test.csv' --save=./checkpoints/deepchrome/E084",
        "294_deepchrome_E082" : "python3 main.py  --globstr-train='./dataset/E082/classification/train.csv'  --globstr-val='./dataset/E082/classification/valid.csv' --globstr-test='./dataset/E082/classification/test.csv' --save=./checkpoints/deepchrome/E082",
        "294_deepchrome_E062" : "python3 main.py  --globstr-train='./dataset/E062/classification/train.csv'  --globstr-val='./dataset/E062/classification/valid.csv' --globstr-test='./dataset/E062/classification/test.csv' --save=./checkpoints/deepchrome/E062",
        "294_deepchrome_E005" : "python3 main.py  --globstr-train='./dataset/E005/classification/train.csv'  --globstr-val='./dataset/E005/classification/valid.csv' --globstr-test='./dataset/E005/classification/test.csv' --save=./checkpoints/deepchrome/E005",
        "294_deepchrome_E061" : "python3 main.py  --globstr-train='./dataset/E061/classification/train.csv'  --globstr-val='./dataset/E061/classification/valid.csv' --globstr-test='./dataset/E061/classification/test.csv' --save=./checkpoints/deepchrome/E061",
        "294_deepchrome_E028" : "python3 main.py  --globstr-train='./dataset/E028/classification/train.csv'  --globstr-val='./dataset/E028/classification/valid.csv' --globstr-test='./dataset/E028/classification/test.csv' --save=./checkpoints/deepchrome/E028",
        "294_deepchrome_E037" : "python3 main.py  --globstr-train='./dataset/E037/classification/train.csv'  --globstr-val='./dataset/E037/classification/valid.csv' --globstr-test='./dataset/E037/classification/test.csv' --save=./checkpoints/deepchrome/E037",
    }

    # Time to wait between putting jobs on GPUs (in seconds). This is useful because it might take time
    # for a process to actually load the network onto the GPU, so we wait until that is done before
    # selecting the GPU for the next process.
    SLEEP_TIME = 20

    # Minimum memory required on a GPU to consider putting a job on it (MiB).
    MIN_MEMORY_REQUIRED = 20000


# Stick the shared args onto each JOB
for key, value in Config.JOBS.items():
    new_value = value + " " + Config.SHARED_ARGS
    Config.JOBS[key] = new_value

def select_gpu(GPUs):
    """
    Select the next best available GPU to run on. If nothing exists, return None
    """
    GPUs = list(filter(lambda gpu: gpu.memoryFree > Config.MIN_MEMORY_REQUIRED and gpu.id in Config.ALLOWED_GPUIDS, GPUs))
    if len(GPUs) == 0:
        return None
    GPUs = sorted(GPUs, key=lambda gpu: gpu.memoryFree)
    return GPUs[-1]

for index, (tmux_session_name, command) in enumerate(Config.JOBS.items()):

    # Check if we skip this one
    savedir = command.split("--")[-1][5:].strip()
    if os.path.exists(os.path.join(savedir, "test_results.json")):
        print(f"Skipping {tmux_session_name} because done")
        continue

    # Get the best available GPU
    print("Finding GPU for command \"{0}\"".format(command))
    curr_gpu = None
    while curr_gpu == None:
        curr_gpu = select_gpu(GPUtil.getGPUs())
        if curr_gpu != None:
            break
        else:
            time.sleep(Config.SLEEP_TIME)

    print("SUCCESS! Found GPU id = {0} which has {1} MiB free memory".format(curr_gpu.id, curr_gpu.memoryFree))

    result = subprocess.run("tmux new-session -d -s {0}".format(tmux_session_name), shell=True)
    if result.returncode != 0:
        print("Failed to create new tmux session called {0}".format(tmux_session_name))
        print("Continuing...")
        continue

    result = subprocess.run("tmux send-keys 'CUDA_VISIBLE_DEVICES={0} {1}' C-m".format(
        curr_gpu.id, command
    ), shell=True)
    if result.returncode != 0:
        print("Failed to run {0} in tmux session".format(command, tmux_session_name))
        sys.exit(result.returncode)

    print("---------------------------------------------------------------")

    if index < len(Config.JOBS) - 1:
        time.sleep(Config.SLEEP_TIME)