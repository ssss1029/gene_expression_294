

import json 

get_train = lambda x : f'./dataset/{x}/classification/train.csv'
get_test = lambda x : f'{x}'
get_val = lambda x : f'{x}'

with open("sam_experiments.json", 'r') as f:
    EXPS = json.load(f)

kill_commands = []

for exp_name in EXPS:
    train_set = EXPS[exp_name]['train']
    test_set = EXPS[exp_name]['test']

    exp_name_prefix = exp_name[:-5]
    exp_name_suffix = exp_name[-4:]
    tmux_name = "294_" + exp_name

    kill_commands.append(f"tmux kill-session -t {tmux_name}")

    train_arg = ""
    for elem in train_set:
        train_arg += " --globstr-train='{}'".format(get_train(elem))

    test_arg = ""
    for elem in test_set:
        test_arg += " --globstr-test-cell-id='{}'".format(get_test(elem))
    
    val_arg = ""
    for elem in test_set:
        val_arg += " --globstr-val-cell-id='{}'".format(get_val(elem))

    save_arg = "--save=./checkpoints/{}/{}".format(exp_name_prefix, exp_name_suffix)

    print("\"{}\" : \"python main.py {} {} {} {}\"".format(tmux_name, train_arg, test_arg, val_arg, save_arg))


print("------- KILL COMMANDS ------")
for k in kill_commands:
    print(k)
