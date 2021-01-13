from replay_buffer import ReplayBuffer
import json
from utils import mkdir


with open ("param.json", "r") as f:
    param = json.load(f)

config = param

memory = ReplayBuffer((6, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
memory.load_memory(config["buffer_path"])

memory_expert = ReplayBuffer((6, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])

for idx in range(memory.idx):
    for a in range(8):
        memory_expert.add(memory.obses[idx], a, memory.rewards[idx], memory.next_obses[idx], memory.not_dones[idx], memory.not_dones_no_max[idx])
    print(memory_expert.idx)
path = "all_action_expert"
mkdir("", path)
memory_expert.save_memory(path)
