from replay_buffer import ReplayBuffer
from agent_iql import Agent
import sys
from datetime import datetime
import time
from utils import write_into_file, write_parameter

def time_format(sec):
    """
    
    Args:
        param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)



def train(env, args, config):
    """

    """
    data_path = "../../data/"
    # now = datetime.now()
    # dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    if args.mode == "finetune":
        print("fine tune q")
        agent = Agent(state_size=200, action_size=8,  config=config) 
        agent.fit_q(args, config)
        return
    t0 = time.time()
    memory = ReplayBuffer((6, config["size"], config["size"]), (1,), 8, config["buffer_size"], config["batch_size"], config["image_pad"], config["device"])
    memory.load_memory(data_path + config["buffer_path"])
    print("memory idx ",memory.idx)  
    # agent.load("models-28_11_2020_22:25:27/27000-")
    if args.limit_data:
        print("Use less data ")
        memory.idx = config["idx"] 
    print("memory idx ",memory.idx)  
    agent = Agent(state_size=200, action_size=8,  config=config) 
    if config["mode"] == "pretrain":
        print("Pretrain Encoder and Predicter")
        for t in range(config["predicter_time_steps"]):
            text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
            print(text, end = '')
            agent.pretrain(memory)
            if t % int(config["eval"]) == 0:
                agent.save("models-{}/{}-".format(dt_string, t))
                agent.test_predicter(memory)

    #print("memory_expert idx ",memory_t.idx)
    model_path = str(config['locexp']) + "/models"
    for t in range(config["predicter_time_steps"]):
        text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
        print(text, end = '')
        agent.learn(memory)
        if t % int(config["eval"]) == 0 and t > 150:
            lr_q = agent.optimizer.param_groups[0]['lr']
            lr_q_sh = agent.optimizer_shift.param_groups[0]['lr']
            lr_r = agent.optimizer_r.param_groups[0]['lr']
            text = "Learning Rate  q {}  \ shift {}  r {} ".format(lr_q, lr_q_sh, lr_r)
            print(text, end = '')
            agent.eval_policy(t, args, False, True)
            #agent.eval_policy(t, args, False, False)
            agent.save(model_path + "/{}-".format(t))
            #agent.test_predicter(memory)
            agent.test_q_value(memory)


    print("done training write in file")
    text = "Run {}  best r {} at {} | best q {} at {}  time {}".format(config['run'], agent.best_r, agent.best_r_step, agent.best_q, agent.best_q_step, time_format(time.time() - t0))
    filepath = "results"
    write_into_file(filepath, text)
