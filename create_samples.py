import gym
import time
import sys
from helper import FrameStack
from replay_buffer import ReplayBuffer 
import argparse

def main(args):
    print(sys.version) 
    env = gym.make("MontezumaRevenge-v0")
    print(env.observation_space)
    memory = ReplayBuffer((3,84,84),(1,), args.buffer_size, 0, args.device)
    memory.load_memory("expert_policy/")
    print("continue with {}  samples".format(memory.idx))
    env  = FrameStack(env, args)
    state = env.reset()
    score = 0 
    done = False
    steps = 0
    continue_true = False
    continue_true = True
    if continue_true:
        memory.load_memory("expert_policy")
        amout = 400
        for idx in range(amout):
            action = memory.actions[idx]
            print(action)
            env.render()
            time.sleep(0.01)
            next_state, reward, done, _ = env.step(action)

    print("buffer actions done")
    while True:
        #action = env.action_space.sample()
        steps += 1
        while True:
            try:
                action = input()
                action = int(action)
                if action > 1000:
                    sys.exit()
                if action > 13:
                    continue

                break
            except:
                continue
               
        # 3 move right 4 move left
        # 5 go done 2 up
        # 10 jump
        # 11 jump right  12 jumpy left
        #action = 13

        print("a", action)
        env.render()
        next_state, reward, done, _ = env.step(action)
        memory.add(state, action, next_state)
        state = next_state
        print("state ", next_state.shape)
        print("action")
        print("reward ", reward)
        #action = 0
        #time.sleep(1)
        score += reward
        if steps % 10 == 0:
            path = "expert_policy-{}/".format(steps)
            memory.save_memory(path)
        if done:
            print("score ", score)
            break


    env.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="kuka_block_grasping-v0", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--buffer_size', default=20000, type=int)
    arg = parser.parse_args()
    main(arg)
