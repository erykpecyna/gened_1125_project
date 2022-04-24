from utility import run_experiment
from Agents import LunarAgent
from Environments import LunarEnvironment

run_config = {
    "num_episodes"       : 100,
    "batch_size"         : 8,
    "memory_size"        : 50000,
    "gamma"              : 0.99,
    "learning_rate"      : 1e-4,
    "tau"                : 0.01,
    "num_replay_updates" : 5,
}

def main():
    run_experiment(LunarEnvironment, LunarAgent, run_config, model_path="models/lunar_{}.pth")

if __name__ == "__main__":
    main()