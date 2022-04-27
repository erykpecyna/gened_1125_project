from utility import run_experiment
from Agents import LunarAgent_UCB
from Environments import LunarEnvironment

run_config = {
    "num_episodes"       : 1400,
    "batch_size"         : 64,
    "memory_size"        : int(1e5),
    "gamma"              : 0.99,
    "learning_rate"      : 5e-4,
    "tau"                : 1e-3,
    "replay_iterations"  : 4,
    "tune"               : 0,
    "sampling method"    : "My method"
}

def main():
    run_experiment(LunarEnvironment, LunarAgent_UCB, run_config, name="models/lunar_ucb_1000.pt", model_path="models/lunar_ucb_{}.pt")

if __name__ == "__main__":
    main()