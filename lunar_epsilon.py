from utility import run_experiment
from Agents import LunarAgent_epsilon
from Environments import LunarEnvironment

run_config = {
    "num_episodes"       : 1000,
    "batch_size"         : 8,
    "memory_size"        : 50000,
    "gamma"              : 0.99,
    "learning_rate"      : 1e-4,
    "tau"                : 0.01,
    "replay_iterations"  : 5,
    "tune"               : 0,
    "sampling method"    : "Epsilon",
    "epsilon"            : 1,
    "epsilon_decay"      : 0.995,
    "epsilon_min"        : 0,
}

def main():
    run_experiment(LunarEnvironment, LunarAgent_epsilon, run_config, model_path="models/lunar_epsilon_{}.pt")

if __name__ == "__main__":
    main()