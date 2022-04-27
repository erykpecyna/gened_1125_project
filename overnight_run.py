from utility import run_experiment
from Agents import LunarAgent_epsilon, LunarAgent_hybrid
from Environments import LunarEnvironment

epsilon_config = {
    "num_episodes"       : 1400,
    "batch_size"         : 64,
    "memory_size"        : int(1e5),
    "gamma"              : 0.99,
    "learning_rate"      : 5e-4,
    "tau"                : 1e-3,
    "replay_iterations"  : 4,
    "tune"               : 0,
    "sampling method"    : "Epsilon",
    "epsilon"            : 1,
    "epsilon_decay"      : 0.995,
    "epsilon_min"        : 0.01,
}

hybrid_config = {
    "num_episodes"       : 1400,
    "batch_size"         : 64,
    "memory_size"        : int(1e5),
    "gamma"              : 0.99,
    "learning_rate"      : 5e-4,
    "tau"                : 1e-3,
    "replay_iterations"  : 4,
    "tune"               : 0,
    "sampling method"    : "Hybrid",
    "epsilon"            : 1,
    "epsilon_decay"      : 0.995,
    "epsilon_min"        : 0.01,
}

def main():
    for i in range(5):
        run_experiment(
            LunarEnvironment,
            LunarAgent_epsilon,
            epsilon_config,
            model_path="models/overnight/epsilon_" + str(i) + "_{}.pt")
        run_experiment(
            LunarEnvironment,
            LunarAgent_hybrid,
            hybrid_config,
            model_path="models/overnight/hybrid_" + str(i) + "_{}.pt")

if __name__ == "__main__":
    main()