import torch
import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def softmax(act_vals, tau = 1.0):
    max_vals = torch.max(act_vals, axis=1, keepdim=True)[0]/tau
    act_vals = act_vals/tau

    pref = act_vals - max_vals

    exp_act = torch.exp(pref)
    sum_exp_act = torch.sum(exp_act, axis=1).view(-1,1)

    p = exp_act / sum_exp_act
    return p

def train_network(memory_sample, model, target_model, optimizer, criterion, discount, tau):
    """
    One training iteration of the network
    """
    optimizer.zero_grad()

    # Note that states and next_states are lists of tensors, but the rest are not
    states, actions, terminals, rewards, next_states = map(list, zip(*memory_sample))

    # This disables unnecessary gradient computation (= faster)
    with torch.no_grad():
        q_next = target_model(torch.stack(next_states)).squeeze()
    p = softmax(q_next, tau)

    # Figure out maximum rewards reachable
    max_next_q = (1 - torch.tensor(terminals, device=device)) * (torch.max(q_next, axis = 1)[0])
    rewards = torch.tensor(rewards, device=device).float()
    targets = (rewards + (discount * max_next_q)).float()

    outputs = model(torch.stack(states).float()).squeeze()
    actions = torch.stack(actions)
    outputs = torch.gather(outputs, 1, actions).squeeze()

    # Loss
    loss = criterion(outputs, targets)
    loss.backward()

    # Update weights
    optimizer.step()

def run_experiment(Environment, Agent, config, episode_max_steps = 0, name = "", model_path = "results/model_{}.pth"):
    wandb.init(project="lunar_lander", entity="caramelcougar")

    env = Environment(config)
    # Assuming Discrete action space
    config["num_actions"] = env.env.action_space.n
    agent = Agent(config)

    wandb.config(config)


    if config["tune"]:
        checkpoint = torch.load(name)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        start_episode = checkpoint['episode'] + 1
        print("Tuning model")
    else:
        start_episode = 0
        print("Training model")
    
    for episode in tqdm(range(start_episode, start_episode + config['num_episodes'])):
        # Run an episode
        terminal = False 
        total_reward = 0.0
        num_steps = 1
        last_state = env.start()
        last_action = agent.start(last_state)
        obs = (last_state, last_action)

        while (not terminal) and (num_steps < episode_max_steps or episode_max_steps == 0):
            (reward, last_state, term) = env.step(last_action)
            total_reward += reward
            if term:
                agent.end(reward)
                step_obs = (reward, last_state, None, term)
            else:
                num_steps += 1
                last_action = agent.step(reward, last_state)
                step_obs = (reward, last_state, last_action, term)
            terminal = term
        
        # Get episode reward
        episode_reward = agent.get_sum_rewards()

        wandb.log({
            "Episode" : episode,
            "Reward" : episode_reward,
        })

        if episode == start_episode + config['num_episodes'] - 1:
            torch.save({
                'episode' : episode,
                'model_state_dict' : agent.model.state_dict(),
            }, model_path.format((episode + 1)))
        print("Episode: {}, Reward: {}".format(episode, episode_reward))



