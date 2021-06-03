# inspiration from
# https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python/blob/master/Chapter04/SARSA%20Q_learning%20Taxi-v2.py

import time
import numpy as np
import gym
import gym_minigrid
import matplotlib.pyplot as plt


def eps_greedy(nA, Q, s, eps=0.1, ):
    '''
    Epsilon greedy policy
    '''
    if np.random.uniform(0, 1) < eps:
        # Choose a random action
        return np.random.randint(nA)
    else:
        # Choose the action of a greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    '''
    Greedy policy
    return the index corresponding to the maximum action-state value
    '''
    return np.argmax(Q.get(getStateString(s)))


def run_episodes(env, Q, max_steps, num_episodes=100, to_print=False):
    '''
    Run some episodes to test the policy
    '''
    tot_rew = []
    state = env.reset()

    maxsteps = max_steps

    for _ in range(num_episodes):
        done = False
        game_rew = 0
        teststeps = 0
        actionlist = []
        while not done:
            # select a greedy action
            action = greedy(Q, state)
            next_state, rew, done, _ = env.step(action)
            actionlist.append(action)

            state = next_state
            game_rew += rew
            teststeps = teststeps + 1

            if teststeps == maxsteps:
                done = True
            if done:
                state = env.reset()
                tot_rew.append(game_rew)

    if to_print:
        print('Mean score: %.3f of %i games!' % (np.mean(tot_rew), num_episodes))

    # print("action list:",actionlist)
    return np.mean(tot_rew)


def updateDict(Q, nA, state):
    stateString = getStateString(state)

    if stateString not in Q:
        Q[stateString] = np.zeros(nA)


def getStateString(state):
    test = state["image"]

    stateString = ""

    for array in test:
        for i in array:
            for j in i:
                stateString = stateString + str(j)

    stateString = stateString + str(state["mission"])
    stateString = stateString + str(state["direction"])

    return stateString


def plotgraph(games_rewards, test_rewards, title, save):
    # plt.plot(games_rewards, label="all runs")
    plt.figure()
    plt.plot(test_rewards, label="100 run avg")

    plt.xlabel('100 run avg test (every 500 runs)')
    plt.ylabel('reward')
    plt.title(title)
    plt.legend()
    plt.ylim(0, 1.2)
    savestring = "images/" + save + ".png"
    plt.savefig(savestring)
    plt.close()


def plotComparison(Q_games_rewards, Q_test_rewards, SARSA_games_rewards, SARSA_test_rewards, title, save):
    plt.figure()
    # plt.plot(Q_games_rewards, label="Q_all runs")
    plt.plot(Q_test_rewards, label="Q - 100 run avg")

    # plt.plot(SARSA_games_rewards, label="SARSA_all runs")
    plt.plot(SARSA_test_rewards, label="SARSA - 100 run avg")

    def annot_max(x, y):
        ymax = max(max(Q_test_rewards), max(SARSA_test_rewards))

        if ymax in Q_test_rewards:
            xmax = Q_test_rewards.index(ymax)
        else:
            xmax = SARSA_test_rewards.index(ymax)
        text = "x={:.0f}, y={:.3f}".format(xmax, ymax)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data', textcoords="axes fraction", arrowprops=arrowprops, bbox=bbox_props, ha="right",
                  va="top")
        plt.annotate(text, xy=(xmax, ymax), xytext=(0.5, 0.96), **kw)

    annot_max(Q_test_rewards, SARSA_test_rewards)
    plt.xlabel('100 run avg test (every 500 runs)')
    plt.ylabel('reward')
    plt.title(title)
    plt.legend()
    plt.ylim(0, 1.2)
    savestring = "images/" + save + ".png"
    plt.savefig(savestring)
    plt.close()

def save_dec(nr: float) -> str:
    nr = str(nr)
    nr = nr.split(".")
    nr = "_".join(nr)
    return nr


Q_games_rewards, Q_test_rewards, SARSA_games_rewards, SARSA_test_rewards = [], [], [], []


def Q_learning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    nA = env.action_space.n
    starting_eps = eps
    Q = {}
    games_reward = []
    test_rewards = []

    bestScore = 0
    counter = 0

    for ep in range(num_episodes):
        if test_rewards != []:
            if max(test_rewards) > bestScore:
                bestScore = max(test_rewards)
                counter = 0
            else:
                counter = counter + 1

            if counter > 20000:
                break

        state = env.reset()

        updateDict(Q, nA, state)

        done = False
        tot_rew = 0

        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > 0.01:
            eps -= eps_decay

        # loop the main body until the environment stops
        while not done:
            # select an action following the eps-greedy policy
            action = eps_greedy(nA, Q, state, eps)

            next_state, rew, done, _ = env.step(action)  # Take one step in the environment

            updateDict(Q, nA, next_state)

            # Q-learning update the state-action value (get the max Q value for the next state)
            Q.get(getStateString(state))[action] = Q.get(getStateString(state))[action] + lr * (rew + gamma * np.max(Q.get(getStateString(next_state))) - Q.get(getStateString(state))[action])

            state = next_state
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

        # Test the policy every 500 episodes and print the results
        if (ep % 500) == 0 and ep != 0:
            test_rew = run_episodes(env, Q, env.max_steps, 100)
            test_rewards.append(test_rew)
            if (ep % 500) == 0 and ep != 0:
                print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}  len(Q):{:5d}".format(ep, eps, test_rew, len(Q)))
        if ep % 10000 == 0 and ep != 0:
            print("plotting and saving")
            q_save = "Q_LEARNING_" + name + save_dec(lr) + "lr" + save_dec(ep) + "eps" + save_dec(gamma) + \
                     "gamma" + str(ep) + "episodes"
            q_title = "Q_LEARNING_" + name
            plotgraph(games_reward, test_rewards, q_title, q_save)
            np.savez(q_save, *Q.items())


    global Q_test_rewards
    global Q_games_rewards
    Q_test_rewards = test_rewards
    Q_games_rewards = games_reward
    q_save = name + save_dec(lr) + "lr-" + save_dec(starting_eps) + "eps-" + save_dec(gamma) + \
                 "gamma-" + str(num_episodes) + "episodes- Q_LEARNING"
    q_title = "Q_LEARNING_" + name
    plotgraph(games_reward, test_rewards, q_title, q_save)
    np.savez(q_save, *Q.items())
    return Q


def SARSA(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    nA = env.action_space.n
    starting_eps = eps
    Q = {}
    games_reward = []
    test_rewards = []

    bestScore = 0
    counter = 0

    for ep in range(num_episodes):
        if test_rewards != []:
            if max(test_rewards) > bestScore:
                bestScore = max(test_rewards)
                counter = 0
            else:
                counter = counter + 1

            if counter > 20000:
                break


        done = False
        tot_rew = 0

        state = env.reset()

        updateDict(Q, nA, state)

        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > 0.01:
            eps -= eps_decay

        action = eps_greedy(nA, Q, state, eps)

        # loop the main body until the environment stops
        while not done:
            next_state, rew, done, _ = env.step(action)  # Take one step in the environment

            updateDict(Q, nA, next_state)

            # choose the next action (needed for the SARSA update)
            next_action = eps_greedy(nA, Q, next_state, eps)

            # SARSA update
            Q.get(getStateString(state))[action] = Q.get(getStateString(state))[action] + lr * (rew + gamma * Q.get(getStateString(next_state))[next_action] - Q.get(getStateString(state))[action])

            state = next_state
            action = next_action
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

        # Test the policy every 300 episodes and print the results
        if (ep % 500) == 0 and ep != 0:
            test_rew = run_episodes(env, Q, env.max_steps, 100)
            test_rewards.append(test_rew)
            if (ep % 500) == 0 and ep != 0:
                print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}  len(Q):{:5d}".format(ep, eps, test_rew, len(Q)))

        if ep % 10000 == 0 and ep != 0:
            print("plotting and saving")
            sarsa_save = "SARSA_" + name + save_dec(lr) + "lr" + save_dec(starting_eps) + "eps" + save_dec(gamma) + \
                         "gamma" + str(ep) + "episodes"
            sarsa_title = "SARSA_" + name
            plotgraph(games_reward, test_rewards, sarsa_title, sarsa_save)
            np.savez(sarsa_save, *Q.items())

    global SARSA_test_rewards
    global SARSA_games_rewards
    SARSA_test_rewards = test_rewards
    SARSA_games_rewards = games_reward
    sarsa_save = name + save_dec(lr) + "lr-" + save_dec(starting_eps) + "eps-" + save_dec(gamma) + \
                 "gamma-" + str(num_episodes) + "episodes- SARSA"
    sarsa_title = "SARSA_" + name
    plotgraph(games_reward, test_rewards, sarsa_title, sarsa_save)
    np.savez(sarsa_save, *Q.items())
    return Q


if __name__ == '__main__':

    fullstarttime = time.time()

    envs = ["MiniGrid-Empty-Random-6x6-v0", "MiniGrid-Empty-8x8-v0", "MiniGrid-Dynamic-Obstacles-5x5-v0", "MiniGrid-GoToDoor-5x5-v0",
            "MiniGrid-LavaGapS5-v0", "MiniGrid-Unlock-v0", "MiniGrid-SimpleCrossingS9N1-v0", "MiniGrid-Fetch-5x5-N2-v0"]


    episodes = 20000
    epsilon = 0.4
    gamma = 0.95
    eps_dec = 0.00001
    lr = .1

    for name in envs:
        env = gym.make(name)

        comparisontitle = "Q vs SARSA - " + name + str(episodes) + "episodes"
        save = "Q_VS_SARSA" + name + save_dec(lr) + "lr" + save_dec(epsilon) + "eps" + save_dec(gamma) + \
               "gamma" + str(episodes) + "episodes"

        print()
        print(name)

        start_time = time.time()
        print("SARSA")
        Q_sarsa = SARSA(env, lr=lr, num_episodes=episodes, eps=epsilon, gamma=gamma, eps_decay=eps_dec)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:0.0f} seconds")

        start_time = time.time()
        print("Q-LEARNING")
        Q_qlearning = Q_learning(env, lr=lr, num_episodes=episodes, eps=epsilon, gamma=gamma, eps_decay=eps_dec)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:0.0f} seconds")



        plotComparison(Q_games_rewards, Q_test_rewards, SARSA_games_rewards, SARSA_test_rewards, comparisontitle, save)
        Q_games_rewards, Q_test_rewards, SARSA_games_rewards, SARSA_test_rewards = [], [], [], []
    fullendtime = time.time()

    print()
    print(f"Total time taken: {fullendtime - fullstarttime:0.0f} seconds")
