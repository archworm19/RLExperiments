"""PPO"""
import numpy as np
import numpy.random as npr
from run_scripts.runner import runner_epoch, simple_run
from run_scripts.builders import EnvsDiscrete, build_discrete_ppo


def _one_hot(x: np.ndarray, num_action):
    # --> T x num_action
    v = [1. * (x == i) for i in range(num_action)]
    return np.concatenate([vi[:, None] for vi in v], axis=1)


def run_and_train(env_run, env_viz, agent,
                  num_actions: int,
                  num_epoch: int,
                  num_run_lb: int,
                  T_run: int,
                  T_test: int):
    # num_run_lb = lower bound on the number of env runs
    #       per training
    # total timepoints = num_run_lb * T_run
    # T_test = max run length for visual env test
    rng = npr.default_rng(42)
    for _ in range(num_epoch):
        sv, av, rv, tv = [], [], [], []
        for _ in range(num_run_lb):
            env_run.reset(seed=int(rng.integers(10000)))
            states, actions, rewards, terms = runner_epoch(env_run, agent, T_run, rng)
            actions = [_one_hot(np.array(ai), num_actions) for ai in actions]
            sv += states
            av += actions
            rv += rewards
            tv += terms
        agent.train([{"core_state": np.array(si)} for si in sv],
                    [np.array(ri) for ri in rv],
                    [np.array(ai) for ai in av],
                    tv)
        # test with viz
        env_viz.reset(seed=int(rng.integers(10000)))
        print(simple_run(env_viz, agent, T_test))


if __name__ == "__main__":
    # env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.cartpole)
    # num_actions = EnvsDiscrete.cartpole.value.dims_actions
    # env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.acrobot)
    # num_actions = EnvsDiscrete.acrobot.value.dims_actions
    env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.lunar)
    num_actions = EnvsDiscrete.lunar.value.dims_actions
    run_and_train(env_run, env_viz, agent, num_actions, 15, 10, 1000, 1000)
