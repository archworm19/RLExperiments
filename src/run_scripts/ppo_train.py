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
                  T_run: int, T_test: int):
    rng = npr.default_rng(42)
    for _ in range(num_epoch):
        env_run.reset(seed=int(rng.integers(10000)))
        states, actions, rewards, terms = runner_epoch(env_run, agent, T_run, rng)
        actions = [_one_hot(np.array(ai), num_actions) for ai in actions]
        agent.train([{"core_state": np.array(si)} for si in states],
                    [np.array(ri) for ri in rewards],
                    [np.array(ai) for ai in actions],
                    terms)
        # test with viz
        env_viz.reset(seed=int(rng.integers(10000)))
        print(simple_run(env_viz, agent, T_test))


if __name__ == "__main__":
    env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.cartpole)
    run_and_train(env_run, env_viz, agent, EnvsDiscrete.cartpole.value.dims_actions, 50, 5000, 1000)
