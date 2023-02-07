"""PPO"""
import numpy as np
import numpy.random as npr
from run_scripts.runner import simple_run
from run_scripts.builders import EnvsDiscrete, build_discrete_ppo


def _one_hot(x: np.ndarray, num_action: int):
    # --> T x num_action
    v = [1. * (x == i) for i in range(num_action)]
    return np.concatenate([vi[:, None] for vi in v], axis=1)


def run_and_train(env_run, env_viz, agent,
                  num_actions: int,
                  num_epoch: int,
                  T_run: int,
                  T_test: int,
                  run_cutoff: int,
                  viz_debug: bool = False):
    # total train timepoints = T_run
    #   cutoff runs after [run_cutoff] steps
    # T_test = max run length for visual env test
    rng = npr.default_rng(42)
    for _ in range(num_epoch):
        sv, av, rv, tv = [], [], [], []
        num_step = 0
        while(num_step < T_run):
            env_run.reset(seed=int(rng.integers(10000)))
            states, actions, rewards, term = simple_run(env_run, agent, run_cutoff, debug=False)
            sv.append(np.array(states))
            av.append(_one_hot(np.array(actions), num_actions))
            rv.append(np.array(rewards))
            tv.append(term)
            num_step += np.shape(rv[-1])[0]
        agent.train([{"core_state": si} for si in sv],
                    rv,
                    av,
                    tv)
        # test with viz
        env_viz.reset(seed=int(rng.integers(10000)))
        _, _, reward, _ = simple_run(env_viz, agent, T_test, debug=viz_debug)
        print(np.sum(reward))


if __name__ == "__main__":
    # env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.cartpole)
    # num_actions = EnvsDiscrete.cartpole.value.dims_actions
    # env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.acrobot)
    # num_actions = EnvsDiscrete.acrobot.value.dims_actions
    env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.lunar, entropy_scale=0.5, embed_dim=16, layer_sizes=[128, 64])
    num_actions = EnvsDiscrete.lunar.value.dims_actions
    run_and_train(env_run, env_viz, agent, num_actions, 50, 5000, 500, 500, viz_debug=False)
