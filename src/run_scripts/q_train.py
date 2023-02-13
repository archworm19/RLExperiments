"""
    Deep Q Learning + Open AI Gym

"""
import numpy as np
import numpy.random as npr
from frameworks.agent import Agent

from run_scripts.runner import runner, simple_run
from run_scripts.builders import (EnvsDiscrete, EnvsContinuous,
                                  build_discrete_q, build_discrete_q_atoms, build_continuous_q)


def run_and_train(env_run,  # TODO: object type?
                  env_disp,  # TODO: object type?
                  agent: Agent,  # TODO: need more specific interface for discrete/update agent
                  run_length: int = 1000,
                  num_runs: int = 400,
                  seed_runs: int = 20,
                  step_per_train: int = 1,
                  step_per_copy: int = 1,
                  runs_per_display: int = 5,
                  debug_viz: bool = False,
                  discrete: bool = True):
    # NOTE: assumes env and agent are compatible
    #   should be handled by builder functions
    # number of seeds per 
    num_runs = num_runs

    rews = []
    # different seed for each run:
    for i in range(num_runs):
        if i % runs_per_display == 0 or i < seed_runs:  # use display
            env_disp.reset(seed=npr.randint(num_runs))
            _, _, rewards, _ = simple_run(env_disp, agent, run_length, debug=debug_viz, discrete=discrete)
        else:
            env_run.reset(seed=npr.randint(num_runs))
            rewards = runner(env_run, agent, run_length,
                             debug=False,
                             step_per_train=step_per_train,
                             step_per_copy=step_per_copy,
                             discrete=discrete)

        rews.append(np.sum(np.array(rewards)))
        print(rews[-1])
        # signal end of epoch
        agent.end_epoch()
    return rews


if __name__ == "__main__":
    # cartpole:
    # env_run, env_disp, agent = build_discrete_q(EnvsDiscrete.cartpole)
    # run_and_train(env_run, env_disp, agent, run_length=EnvsDiscrete.cartpole.value.run_length, seed_runs=10, discrete=True, debug_viz=False)
    # lunar lander:
    # env_run, env_disp, agent = build_discrete_q(EnvsDiscrete.lunar)
    # run_and_train(env_run, env_disp, agent, run_length=EnvsDiscrete.lunar.value.run_length, seed_runs=10,
    #               timeout=False, debug_viz=False)

    # cartpole + C51:
    # env_run, env_disp, agent = build_discrete_q_atoms(EnvsDiscrete.cartpole, Vmin=0., Vmax=1000.)
    # run_and_train(env_run, env_disp, agent, run_length=EnvsDiscrete.cartpole.value.run_length, seed_runs=10,
    #               debug_viz=False, discrete=True)

    # pendulum + continuous DQN
    env_run, env_disp, agent = build_continuous_q(EnvsContinuous.pendulum)
    run_and_train(env_run, env_disp, agent, run_length=EnvsContinuous.pendulum.value.run_length, seed_runs=10,
                  debug_viz=False, discrete=False)

    # walker + continuous DQN
    # env_run, env_disp, agent = build_continuous_q(EnvsContinuous.bi_walker, embed_dim=8, layer_sizes=[256, 128],
    #                                               tau=.05, sigma=0.6, theta=0.45)
    # run_and_train(env_run, env_disp, agent, run_length=EnvsContinuous.bi_walker.value.run_length, seed_runs=10,
    #               timeout=False, debug_viz=False)

    env_run.close()
    env_disp.close()
