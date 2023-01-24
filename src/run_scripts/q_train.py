"""
    Deep Q Learning + Open AI Gym

"""
import numpy as np
import numpy.random as npr
from frameworks.agent import Agent

from run_scripts.runner import runner
from run_scripts.builders import EnvsDiscrete, build_discrete_q



def run_and_train(env_run,  # TODO: object type?
                  env_disp,  # TODO: object type?
                  agent: Agent,  # TODO: need more specific interface for discrete/update agent
                  run_length: int = 1000,
                  num_runs: int = 400,
                  show_progress: bool = True,
                  seed_runs: int = 20,
                  step_per_train: int = 1,
                  step_per_copy: int = 1,
                  runs_per_display: int = 5,
                  timeout: bool = False,
                  debug_viz: bool = False):
    # NOTE: assumes env and agent are compatible
    #   should be handled by builder functions
    # number of seeds per 
    num_runs = num_runs

    rews = []
    # different seed for each run:
    for i in range(num_runs):
        if i % runs_per_display == 0:  # use display
            active_env = env_disp
            debug_set = debug_viz
            train_mode = False
        else:
            active_env = env_run
            debug_set = False
            train_mode = True

        if i < seed_runs:
            train_mode = False

        active_env.reset(seed=npr.randint(num_runs))
        # gather run data
        rewards = runner(active_env, agent, run_length,
                            debug=debug_set, train_mode=train_mode,
                            step_per_train=step_per_train,
                            step_per_copy=step_per_copy,
                            timeout=timeout)
        rews.append(np.sum(np.array(rewards)))

        print("train mode: {0}, reward: {1}".format(train_mode, rews[-1]))

        # signal end of epoch
        agent.end_epoch()
    return rews


if __name__ == "__main__":
    # cartpole:
    env_run, env_disp, agent = build_discrete_q(EnvsDiscrete.cartpole)
    run_and_train(env_run, env_disp, agent, run_length=EnvsDiscrete.cartpole.value.run_length, seed_runs=10)
    # lunar lander:
    # env_run, env_disp, agent = build_discrete_q(EnvsDiscrete.lunar)
    # run_and_train(env_run, env_disp, agent, run_length=EnvsDiscrete.lunar.value.run_length, seed_runs=10,
    #               timeout=False, debug_viz=False)
    env_run.close()
    env_disp.close()
