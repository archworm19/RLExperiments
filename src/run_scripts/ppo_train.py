"""PPO"""
import numpy as np
import numpy.random as npr
from run_scripts.runner import simple_run
from run_scripts.ll import async_master, sync_master
from run_scripts.builders import EnvsDiscrete, EnvsContinuous, build_discrete_ppo, build_continuous_ppo


def run_and_train(env_run, env_viz, agent,
                  num_actions: int,
                  num_epoch: int,
                  T_run: int,
                  T_test: int,
                  run_cutoff: int,
                  viz_debug: bool = False,
                  discrete_mode: bool = True):
    # total train timepoints = T_run
    #   cutoff runs after [run_cutoff] steps
    # NOTE: applies one-hot to actions in discrete mode
    # T_test = max run length for visual env test
    rng = npr.default_rng(42)
    for _ in range(num_epoch):
        sv, av, rv, tv = [], [], [], []
        num_step = 0
        while(num_step < T_run):
            env_run.reset(seed=int(rng.integers(10000)))
            print("pre run")
            states, actions, rewards, term = simple_run(env_run, agent, run_cutoff, debug=False, discrete=discrete_mode)
            print("post run")
            sv.append(np.array(states))
            av.append(np.array(actions))
            rv.append(np.array(rewards))
            tv.append(term)
            num_step += np.shape(rv[-1])[0]
        r_fin = []
        for tvi, rvi in zip(tv, rv):
            if tvi or (len(rvi) == run_cutoff):
                r_fin.append(np.sum(rvi))
        print("Average Reward For Terminated Runs: " + str(np.mean(np.array(r_fin))))
        print("Max Reward For Terminated Runs: " + str(np.amax(np.array(r_fin))))
        agent.train([{"core_state": si} for si in sv],
                    rv,
                    av,
                    tv)
        # test with viz
        env_viz.reset(seed=int(rng.integers(10000)))
        _, _, reward, _ = simple_run(env_viz, agent, T_test, debug=viz_debug, discrete=discrete_mode)
        print(np.sum(reward))


if __name__ == "__main__":

    # async testing
    # _, _, agent = build_continuous_ppo(EnvsContinuous.pendulum, gamma=0.95,
    #                                                learning_rate=.0001,
    #                                                layer_sizes=[256, 128, 64],
    #                                                embed_dim=16,
    #                                                entropy_scale=0.0, eta=0.3)
    # _, _, agent = build_continuous_ppo(EnvsContinuous.lunar_continuous,
    #                                                gamma=0.95,
    #                                                learning_rate=.0001,
    #                                                entropy_scale=0.0, eta=0.3,
    #                                                layer_sizes=[256, 128, 64])
    _, _, agent = build_continuous_ppo(EnvsContinuous.mountain_car, gamma=0.95,
                                                   learning_rate=.0001,
                                                   layer_sizes=[256, 128, 64],
                                                   embed_dim=16,
                                                   entropy_scale=0.0, eta=0.3)
    test_T = 500
    async_master(2, EnvsContinuous.mountain_car.value, agent, 100, int(5*test_T), test_T, "weights/",
                 load_pretrained=False)

    # # sync testing
    # builder = partial(build_continuous_ppo, env=EnvsContinuous.pendulum, learning_rate=.0001,
    #                   layer_sizes=[256, 128, 64], embed_dim=16,
    #                   entropy_scale=0.0, eta=0.3)
    # test_T = 500
    # sync_master(builder, 2, "weights/", 100, int(test_T * 5), test_T, 42, False, True, load_pretrained=True)
