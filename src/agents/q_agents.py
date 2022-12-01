"""Agents that use Q Learning

    from frameworks.q_learning:
        scalar_model assumption:
                call has the following signature:
                    call(action_t: tf.Tensor, state_t: List[tf.Tensor])
"""
import numpy as np
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer
from frameworks.agent import Agent, RunData
from frameworks.q_learning import QLearning
from frameworks.custom_model import CustomModel
from agents.utils import build_action_probes


class QAgent(Agent):

    def __init__(self,
                 scalar_model: Layer,
                 num_actions: int,
                 state_dims: int,
                 gamma: float = 0.95):
        """
        Args:
            scalar_model (Layer): keras layer where the call
                has the following signature:
                    call(action_t: )
            num_actions (int): number of actions available to
                the agent
            state_dims (int): number of dimensions in state
                assumes state can be easily represented by
                single tensor
            gamma (float): discount factor
        """
        super(QAgent, self).__init__()
        self.scalar_model = scalar_model
        self.num_actions = num_actions

        # TODO: is this really the right place
        # to build the custom model?
        inputs = [tf.keras.Input(shape=(num_actions,),
                                 name="action", dtype=tf.float32),
                  tf.keras.Input(shape=(),
                                 name="reward", dtype=tf.float32),
                  tf.keras.Input(shape=(state_dims,),
                                 name="state", dtype=tf.float32),
                  tf.keras.Input(shape=(state_dims,),
                                 name="state_t1", dtype=tf.float32)]
        Qerr, _ = QLearning(gamma).calc_error(num_actions,
                                              scalar_model,
                                              inputs[1],
                                              inputs[0],
                                              [inputs[2]],
                                              [inputs[3]])
        outputs = {"loss": tf.math.reduce_mean(Qerr)}
        self.kmodel = CustomModel("loss",
                                  inputs=inputs,
                                  outputs=outputs)
        self.kmodel.compile(optimizer=tf.keras.optimizers.Adam(.01))

    def select_action(self, state: List[np.ndarray]):
        # TODO: missing randomness!!!
        """greedy action selection

        Args:
            state (List[np.ndarray]): set of unbatched input tensors
                each with shape:
                    ...

        Returns:
            int: index of selected action
        """
        # --> action_t = num_actions x num_actions
        # --> state_t = num_actions x ...
        action_t, state_t = build_action_probes(state, self.num_actions)
        # --> shape = num_actions
        scores = self.scalar_model(action_t, state_t)
        # greedy
        return tf.argmax(scores).numpy()

    def _build_dset(self, run_data: RunData):
        # NOTE: keys must match keras input names
        dmap = {"action": run_data.actions,
                "reward": run_data.rewards,
                "state": run_data.states,
                "state_t1": run_data.states_t1}
        dset = tf.data.Dataset.from_tensor_slices(dmap)
        return dset.shuffle(5000).batch(32)

    def train(self, run_data: RunData, num_epoch: int):
        """train agent on run data

        Args:
            run_data (RunData):
            num_epoch (int):
        """
        dset = self._build_dset(run_data)
        history = self.kmodel.fit(dset,
                                  epochs=num_epoch)
        return history


if __name__ == "__main__":
    # test on fake dataset:
    #   sum[state] > 0 and action = 1 --> reward
    #   else --> no reward
    import numpy.random as npr
    rng = npr.default_rng(42)
    states = rng.random((1000, 2)) - 0.5
    action0 = (rng.random((1000,)) > 0.5) * 1
    actions = np.vstack((action0, 1. - action0)).T
    rewards = (1. * (np.sum(states, axis=1) > 0.)) * action0
    dat = RunData(states[:-1], states[1:], actions[:-1],
                  rewards[:-1])
    print(dat)

    from tensorflow.keras.layers import Dense
    from arch_layers.simple_networks import DenseNetwork
    class DenseScalar(Layer):
        def __init__(self):
            super(DenseScalar, self).__init__()
            self.d_act = Dense(4)
            self.d_state = Dense(4)
            self.net = DenseNetwork([10], 1, 0.)

        def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
            x_a = self.d_act(action_t)
            x_s = self.d_state(state_t[0])
            yp = self.net(tf.concat([x_a, x_s], axis=1))
            return yp[:, 0]  # to scalar

    QA = QAgent(DenseScalar(), 2, 2,
                gamma=0.8)
    dset = QA._build_dset(dat)

    # testing dataset format and model running
    for v in dset:
        print(v)
        print(QA.kmodel(v))
        break
    # testing action selection (pre train)
    print(QA.select_action([states[0]]))

    # testing training
    QA.train(dat, 8)

    # expectation?
    # > s_{t} > 0, a_{t} = 0: max ~ when reward is 1
    # > all else will be smaller
    # ... difference will get more extreme as gamma --> 0
    for v in dset:
        rews = v["reward"].numpy()
        q = QA.scalar_model(v["action"], [v["state"]]).numpy()
        print(np.mean(q[rews >= 0.5]))
        print(np.mean(q[rews <= 0.5]))
        break
