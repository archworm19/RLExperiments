import numpy as np
from unittest import TestCase
from replay_buffers.async_storage import TrajectoryStore


class TestTrajectoryStore(TestCase):

    def test_feedin_feedout(self):
        # best test:
        # > make trajectories
        # > feed 'em into TS --> get 'em back
        TS = TrajectoryStore(3, 10, [2, 4])

        # (pid, data)
        trajs = [[0, [np.ones((6, 2)), np.ones((6, 4))]],
                [1, [10 + np.ones((5, 2)), 10 + np.ones((5, 4))]],
                [2, [100 + np.ones((10, 2)), 100 + np.ones((10, 4))]],
                [1, [11 + np.ones((5, 2)), 11 + np.ones((5, 4))]],
                [0, [1 + np.ones((4, 2)), 1 + np.ones((4, 4))]]]
        for traj in trajs:
            for i in range(len(traj[1][0])):
                if i == len(traj[1][0]) - 1:
                    TS.add_datapt(traj[0], [traj[1][0][i], traj[1][1][i]], True)
                else:
                    TS.add_datapt(traj[0], [traj[1][0][i], traj[1][1][i]], False)

        [var0, var1] = TS.pull_trajectories()
        targ_vals = [11., 1., 2., 12., 101.]
        targ_T = [5, 6, 4, 5, 10]
        for v0, v1, tvi, Ti in zip(var0, var1, targ_vals, targ_T):
            self.assertTrue(np.all(v0 == tvi))
            self.assertTrue(np.all(v1 == tvi))
            self.assertTrue(np.shape(v0)[0] == Ti)
            self.assertTrue(np.shape(v1)[0] == Ti)
            self.assertTrue(np.shape(v0)[1] == 2)
            self.assertTrue(np.shape(v1)[1] == 4)

    def test_mix(self):
        TS = TrajectoryStore(2, 5, [1])
        TS.add_datapt(0, [np.array([1.])], False)
        TS.add_datapt(1, [np.array([10.])], False)
        TS.add_datapt(0, [np.array([2.])], False)
        TS.add_datapt(1, [np.array([11.])], False)
        TS.add_datapt(1, [np.array([12.])], False)
        [x0] = TS.pull_trajectories()
        self.assertTrue(np.all(x0[0] == np.array([[1.], [2.]])))
        self.assertTrue(np.all(x0[1] == np.array([[10.], [11.], [12.]])))


if __name__ == "__main__":
    TTS = TestTrajectoryStore()
    TTS.test_feedin_feedout()
    TTS.test_mix()
