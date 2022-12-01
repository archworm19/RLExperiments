import tensorflow as tf
from typing import List


def build_action_probes(state: List[tf.Tensor],
                        num_actions: int):
    """tile state for each potential action
        and make identity matrix for actions
        --> run thru models to predict quantity
            for each possible action given state

    Args:
        state (List[tf.Tensor]): state at time t
            each tensor has shape ...
            == state tensors must be unbatched
                (single timepoint)
        num_actions (int): number of actions available
            to the agent

    Returns:
        tf.Tensor: action identity matrix
            num_actions x num_actions
        List[tf.Tensor]: tiled(st) for st in state
            each state tensor now has shape
            num_actions x ...
    """
    action_t = tf.eye(num_actions)
    tile_states = []
    for sti in state:
        num_dims = tf.shape(tf.shape(sti))[0]
        tile_sh = tf.concat([[num_actions],
                                tf.ones(num_dims, tf.int32)],
                                axis=0)
        tile_states.append(tf.tile(tf.expand_dims(sti, 0),
                                    tile_sh))
    return action_t, tile_states
