"""Forward Model Variance Intrinsic Reward

    Relies on bagging principle:
    > multiple, independent models fit on different subsets of trajectories

    Theory: once a state region has been visited a reasonable number of times
        --> independent model prediction models will converge
    Key: we should get prediction convergence even if high prediciton error
        > Ex: gaussian prediction. In high error environment with large
            number of visits --> model converge to same gaussian
            with high variance
"""


# TODO: Mean; Continuous space
# > L2 distance(average model preds)
# > Average(Pairwise L2 dists)


# TODO: Gaussian Variance Calcs
# Ideas?
# > KL-Divergence(average gaussian)
# > Average(Pairwise KL-divergence)


# TODO: Categorical Variance Calcs
# > Average (x-entropy from average probs)?
# > Average Pairwise x-entropy?


# TODO: high order funcs
# Q? should errors and variances be computed separately?
# ... probably cuz variances won't be part of the graph
# Q? can we use existing forward model errors?
#   nope: these depend on parallel model signatures
