"""Forward Model Exploration Rewards (intrinsic motivation)
"""

# TODO: overarching design for frameworks
# > lowest level = tensors --> allows for more generalization
# > next level = models with certain signatures --> more foolproof


# TODO: forward model error ~ 'surprisal'
# > inputs?
#       1. states: x_t, x_{t+1}
#           TODO: worth it to generalize to more states?
#       2. action: a_t
# > networks?
#       1. embedding network: phi(x_t)
#       2. forward model: p(phi(x_{t+1}) | x_t, a_t)
#            TODO: should it be p(phi(x_{t+1}) | X, A)
#               whole action, state series
#               ... yes, at least give the option of state seqs
# > errors? start with this! (in terms of tensors)
#       1. Burda 2018 uses MSE between foreward model and phi(x_{t+1})


# TODO: encoding models? phi
# > random phi ~ no machinery needed here
# > Inverse Dynamics Features (IDF)
#       maths?
#           inverse model = p(a_t | phi(s_t), phi(s_{t+1}))
#       1. first level: errors using tensors ~ need separate for discrete and continuous action spaces
#       2. second level: wrap first level with function signatures!