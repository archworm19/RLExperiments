# RLExperiments
Reinforcement Learning Experiments

Design Philosophy
- Architecture components implement Layer Signatures. These signatures provide tests based on expected output shapes and types.
- Framework examples: deep Q learning, PPO, forward error intrinsic motivations, etc...
- Bottom-level framework functions should operate on tensors. They are flexible but provide no automatic testing.
- High-level framework function are optional. These functions operate on layer signatures. They have less flexibility but provide automatic testing.
