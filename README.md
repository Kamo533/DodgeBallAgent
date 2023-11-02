# DodgeBallEANN

Evolutionary algorithm based on neural networks for developing game agents. MLAgents Unity package is used for Unity communication.

Folders to create to run NEAT algorithm (in UnityInterface): result/fitness and checkpoints

To run NEAT algorithm, run UnityInterface.py

To play against best genome in Unity (NEAT scene), you must first run UnityInterfaceCTRNN.py

Inspiration TODO:

- Set start to train with against an agent that don't not move
- change compatibility_threshold up (looks like mean genetic distance ~1.93 (climbing through run), with standard deviation ~0.3, when compatibility_threshold = 1.5)
- turn up conn_add_prob and node_add_prob ?
- change fitness function
  - penalty for standing still (corners)
  - look at patterns in positions over time
  - other things?
