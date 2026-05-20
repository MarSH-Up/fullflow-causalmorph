import numpy as np

# Connectivity matrix A
A = np.array(
    [
        [0, 00.49, -0.49],  # Region 1 (no incoming connections)
        [-0.49, 0, 0],  # Region 2 (connected to Region 1)
        [0.49, 0, 0],  # Region 3 (connected from Region 1)
    ]
)

# Change in connectivity induced by kth input, here B1 and B2
B1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
B2 = np.array([[0, 0, 0], [0.77, 0, 0], [0.77, 0, 0]])

B = np.zeros((3, 3, 2))
B[:, :, 0] = B1
B[:, :, 1] = B2

# Influence of input on regions
C = np.array([[0.08, 0, 0], [0, 0.08, 0], [0, 0, 0.08]])

# Other parameters
P_SD = np.array(
    [[0.0775, -0.0087], [-0.1066, 0.0299], [0.0440, -0.0129], [0.8043, -0.7577]]
)

freq = 10.84
step = 1 / freq

actionTime = [5, 5, 5]
restTime = [35, 35, 35]
cycles = [5, 5, 5]

Parameters = {
    "A": A,
    "B": B,
    "C": C,
    "P_SD": P_SD,
    "freq": freq,
    "step": step,
    "actionTime": actionTime,
    "restTime": restTime,
    "cycles": cycles,
}
