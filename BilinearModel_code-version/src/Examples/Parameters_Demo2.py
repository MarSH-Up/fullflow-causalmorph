import numpy as np

A = np.array([[-0.16, -0.49], [-0.02, -0.33]])

B1 = np.array([[0, 0], [0, 0]])
B2 = np.array([[-0.02, -0.77], [0.33, -1.31]])

B = np.zeros((2, 2, 2))
B[:, :, 0] = B1
B[:, :, 1] = B2

C = np.array([[0.08, 0], [0, 0.06]])

freq = 10.84
step = 1 / freq

actionTime = [5, 5]
restTime = [25, 25]
cycles = [6, 6]

P_SD = np.array(
    [[0.0775, -0.0087], [-0.1066, 0.0299], [0.0440, -0.0129], [0.8043, -0.7577]]
)

Parameters = {
    "A": A,
    "B1": B1,
    "B2": B2,
    "B": B,
    "C": C,
    "P_SD": P_SD,
    "freq": freq,
    "step": step,
    "actionTime": actionTime,
    "restTime": restTime,
    "cycles": cycles,
}
