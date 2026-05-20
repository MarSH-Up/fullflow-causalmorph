import numpy as np
from scipy.integrate import odeint


def Neurodynamics_Model(Z, t, A, B, C, U):
    """
    Neurodynamics model for multiple brain regions.

    Parameters:
    - Z: The state of the system. Shape: (nRegions,).
    - t: Current time.
    - A: Connectivity matrix. Shape: (nRegions, nRegions).
    - B: Influence matrix. Shape: (nRegions, nRegions, number of inputs).
    - C: Input effect matrix. Shape: (nRegions, number of inputs).
    - U: Input matrix. Shape: (number of inputs, number of timestamps).

    Returns:
    - dZdt: The rate of change of the system's state.
    """

    # Get the number of brain regions
    nRegions = A.shape[0]
    index = min(int(t * 10), U.shape[1] - 1)

    # Calculate the contribution from the influence matrix B and the input U
    T = np.zeros(nRegions)
    for uu in range(B.shape[2]):
        tmp = U[uu, index] * B[:, :, uu]
        T += -0.5 * np.exp(np.sum(tmp, axis=1))

    # Modify the diagonal of the connectivity matrix A
    SI = np.diag(A)
    new_diag = np.exp(SI) / 2 + SI
    A -= np.diagflat(new_diag)
    J_t = A + T

    # Calculate the rate of change of the system's state
    dZdt = np.dot(J_t, Z) + np.dot(C, U[:, index])
    return dZdt


def Neurodynamics(Z0, timestamps, A, B, C, U_stimulus):
    """
    Integrate the neurodynamics across all brain regions.

    Parameters:
    - Z0: Initial state of the system. Shape: (nRegions,).
    - timestamps: Array of time points.
    - A: Connectivity matrix. Shape: (nRegions, nRegions).
    - B: Influence matrix. Shape: (nRegions, nRegions, number of inputs).
    - C: Input effect matrix. Shape: (nRegions, number of inputs).
    - U_stimulus: Stimulus input matrix. Shape: (number of inputs, number of timestamps).

    Returns:
    - Z: The system's state at each timestamp. Shape: (number of timestamps, nRegions).
    """

    Z = odeint(
        Neurodynamics_Model,
        Z0,
        t=timestamps,
        args=(A, B, C, U_stimulus),
    )

    return Z
