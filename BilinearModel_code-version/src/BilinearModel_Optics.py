import numpy as np


def BilinearModel_Optics(pj, qj, U, A):
    """
    Compute the optics based on the bilinear model for multiple brain regions.

    Parameters:
    - pj: Total hemoglobin concentration. A 2D array of shape (nRegions, simulationLength).
    - qj: Deoxyhemoglobin concentration. A 2D array of shape (nRegions, simulationLength).
    - U: Input matrix. Shape: (nRegions, simulationLength).
    - A: Connectivity matrix. Shape: (nRegions, nRegions).

    Returns:
    - Y: Optic response. A 2D array of shape (2 * nRegions, simulationLength).
    - dh: Change in total hemoglobin concentration for all brain regions.
    - dq: Change in deoxyhemoglobin concentration for all brain regions.
    """

    # Get the number of brain regions and the length of the simulation
    nRegions = A.shape[0]
    simulationLength = U.shape[1]

    # Define standard optics parameters
    N = [0.65, 71, 2]
    P0 = N[1]
    base_hbr = N[1] * (1 - N[0])

    # Initialize variables to store the changes in concentrations and the optical response for each brain region
    dq = np.zeros((nRegions, simulationLength))
    dp = np.zeros((nRegions, simulationLength))
    dh = np.zeros((nRegions, simulationLength))
    Y = np.zeros((2 * nRegions, simulationLength))

    # Coefficients for computing the optic response
    F_P = np.array(
        [
            (0.0007358251 * 7.5, 0.001104715 * 6.5),
            (0.001159306 * 7.5, 0.0007858993 * 6.5),
        ]
    )

    # Compute the optical response for each time point and for all brain regions
    for t in range(simulationLength):
        dp[:, t] = (pj[:, t] - 1) * P0
        dq[:, t] = (qj[:, t] - 1) * base_hbr
        dh[:, t] = dp[:, t] - dq[:, t]

        for r in range(nRegions):
            dhq = np.array([dq[r, t], dh[r, t]])
            Y_r = F_P @ dhq
            Y[2 * r : 2 * r + 2, t] = Y_r

    return Y, dh, dq


def calculate_hemoglobin_changes(pj, qj):
    """
    Calculate the changes in hemoglobin concentrations.

    Parameters:
    - pj: Total hemoglobin concentration. A 2D array of shape (nRegions, simulationLength).
    - qj: Deoxyhemoglobin concentration. A 2D array of shape (nRegions, simulationLength).

    Returns:
    - dq: Change in deoxyhemoglobin concentration for all brain regions.
    - dh: Change in total hemoglobin concentration for all brain regions.
    - dp: Change in total hemoglobin concentration for all brain regions.
    """
    nRegions, simulationLength = pj.shape
    N = [0.65, 71, 2]
    P0 = N[1]
    base_hbr = N[1] * (1 - N[0])

    dq = np.zeros((nRegions, simulationLength))
    dp = np.zeros((nRegions, simulationLength))
    dh = np.zeros((nRegions, simulationLength))

    for t in range(simulationLength):
        dp[:, t] = (pj[:, t] - 1) * P0
        dq[:, t] = (qj[:, t] - 1) * base_hbr
        dh[:, t] = dp[:, t] - dq[:, t]

    return dq, dh


def compute_optical_response(dq, dh):
    """
    Compute the optics based on the bilinear model for multiple brain regions.

    Parameters:
    - dq: Change in deoxyhemoglobin concentration. A 2D array of shape (nRegions, simulationLength).
    - dh: Change in total hemoglobin concentration. A 2D array of shape (nRegions, simulationLength).
    - A: Connectivity matrix. Shape: (nRegions, nRegions).

    Returns:
    - Y: Optic response. A 2D array of shape (2 * nRegions, simulationLength).
    """
    nRegions, simulationLength = dq.shape
    Y = np.zeros((2 * nRegions, simulationLength))

    F_P = np.array(
        [
            (0.0007358251 * 7.5, 0.001104715 * 6.5),
            (0.001159306 * 7.5, 0.0007858993 * 6.5),
        ]
    )

    for t in range(simulationLength):
        for r in range(nRegions):
            dhq = np.array([dq[r, t], dh[r, t]])
            Y_r = F_P @ dhq
            Y[2 * r : 2 * r + 2, t] = Y_r

    return Y
