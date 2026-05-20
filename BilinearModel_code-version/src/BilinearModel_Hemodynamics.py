import numpy as np


def Hemodynamics_Legacy(Z, P_SD, Step):
    """
    Simulate the hemodynamic response for multiple brain regions using a bilinear model.

    Parameters:
    - Z: Neural activity. A 2D array of shape (nRegions, simulationLength) where each row represents a brain region and each column represents a time point.
    - P_SD: Parameters for each region. A 2D array where each row represents a different parameter and each column represents a brain region.
    - Step: Time step for the Euler method.

    Returns:
    - qj: Deoxyhemoglobin concentration. A 2D array of shape (nRegions, simulationLength) where each row represents a brain region and each column represents a time point.
    - pj: Total hemoglobin concentration. A 2D array of shape (nRegions, simulationLength) where each row represents a brain region and each column represents a time point.
    """

    # Determine the number of brain regions and simulation length
    nRegions, simulationLength = Z.shape

    # Initialization of hemodynamic state variables for each of the nRegions over the simulation time

    # Vasodilatory Signal variable for all regions
    Sj = np.full((nRegions, simulationLength), np.nan)
    Sj[:, 0] = np.zeros(nRegions)

    # Rate of blood volume for all regions
    Vj = np.zeros((nRegions, simulationLength))
    Vj[:, 0] = np.exp(np.zeros(nRegions))

    # HbT concentration (Rate) for all regions
    pj = np.full((nRegions, simulationLength), np.nan)
    pj[:, 0] = np.exp(np.zeros(nRegions))

    # HbR concentration (Rate) for all regions
    qj = np.full((nRegions, simulationLength), np.nan)
    qj[:, 0] = np.exp(np.zeros(nRegions))

    # Inflow for all regions
    fjin = np.full((nRegions, simulationLength), np.nan)
    fjin[:, 0] = np.exp(np.zeros(nRegions))
    fjout_s = np.full((nRegions, simulationLength), np.nan)
    fjout_s1 = np.full((nRegions, simulationLength), np.nan)

    # Define constants for the hemodynamic model. These are standard values in the literature.
    H = [0.64, 0.32, 2.00, 0.32, 0.32, 2.00]

    # Extract and adjust parameters for each of the nRegions

    Kj = H[0] * np.exp(np.concatenate(([0], [P_SD[0, 0] for _ in range(nRegions - 1)])))
    Yj = H[1] * np.exp(np.concatenate(([0], [P_SD[1, 0] for _ in range(nRegions - 1)])))
    Tj = H[2] * np.exp(np.concatenate(([0], [P_SD[2, 0] for _ in range(nRegions - 1)])))
    Tjv = H[5] * np.exp(
        np.concatenate(([0], [P_SD[3, 0] for _ in range(nRegions - 1)]))
    )
    phi = H[3]

    # Hemodynamics simulation for each time point using the Euler method for all regions
    for t in range(1, simulationLength):
        # Compute changes in hemodynamic state variables across all regions
        Sj_dot = Z[:, t - 1] - Kj * Sj[:, t - 1] - Yj * (fjin[:, t - 1] - 1)
        fjin_dot = Sj[:, t - 1]

        fv_s = Vj[:, t - 1] ** (1 / phi)
        Vj_dot = (fjin[:, t - 1] - fv_s) / (Tj * Tjv * Vj[:, t - 1])
        fjout = fv_s + Tjv * Vj_dot
        Efp = (1 - (1 - H[4]) ** (1 / fjin[:, t - 1])) / H[4]
        qj_dot = ((fjin[:, t - 1] * Efp - fjout * qj[:, t - 1]) / Vj[:, t - 1]) / (
            Tj * qj[:, t - 1]
        )
        pj_dot = (fjin[:, t - 1] - (fjout * pj[:, t - 1]) / Vj[:, t - 1]) / Tj

        # Update the state variables for the next time point for all regions using the Euler method
        Sj[:, t] = Sj[:, t - 1] + Step * Sj_dot
        Vj[:, t] = Vj[:, t - 1] + Step * Vj_dot
        fjin[:, t] = fjin[:, t - 1] + Step * fjin_dot
        qj[:, t] = qj[:, t - 1] + Step * qj_dot
        pj[:, t] = pj[:, t - 1] + Step * pj_dot
        fjout_s[:, t] = fjout
        fjout_s1[:, t] = Efp

    return qj, pj


def Hemodynamics(Z, P_SD, Step):
    """
    Simulate the hemodynamic response for multiple brain regions using a bilinear model.
    Corrected to match Tak et al. (2015) and verified against SPM source code.
    """

    # Determine the number of brain regions and simulation length
    nRegions, simulationLength = Z.shape

    # Initialization
    # Vasodilatory Signal variable for all regions
    Sj = np.full((nRegions, simulationLength), np.nan)
    Sj[:, 0] = np.zeros(nRegions)

    # Rate of blood volume (normalized, starts at 1.0)
    Vj = np.zeros((nRegions, simulationLength))
    Vj[:, 0] = np.exp(np.zeros(nRegions))  # exp(0) = 1

    # HbT concentration (normalized, starts at 1.0)
    pj = np.full((nRegions, simulationLength), np.nan)
    pj[:, 0] = np.exp(np.zeros(nRegions))

    # HbR concentration (normalized, starts at 1.0)
    qj = np.full((nRegions, simulationLength), np.nan)
    qj[:, 0] = np.exp(np.zeros(nRegions))

    # Inflow (normalized, starts at 1.0)
    fjin = np.full((nRegions, simulationLength), np.nan)
    fjin[:, 0] = np.exp(np.zeros(nRegions))

    fjout_s = np.full((nRegions, simulationLength), np.nan)
    fjout_s1 = np.full((nRegions, simulationLength), np.nan)

    # Define constants (H array in the MATLAB code)
    # H = [decay, autoreg, transit, alpha, E0, viscoelastic]
    H = [0.64, 0.32, 2.00, 0.32, 0.32, 2.00]

    # Extract and adjust parameters (P_SD corresponds to P in MATLAB)
    # Note: Parameters are often estimated in log-space, so we exp() them here.
    Kj = H[0] * np.exp(np.concatenate(([0], [P_SD[0, 0] for _ in range(nRegions - 1)])))
    Yj = H[1] * np.exp(np.concatenate(([0], [P_SD[1, 0] for _ in range(nRegions - 1)])))
    Tj = H[2] * np.exp(np.concatenate(([0], [P_SD[2, 0] for _ in range(nRegions - 1)])))
    Tjv = H[5] * np.exp(
        np.concatenate(([0], [P_SD[3, 0] for _ in range(nRegions - 1)]))
    )

    alpha = H[3]  # Grubb's exponent
    rho = H[4]  # Resting oxygen extraction fraction (E0)

    # Hemodynamics simulation
    for t in range(1, simulationLength):
        # 1. Signal Decay (s_dot)
        Sj_dot = Z[:, t - 1] - Kj * Sj[:, t - 1] - Yj * (fjin[:, t - 1] - 1)

        # 2. Inflow (f_dot)
        # Matches Tak Eq 2: f_dot = s
        fjin_dot = Sj[:, t - 1]

        # 3. Volume (v_dot)
        # Steady state outflow component
        fv_s = Vj[:, t - 1] ** (1 / alpha)

        # (linear space).
        Vj_dot = (fjin[:, t - 1] - fv_s) / (Tj + Tjv)

        # Viscoelastic outflow (Tak Eq 5)
        fjout = fv_s + Tjv * Vj_dot

        # Oxygen extraction fraction E(f,rho)
        Efp = (1 - (1 - rho) ** (1 / fjin[:, t - 1])) / rho

        # 4. Deoxy-hemoglobin (q_dot)
        # Tak Eq 4: Tau * q_dot = f_in * E/rho - f_out * q/v
        qj_dot = ((fjin[:, t - 1] * Efp) - (fjout * qj[:, t - 1] / Vj[:, t - 1])) / Tj

        # 5. Total hemoglobin (p_dot) (Assuming p = v based on standard simplifications or Tak derivation)
        # Tak Eq 3 (second part): Tau * p_dot = (f_in - f_out) * p/v
        pj_dot = (fjin[:, t - 1] - fjout) * (pj[:, t - 1] / Vj[:, t - 1]) / Tj

        # Update state variables (Euler integration)
        Sj[:, t] = Sj[:, t - 1] + Step * Sj_dot
        Vj[:, t] = Vj[:, t - 1] + Step * Vj_dot
        fjin[:, t] = fjin[:, t - 1] + Step * fjin_dot
        qj[:, t] = qj[:, t - 1] + Step * qj_dot
        pj[:, t] = pj[:, t - 1] + Step * pj_dot

        fjout_s[:, t] = fjout
        fjout_s1[:, t] = Efp

    return qj, pj


def HemoglobinConcentrations(qj, pj):
    deltaQ = (qj - 1) * (71 * (1 - 0, 65))
    deltaP = (pj - 1) * 71
    deltaH = deltaP - deltaQ

    return deltaH, deltaQ
