#!/usr/bin/env python
# Author: Matias Mattamala
# Description: Numpy-based kalman filter with oulier rejection
# This implementation is stateless, it doesn't store the estimates
# This is the same used in Wild Visual Navigation 
# https://github.com/leggedrobotics/wild_visual_navigation/blob/main/wild_visual_navigation/utils/kalman_filter.py
#
# Dependencies
#  python3 -m venv env
#  source activate env/bin/activate
#  pip install numpy matplotlib

import numpy as np


class KalmanFilter:
    def __init__(
        self,
        dim_state: int = 1,
        dim_control: int = 1,
        dim_meas: int = 1,
        outlier_rejection: str = "none",
        outlier_delta: float = 1.0,
    ):
        super().__init__()

        # Store dimensions
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._dim_meas = dim_meas

        # Prediction model
        self._proc_model = np.eye(dim_state)
        self._proc_cov = np.eye(dim_state)
        self._control_model = np.eye(dim_state, dim_control)

        # Measurement model
        self._meas_model = np.eye(dim_meas, dim_state)
        self._meas_cov = np.eye(dim_meas, dim_meas)

        # Outlier rejection
        self._outlier_rejection = outlier_rejection
        self._outlier_delta = outlier_delta
    
    def _get_outlier_weight(self, error: np.ndarray, cov: np.ndarray):
        if self._outlier_rejection != "none":
            # Compute residual
            r = np.sqrt(error.T @ np.linalg.inv(cov) @ error)

            # Apply outlier rejection strategy
            if self._outlier_rejection == "hard":
                weight = 0.0 if r.item() >= self._outlier_delta else 1.0

            elif self._outlier_rejection == "huber":
                # Prepare Huber loss
                abs_r = np.abs(r)
                weight = 1.0 if abs_r <= self._outlier_delta else (self._outlier_delta / abs_r).item()
                return weight
            else:
                print(f"Invalid option outlier_rejection [{self._outlier_rejection}]. Ignore (w = 1.0)")
                return 1.0
        else:
            return 1.0
    
    def _check_dim(self, reference, matrix):
        assert (
                matrix.shape == reference.shape
            ), f"Dimensions do not match {matrix.shape} != {reference.shape}"


    def set_process_model(self, proc_model=None, proc_cov=None, control_model=None):
        # Initialize process model
        if proc_model is not None:
            self._check_dim(self._proc_model, proc_model)
            self._proc_model = proc_model

        # Initialize process model covariance
        if proc_cov is not None:
            self._check_dim(self._proc_cov, proc_cov)
            self._proc_cov = proc_cov

        # Initialize control
        if control_model is not None:
            self._check_dim(self._control_model, control_model)
            self._control_model = control_model

    def set_meas_model(self, meas_model=None, meas_cov=None):
        # Initialize measurement model
        if meas_model is not None:
            self._check_dim(self._meas_model, meas_model)
            self._meas_model = meas_model

        # Initialize measurement model covariance
        if meas_cov is not None:
            self._check_dim(self._meas_cov, meas_cov)
            self._meas_cov = meas_cov

    def prediction(self, state: np.ndarray, state_cov: np.ndarray, control: np.ndarray = None):
        # Update prior
        if control is None:
            state = self._proc_model @ state
        else:
            state = self._proc_model @ state + self._control_model @ control

        # Update covariance
        state_cov = self._proc_model @ state_cov @ self._proc_model.T + self._proc_cov

        return state, state_cov

    def correction(self, state: np.ndarray, state_cov: np.ndarray, meas: np.ndarray):
        # Innovation
        innovation = meas - self._meas_model @ state

        # Get outlier rejection weight
        outlier_weight = self._get_outlier_weight(innovation, self._meas_cov)

        # Innovation covariance
        innovation_cov = self._meas_model @ state_cov @ self._meas_model.T + self._meas_cov

        # Kalman gain
        kalman_gain = outlier_weight * state_cov @ self._meas_model.T @ np.linalg.inv(innovation_cov)

        # Updated state
        state = state + (kalman_gain @ innovation)
        state_cov = (np.eye(self._dim_state) - kalman_gain @ self._meas_model) @ state_cov

        return state, state_cov

    def step(self, state, state_cov, meas, control=None):
        state, state_cov = self.prediction(state, state_cov, control)
        state, state_cov = self.correction(state, state_cov, meas)
        return state, state_cov


def run_kalman_filter():
    """Tests Kalman Filter"""

    import matplotlib.pyplot as plt

    # Normal KF
    kf1 = KalmanFilter(dim_state=1, dim_control=1, dim_meas=1, outlier_rejection="none")
    kf1.set_process_model(proc_model=np.eye(1) * 1, proc_cov=np.eye(1) * 0.5)
    kf1.set_meas_model(meas_model=np.eye(1), meas_cov=np.eye(1) * 1)

    # Outlier-robust KF
    kf2 = KalmanFilter(
        dim_state=1,
        dim_control=1,
        dim_meas=1,
        outlier_rejection="huber",
        outlier_delta=0.2,
    )
    kf2.set_process_model(proc_model=np.eye(1) * 1, proc_cov=np.eye(1) * 0.5)
    kf2.set_meas_model(meas_model=np.eye(1), meas_cov=np.eye(1) * 1)

    N = 300
    # Default signal
    t = np.linspace(0, 10, N)
    T = 5
    x = np.sin(t * 2 * np.pi / T)

    # Noises
    # Salt and pepper
    salt_pepper_noise = np.random.rand(N)
    min_v = 0.05
    max_v = 1.0 - min_v
    salt_pepper_noise[salt_pepper_noise >= max_v] = 1.0
    salt_pepper_noise[salt_pepper_noise <= min_v] = -1.0
    salt_pepper_noise[np.logical_and(salt_pepper_noise > min_v, salt_pepper_noise < max_v)] = 0.0
    # White noise
    white_noise = np.random.rand(N) / 2

    # Add noise to signal
    x_noisy = x + salt_pepper_noise + white_noise

    # Arrays to store the predictions
    x_e = np.zeros((2, N))
    x_cov = np.zeros((2, N))

    # Initial estimate and covariance
    e = np.array([0])
    cov = np.array([0.1])

    # Initial value
    x_e[0, 0] = x_e[1, 0] = e.item()
    x_cov[0, 0] = x_cov[1, 0] = cov.item()

    # Run
    for i in range(1, N):
        # Get sample
        s = x_noisy[i]

        # Get new estimate and cov
        e1, cov1 = kf1.step(x_e[0, i - 1][None], x_cov[0, i - 1][None], s[None])
        e2, cov2 = kf2.step(x_e[1, i - 1][None], x_cov[1, i - 1][None], s[None])

        # Save predictions
        x_e[0, i] = e1.item()
        x_cov[0, i] = cov1.item()
        x_e[1, i] = e2.item()
        x_cov[1, i] = cov2.item()

    # Plot
    plt.plot(t, x_noisy, label="Noisy signal", color="k")
    plt.plot(t, x_e[0], label="Filtered", color="r")
    plt.plot(t, x_e[1], label="Filtered w/OR", color="b")
    # plt.plot(t, x_cov[1], label="Cov - w/outlier rejection", color="g")
    plt.fill_between(
        t,
        x_e[0] - x_cov[0],
        x_e[0] + x_cov[0],
        alpha=0.3,
        label="Confidence bounds (1$\sigma$)",
        color="r",
    )
    plt.fill_between(
        t,
        x_e[1] - x_cov[1],
        x_e[1] + x_cov[1],
        alpha=0.3,
        label="Confidence bounds w/OR(1$\sigma$)",
        color="b",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Signal [adimensional]")
    plt.title("Kalman Filter with and without Outlier Rejection (OR)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_kalman_filter()