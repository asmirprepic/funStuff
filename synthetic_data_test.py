
import numpy as np

def generate_synthetic_data(n_samples=1000, n_outliers=50):
    rng = np.random.RandomState(42)
    # Generate normal data: for example, normal operating metrics
    X_normal = 0.3 * rng.randn(n_samples, 2) + 2
    X_normal = np.r_[X_normal, 0.3 * rng.randn(n_samples, 2) - 2]
    # Add outliers: anomalies
    X_outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))
    X = np.vstack([X_normal, X_outliers])
    return X
