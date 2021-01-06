import os
import numpy as np
from ahrs.common.orientation import triad
from ahrs.common.quaternion import sarabandi

from ahrs.filters.angular import AngularRate

app_root = os.path.dirname(os.path.dirname(__file__))
print(app_root)

if __name__ == "__main__":
    from ahrs.utils import plot
    data = np.genfromtxt(app_root + '/tests/repoIMU.csv',
                         dtype=float,
                         delimiter=';',
                         skip_header=2)
    q_ref = data[:, 1:5]
    acc = data[:, 5:8]
    mag = data[:, 11:14]
    num_samples = data.shape[0]
    # Estimate Orientations with IMU
    q = np.tile([1., 0., 0., 0.], (num_samples, 1))
    for i in range(num_samples):
        dcm = triad(acc[i], mag[i])
        q[i] = sarabandi(dcm)
    # Compute Error
    sqe = abs(q_ref - q).sum(axis=1)**2
    # Plot results
    plot(q_ref,
         q,
         sqe,
         title="TRIAD estimation",
         subtitles=[
             "Reference Quaternions", "Estimated Quaternions", "Squared Errors"
         ],
         yscales=["linear", "linear", "log"],
         labels=[[], [], ["MSE = {:.3e}".format(sqe.mean())]])
