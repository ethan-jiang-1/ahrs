import os
import numpy as np
from ahrs.common.orientation import *
from ahrs.common import DEG2RAD
import pandas as pd

from ahrs.filters.mahony import Mahony

app_root = os.path.dirname(os.path.dirname(__file__))
print(app_root)


if __name__ == '__main__':

    data = np.genfromtxt(app_root + '/tests/repoIMU.csv',
                         dtype=float,
                         delimiter=';',
                         skip_header=2)

    df_data = pd.DataFrame(data,
                           columns=[
                               'sec', 'vor_w', 'vor_x', 'vor_y', 'vor_z',
                               'acc_x', 'acc_y', 'acc_z', "gyro_x", 'gyro_y',
                               'gyro_z', 'mag_x', 'mag_y', 'mag_z'
                           ])
    print(df_data)

    q_ref = data[:, 1:5]
    acc = data[:, 5:8]
    gyr = data[:, 8:11]
    mag = data[:, 11:14]
    num_samples = data.shape[0]

    print("num_samples", num_samples)

    # Estimate Orientations with IMU
    q_imu = np.tile([1., 0., 0., 0.], (num_samples, 1))
    mahony = Mahony()
    for i in range(1, num_samples):
        q_imu[i] = mahony.updateIMU(q_imu[i - 1], gyr[i], acc[i])


    # Estimate Orientations with MARG
    q_marg = np.tile([1., 0., 0., 0.], (num_samples, 1))
    mahony = Mahony()
    for i in range(1, num_samples):
        q_marg[i] = mahony.updateMARG(q_marg[i - 1], gyr[i], acc[i], mag[i])

    # Compute Error
    sqe_imu = abs(q_ref - q_imu).sum(axis=1)**2
    sqe_marg = abs(q_ref - q_marg).sum(axis=1)**2

    # Plot results
    from ahrs.utils import plot    
    plot.plot(data[:, 1:5],
         q_imu,
         q_marg, [sqe_imu, sqe_marg],
         title="Mahony's algorithm",
         subtitles=[
             "Reference Quaternions", "Estimated Quaternions (IMU)",
             "Estimated Quaternions (MARG)", "Squared Errors"
         ],
         yscales=["linear", "linear", "linear", "log"],
         labels=[[], [], [],
                 [
                     "MSE (IMU) = {:.3e}".format(sqe_imu.mean()),
                     "MSE (MARG) = {:.3e}".format(sqe_marg.mean())
                 ]])
