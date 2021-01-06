import os
import numpy as np
from ahrs.common.orientation import *
from ahrs.common import DEG2RAD

import pandas as pd

from ahrs.filters.ekf import EKF

app_root = os.path.dirname(os.path.dirname(__file__))
print(app_root)


class EkfData(object):
    def __init__(self, acc, gyr, mag, num_samples):
        self.num_samples = num_samples
        self.acc = acc
        self.gyr = gyr
        self.mag = mag
        self.in_rads = True

if __name__ == '__main__':

    data = np.genfromtxt(app_root + '/tests/repoIMU.csv', dtype=float, delimiter=';', skip_header=2)

    df_data = pd.DataFrame(data, columns=['sec', 'vor_w', 'vor_x', 'vor_y', 'vor_z', 'acc_x', 'acc_y', 'acc_z', "gyro_x", 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z'])
    print(df_data)

    q_ref = data[:, 1:5]    # the rquat (w,x,y,z) data from valcon
    acc = data[:, 5:8]      # acce from IMU
    gyr = data[:, 8:11]     # gyro from IMU
    mag = data[:, 11:14]    # magn from IMU
    num_samples = data.shape[0]

    print("num_samples", num_samples)

    # Estimate Orientations with IMU
    #q_imu = np.tile([1., 0., 0., 0.], (num_samples, 1))
    #ei = EkfData(acc, gyr, mag, num_samples)
    #ekf = EKF(ei, noises=[0.0001, 0.0002, 0.0003])
    #q_imu = ekf.Q
    #for i in range(1, num_samples):
    #    q_imu[i] = madgwick.updateIMU(q_imu[i-1], gyr[i], acc[i])

    # Estimate Orientations with IMU
    q_imu = np.tile([1., 0., 0., 0.], (num_samples, 1))
    ekf = EKF(frequency=100, beta=0.1)
    for i in range(1, num_samples):
        q_imu[i] = ekf.update(q_imu[i - 1], gyr[i], acc[i], mag[i])

    # Estimate Orientations with MARG
    q_marg = np.tile([1., 0., 0., 0.], (num_samples, 1))
    #madgwick = Madgwick()
    #for i in range(1, num_samples):
    #    q_marg[i] = madgwick.updateMARG(q_marg[i-1], gyr[i], acc[i], mag[i])

    # Compute Error
    sqe_imu = abs(q_ref - q_imu).sum(axis=1)**2
    sqe_marg = abs(q_ref - q_marg).sum(axis=1)**2

    # Plot results
    from ahrs.utils import plot
    plot.plot(data[:, 1:5], q_imu, q_marg, [sqe_imu, sqe_marg],
        title="ELK's algorithm",
        subtitles=["Reference Quaternions", "Estimated Quaternions (IMU)", "Estimated Quaternions (MARG)", "Squared Errors"],
        yscales=["linear", "linear", "linear", "log"],
        labels=[[], [], [], ["MSE (IMU) = {:.3e}".format(sqe_imu.mean()), "MSE (MARG) = {:.3e}".format(sqe_marg.mean())]])
