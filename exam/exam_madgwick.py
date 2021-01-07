import os
import numpy as np
from ahrs.common.orientation import *
from ahrs.common import DEG2RAD

import pandas as pd

from ahrs.filters.madgwick import Madgwick

app_root = os.path.dirname(os.path.dirname(__file__))
print(app_root)


class AhrsData(object):
    def __init__(self, acc, gyr, mag, num_samples):
        self.acc = acc
        self.gyr = gyr
        self.mag = mag
        self.num_samples = num_samples

        self.q_ref = None
        self.in_rads = True
        self.frequency = 100


def _load_ahrs_data_repoIMU():
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

    q_ref = data[:, 1:5]  # the rquat (w,x,y,z) data from valcon
    acc = data[:, 5:8]  # acce from IMU
    gyr = data[:, 8:11]  # gyro from IMU
    mag = data[:, 11:14]  # magn from IMU
    num_samples = data.shape[0]


    ad = AhrsData(acc, gyr, mag, num_samples)
    ad.q_ref = q_ref
    ad.frequency = 100

    return ad


def _get_filter_madgwick(**kwargs):
    madgwick = Madgwick(kwargs)
    return madgwick


def _get_ahrs_plot():
    from ahrs.utils import plot
    return plot.plot


def _plot_result_2(ad, q_est0, q_est1, sqe_mse0, sqe_mse1, mark, est0_name, est1_name):
    q_ref = ad.q_ref

    # Plot results
    plot = _get_ahrs_plot()
    plot(q_ref, q_est0, q_est1, [sqe_mse0, sqe_mse1],
        title="".format(mark),
        subtitles=["Reference Quaternions", "Estimated Quaternions ({})".format(est0_name), "Estimated Quaternions ({})".format(est1_name), "Squared Errors"],
        yscales=["linear", "linear", "linear", "log"],
        labels=[[], [], [], ["MSE({}) = {:.3e}".format(est0_name, sqe_mse0.mean()), "MSE({}) = {:.3e}".format(est1_name, sqe_mse1.mean())]])



def _run_estimate(ad):
    q_ref = ad.q_ref
    acc = ad.acc
    gyr = ad.gyr
    mag = ad.mag
    num_samples = ad.num_samples
    frequency= ad.frequency 

    # Estimate Orientations with IMU
    q_imu = np.tile([1., 0., 0., 0.], (num_samples, 1))
    filter = _get_filter_madgwick(frequency=frequency,beta=0.1)
    for i in range(1, num_samples):
        q_imu[i] = filter.updateIMU(q_imu[i-1], gyr[i], acc[i])

    # Estimate Orientations with MARG
    q_marg = np.tile([1., 0., 0., 0.], (num_samples, 1))
    filter = _get_filter_madgwick(frequency=frequency, beta=0.1)
    for i in range(1, num_samples):
        q_marg[i] = filter.updateMARG(q_marg[i-1], gyr[i], acc[i], mag[i])

    # Compute Error
    sqe_imu = abs(q_ref - q_imu).sum(axis=1)**2
    sqe_marg = abs(q_ref - q_marg).sum(axis=1)**2

    _plot_result_2(ad, q_imu, q_marg, sqe_imu, sqe_marg, "Madwick", "IMU", "MARG")


if __name__ == '__main__':
    ad = _load_ahrs_data_repoIMU()
    _run_estimate(ad)