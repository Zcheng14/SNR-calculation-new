import numpy as np
import settings


class DataTextFile():
    def __init__(self, path):
        self.data = np.loadtxt(path)

    def interpolate(self, x):
        data_interpolated = np.interp(x, self.data[:, 0], self.data[:, 1])
        return data_interpolated


class TwoDataTextFiles():
    def __init__(self, path1, path2):
        self.data1 = DataTextFile(path1).data
        self.data2 = DataTextFile(path2).data
        self.combined_data = self.__combine_two_files()

    def __combine_two_files(self):
        blue_data_interpolated = np.interp(settings.default_blue_wavelengths, self.data1[:, 0], self.data1[:, 1])
        red_data_interpolated = np.interp(settings.default_red_wavelengths, self.data2[:, 0], self.data2[:, 1])
        return np.concatenate((blue_data_interpolated, red_data_interpolated), axis=None)

    def interpolate(self, x):
        data_x = np.concatenate((settings.default_blue_wavelengths, settings.default_red_wavelengths), axis=None)
        return np.interp(x, data_x, self.combined_data)


class SpotSizeAndVignettingFiles(DataTextFile):
    def interpolate(self, wavelengths):
        data = np.zeros((len(settings.field_points), len(wavelengths)))
        data_x = np.concatenate((settings.default_blue_wavelengths, settings.default_red_wavelengths), axis=None)
        for i in range(len(settings.field_points)):
            data[i] = np.interp(wavelengths, data_x, self.data[i])
        return data


def load_AR_coating(x):
    blue_all_data = []
    for path in settings.blue_AR_coating_paths:
        blue_range = np.arange(452, 521)
        data = np.loadtxt(path)
        data = np.interp(blue_range, data[:, 0], data[:, 1])
        blue_all_data.append(data)
    blue_mean_data = np.mean(blue_all_data, axis=0)


    red_all_data = []
    for path in settings.red_AR_coating_paths:
        red_range = np.arange(610, 701)
        data = np.loadtxt(path)
        data = np.interp(red_range, data[:, 0], data[:, 1])
        red_all_data.append(data)
    red_mean_data = np.mean(red_all_data, axis=0)

    mean_data = np.concatenate((blue_mean_data, red_mean_data), axis=None)
    data_x = np.concatenate((blue_range, red_range), axis=None)

    mean_data_interpolated = np.interp(x, data_x, mean_data)
    return 1 - mean_data_interpolated * 0.01
