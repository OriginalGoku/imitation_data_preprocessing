from line_printer import LinePrinter
import pandas as pd
import numpy as np
from sklearn import preprocessing


class DataScaler:
    def __init__(self, sentence_length: int, data_columns: [str], bin_size: float, volatility_precision=0.01,
                 rounding_precision = 4,
                 verbose=False):
        """
        This class generates scaled data based on the scaler chosen. At the moment we have hard coded min_max_scaler
        # todo: provide options for other type of scalers
        :param sentence_length: the sentence length of our data
        :param data_columns: the actual price/volume column names. the length of this field must be equal to sentence length
        :param bin_size: bin size to round the data
        # todo: do more research on how to set the volatility rounding number automatically.
        :param volatility_precision: precision level used for volatility data
        """

        if len(data_columns) != sentence_length:
            raise ValueError("len(data_columns) must be equal to sentence_length")
        self.sentence_length = sentence_length
        self.bin_size = bin_size
        self.data_columns = data_columns
        self.volatility_precision = volatility_precision
        self.verbose = verbose
        self.rounding_precision = rounding_precision
        self.line_printer = LinePrinter()

    def numpy_rounder(self, input_):
        return self.bin_size * np.round(input_ / self.bin_size)

    def volatility_rounder(self, input_, bin_scale):
        return bin_scale * np.round(input_ / bin_scale)

    def generate_scaled_data(self, data_to_be_scaled):

        if self.verbose:
            print('Generating scaled data for: ', data_to_be_scaled.name)
        my_scaler = self.numpy_rounder(preprocessing.minmax_scale(data_to_be_scaled[self.data_columns], axis=1))
        my_scaler = np.round(my_scaler, self.rounding_precision)

        scaled_data = pd.DataFrame(index=data_to_be_scaled.index)
        scaled_data[self.data_columns] = my_scaler
        scaled_data['volatility'] = np.round(self.volatility_rounder(data_to_be_scaled['volatility'],
                                                                     self.volatility_precision),self.rounding_precision)
        scaled_data[['action', 'reward']] = data_to_be_scaled[['action', 'reward']]

        return scaled_data

# clean_data = pd.read_csv('debug_folder/SPLV_From_Csv_aggregated_data_after_integrity_check.csv', index_col=[0],
#                          parse_dates=True)
#
# folder = 'Test_Data/'
# sentence_length = 7
# data_columns = clean_data.columns[:sentence_length]
# bin_size = 0.1
# volatility_precision = 0.001
#
# scale_file = DataScaler(sentence_length, data_columns, bin_size, volatility_precision = volatility_precision)
# # print(clean_data)
# print(scale_file.generate_scaled_data(clean_data.iloc[0:8]))
