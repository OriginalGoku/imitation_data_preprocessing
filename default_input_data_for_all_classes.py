from dataclasses import dataclass



class DefaultInputData:
    def __init__(self, sentence_length, length_to_look_into_future_for_rewards, data_columns,
                 tolerance_percent_from_0_to_take_a_trade, min_volatility_for_explosive_move_filtering,
                 source_data_path, save_destination_path, include_volatility_in_output, generate_binary_output,
                 bin_size=0.1, default_column_name='close', default_date_column_name='date', debug_folder_path='Debug',
                 file_formats_to_load='csv', file_format_to_save='csv', frequency_for_saving_statistics=500,
                 rounding_precision=2, volatility_precision=0.01, debug_mode=False, verbose=False):
        # common
        self.sentence_length = sentence_length
        self.data_columns = data_columns
        self.rounding_precision = rounding_precision
        self.debug_mode = debug_mode
        self.debug_folder_path = debug_folder_path
        self.verbose = verbose
        self.frequency_for_saving_statistics = frequency_for_saving_statistics

        # data_aggregatos:
        self.length_to_look_into_future_for_rewards = length_to_look_into_future_for_rewards
        self.min_volatility_for_explosive_move_filtering = min_volatility_for_explosive_move_filtering
        self.default_column_name = default_column_name
        self.date_column_name = default_date_column_name
        self.include_volatility_in_output = include_volatility_in_output
        self.generate_binary_output = generate_binary_output
        self.tolerance_percent_from_0_to_take_a_trade = tolerance_percent_from_0_to_take_a_trade
        self.min_volatility_for_explosive_move_filtering = min_volatility_for_explosive_move_filtering

        # data_scaler
        self.bin_size = bin_size
        self.volatility_precision = volatility_precision

        # file_utility
        self.source_data_path = source_data_path
        self.save_destination_path = save_destination_path
        self.file_formats_to_load = file_formats_to_load
        self.file_format_to_save = file_format_to_save

    def load_default_data_aggregator(self):
        return {
            'sentence_length': self.sentence_length,
            'length_to_look_into_future_for_rewards': self.length_to_look_into_future_for_rewards,
            'min_volatility_for_explosive_move_filtering': self.min_volatility_for_explosive_move_filtering,
            'include_volatility_in_output': self.include_volatility_in_output,
            'price_column_name': self.default_column_name,
            'date_column_name': self.date_column_name,
            'rounding_precision': self.rounding_precision,
            'tolerance_percent_from_0_to_take_a_trade': self.tolerance_percent_from_0_to_take_a_trade,
            'generate_binary_output' : self.generate_binary_output,
            'debug_mode': self.debug_mode,
            'debug_folder_path': self.debug_folder_path,
            'verbose': self.verbose
        }

    def load_default_data_scaler(self):
        return {'sentence_length': self.sentence_length,
                'data_columns': self.data_columns,
                'bin_size': self.bin_size,
                'volatility_precision': self.volatility_precision,
                'rounding_precision': self.rounding_precision,
                'verbose': self.verbose

                # 'debug_mode': self.debug_mode,
                # 'debug_folder_path': self.debug_folder_path
                }

    def load_default_load_symbol(self):
        return {'column_name_dictionary': {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                                           'Volume': 'volume'},
                'date_index_new_name': self.date_column_name,
                'verbose': self.verbose}

    def load_default_file_utility(self):
        return {'source_data_path': self.source_data_path,
                'save_destination_path': self.save_destination_path,
                'file_formats_to_load': self.file_formats_to_load,
                'file_format_to_save': self.file_format_to_save,
                'verbose': self.verbose
                }


@dataclass(unsafe_hash=True)
class Constants:
    min_number_of_rows_to_aggregate_data = 10
    action_dictionary = {'EXPLOSIVE_SHORT': -3,
                         'GOOD_SHORT': -2,
                         'SHORT': -1,
                         'MIX': 0,
                         'LONG': 1,
                         'GOOD_LONG': 2,
                         'EXPLOSIVE_LONG': 3}
    trade_definition_dictionary = {-3: 'Buy Put',
                                   -2: 'Short/Buy stock',
                                   -1: 'Sell Call',
                                   0: 'Flat',
                                   1: 'Sell Put',
                                   2: 'Buy stock',
                                   3: 'Buy Call'
                                   }
    statistic_save_frequency = 500
