from reinforcement_learning_input_file_generator import RLInputGenerator
import pandas as pd

import timeit

if __name__ == '__main__':
    sentence_len = 31
    future_len = 8

    data_col = ['-' + str(i) for i in range(sentence_len - 1, 0, -1)]
    data_col.append('0')

    print(len(data_col))

    parameters_dictionary = {'sentence_length': sentence_len,
                             'length_to_look_into_future_for_rewards': future_len,
                             'data_columns': data_col,
                             'include_volatility_in_output': True,
                             'generate_binary_output': True,
                             'tolerance_percent_from_0_to_take_a_trade': 0.00,
                             'min_volatility_for_explosive_move_filtering': 0.1,
                             # 'source_data_path': 'Yahoo_Finance_Stock_Data',
                             # 'source_data_path': '../Data_Source/Yahoo/Original_Data',
                             'source_data_path': '../Data_Source/Yahoo/Original_Data/ETF',
                             # 'save_destination_path': '../Data_Source/Yahoo/Processed_Yahoo_Data/Stock_Binary_tolerance_half_std',
                             'save_destination_path': '../Data_Source/Yahoo/Processed_Yahoo_Data/ETF_Binary',
                             'bin_size': 0.05,
                             'default_column_name': 'close',
                             'default_date_column_name': 'date',
                             'debug_folder_path': 'Debug',
                             'file_formats_to_load': 'csv',
                             'file_format_to_save': 'csv',
                             'frequency_for_saving_statistics': 500,
                             'rounding_precision': 2,
                             'volatility_precision': 0.01,
                             'debug_mode': False,
                             'verbose': True}

    rl_generator = RLInputGenerator(parameters_dictionary)
    starting_time = timeit.default_timer()
    print("Start time :", starting_time)
    rl_generator.generate_files()
    print("Time difference :", timeit.default_timer() - starting_time)
