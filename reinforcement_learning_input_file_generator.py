import pandas as pd

from load_symbol import LoadYahooSymbol
from data_aggregator import DataAggregator
from data_scaler import DataScaler
from line_printer import LinePrinter
from file_utility import FileUtility
from default_input_data_for_all_classes import DefaultInputData
from default_input_data_for_all_classes import Constants
from tqdm import tqdm


class RLInputGenerator:
    def __init__(self, default_data_dictionary):
        self.line_printer = LinePrinter()

        self.default_data_loader = DefaultInputData(**default_data_dictionary)
        self.default_constants = Constants()

        self.line_printer.print_text('file_utility')
        self.file_utility = FileUtility(**self.default_data_loader.load_default_file_utility())
        self.line_printer.print_text('yahoo_loader')
        self.yahoo_loader = LoadYahooSymbol(**self.default_data_loader.load_default_load_symbol())
        self.line_printer.print_text('aggregator')
        self.aggregator = DataAggregator(**self.default_data_loader.load_default_data_aggregator())
        self.line_printer.print_text('scaler')
        self.scaler = DataScaler(**self.default_data_loader.load_default_data_scaler())

    def generate_files(self):
        folder_list = self.file_utility.load_all_sub_directories()

        min_data_length_to_store_data = self.default_data_loader.sentence_length + \
                                        self.default_data_loader.length_to_look_into_future_for_rewards + \
                                        self.default_constants.min_number_of_rows_to_aggregate_data

        for folder_counter in tqdm(range(len(folder_list))):
            file_list = self.file_utility.load_file_names_in_directory(folder_list[folder_counter])
            folder_name = folder_list[folder_counter] + '/'

            row_statistics = {'file_name': [],
                              'row_counter': [],
                              'sentence_counter': []}

            for file_counter in tqdm(range(len(file_list))):
                file_name = file_list[file_counter]
                symbol = file_name.replace("." + self.default_data_loader.file_formats_to_load, '')
                file_data = self.yahoo_loader.load_file(
                    self.file_utility.source_data_path + "/" + folder_name, file_name)

                row_statistics['file_name'].append(file_name)
                row_statistics['row_counter'].append(len(file_data))
                if len(file_data) > min_data_length_to_store_data:
                    raw_data = self.aggregator.aggregate_symbol_data(file_data)

                    # if we don't aggregate any useful information, then skip processing data
                    if (len(raw_data)) > 0:
                        file_data.name = symbol
                        scaled_data = self.scaler.generate_scaled_data(raw_data)
                        self.file_utility.save_data(scaled_data, folder_list[folder_counter], symbol)
                        row_statistics['row_counter'].append(len(scaled_data))
                        row_statistics['sentence_counter'].append(len(scaled_data))
                    else:
                        row_statistics['sentence_counter'].append(0)
                else:
                    row_statistics['sentence_counter'].append(0)

                if (file_counter + 1) % self.default_constants.statistic_save_frequency == 0:
                    self.file_utility.save_data(pd.DataFrame(row_statistics), '', folder_list[folder_counter] +
                                                "_" + str(file_counter + 1))

            row_statistics['sentence_length'] = self.default_data_loader.sentence_length
            row_statistics['future_length'] = self.default_data_loader.length_to_look_into_future_for_rewards
            row_statistics['min_number_of_rows_to_aggregate_data'] = \
                min_data_length_to_store_data
            self.file_utility.save_data(pd.DataFrame(row_statistics), '', folder_list[folder_counter])
