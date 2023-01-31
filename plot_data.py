import numpy as np
import pandas as pd
from load_symbol import LoadYahooSymbol
import os
from statistic_generator import StatisticGenerator
from line_printer import LinePrinter
from data_plotter import Plotter
import random
from tqdm import tqdm
import timeit

MIN_DATA_VOLATILITY = 0.1
ROUNDING_PRECISION = 2
STATISTIC_SAVING_INTERVAL = 50
# This parameter is used for memory management. If the system loads more than 2000 files in a folder, it will save
# the statistics of the files inside the folder in batches of 2000 files
MAXIMUM_STATISTIC_FILE_SIZE = 1000


class PlotData:
    def __init__(self, percentage_of_data_to_plot: float, use_different_path_to_save_data: bool, mean_length: int,
                 column_name_for_plotting, plot_title, min_no_of_sentences_for_integrity_check,
                 sentence_length: int, future_length: int, use_sentence_length_for_data_splitting: bool,
                 data_splits_for_plotting=3,
                 save_path=None, path="Yahoo_Stock", file_formats_to_load='csv', save_plots=True, verbose=False,
                 clip=True, percentage_for_normalization=0.75, clip_max=1.5, clip_min=-0.5, bin_size=0.1):
        """
        This function plots a percentage of the entire data and saves them in their own directory.
        :param percentage_of_data_to_plot: % of total data to plot. maximum is 1
        :data_splits_for_plotting: this parameter determines how many parts data is split for printing. a good number
        should be 2 or 3
        :param path:
        :param file_formats_to_load:
        :param sentence_length: The actual size of input (number of bars) we want to give to our learning system
        :param future_length: The number of bars to look into future for generating results
        :param use_sentence_length_for_data_splitting: if True, it will use sentence length to plot the data.
        Also, if true, then the percentage_for_normalization will be calculated as:
        (sentence_length)/(sentence_length+future_length)
        """

        if use_sentence_length_for_data_splitting:
            percentage_for_normalization = sentence_length / (sentence_length + future_length)

        if "/" != path[-1]:
            raise Exception("Path name must end with /")

        if percentage_of_data_to_plot > 1:
            raise ValueError("percentage_of_data_to_plot must be less than 1")

        plot_save_path = path
        if use_different_path_to_save_data:
            if not save_path:
                raise Exception("Have to provide a save path if use_different_path_to_save_data is True")
            elif not os.path.isdir(save_path):
                os.makedirs(save_path)
            plot_save_path = save_path

        if not os.path.isdir(path):
            raise FileNotFoundError(path + " was not Not Found")

        self.path = path
        self.file_formats_to_load = file_formats_to_load
        self.percentage_of_data_to_plot = percentage_of_data_to_plot
        self.folder_list = self.load_all_sub_directories()

        self.plotter = Plotter(mean_length, plot_title, save_plots=save_plots, verbose=verbose, clip=clip,
                               percentage_for_normalization=percentage_for_normalization, clip_max=clip_max,
                               clip_min=clip_min, bin_size=bin_size)

        self.file_loader = LoadYahooSymbol()
        self.data_splits_for_plotting = data_splits_for_plotting
        if use_different_path_to_save_data:
            self.save_path = plot_save_path
        else:
            self.save_path = self.path
        self.line_printer = LinePrinter("-")
        self.column_name_for_plotting = column_name_for_plotting
        self.min_no_of_sentences_for_integrity_check = min_no_of_sentences_for_integrity_check
        statistics_input = {'open_column_name': 'open', 'high_column_name': 'high', 'low_column_name': 'low',
                            'close_column_name': 'close', 'volume_column_name': 'volume',
                            'min_data_volatility': MIN_DATA_VOLATILITY,
                            'sentence_length': sentence_length, 'future_data_length': future_length,
                            'default_column_name': 'close',
                            'min_number_of_sentences': min_no_of_sentences_for_integrity_check,
                            'rounding_precision': ROUNDING_PRECISION}
        self.statistic_generator = StatisticGenerator(**statistics_input)
        self.sentence_length = sentence_length
        self.future_length = future_length
        self.use_sentence_length_for_data_splitting = use_sentence_length_for_data_splitting
    # todo: use file_utility function instead
    # def load_all_sub_directories(self):
    #     all_folders = os.listdir(self.path)
    #     folder_list = []
    #
    #     for file in all_folders:
    #         if os.path.isdir(self.path + file):
    #             folder_list.append(file)
    #
    #     return folder_list

    # todo: use file_utility function instead
    # def get_file_names_in_directory(self, dir_):
    #     """
    #     Get all files in a directory with a specific extension specified at class level (self.file_formats_to_load)
    #
    #     :param dir_: the directory to check
    #     :return: [] of files with self.file_formats_to_load extension in the specified directory
    #     """
    #     all_files = os.listdir(self.path + dir_)
    #     file_list = []
    #     print(self.path + dir_)
    #     for file in all_files:
    #         if os.path.isfile(self.path + dir_ + "/" + file):
    #             if file.split('.')[-1] == self.file_formats_to_load:
    #                 file_list.append(file)
    #     return file_list

    def plotter_all_data(self):

        for i in tqdm(range(len(self.folder_list))):
            # Have to reset the statistic results per folder to preserve memory.
            # todo: Change from a simple array to a numpy array since numpy arrays are more momory efficient
            statistic_results = []#np.array([])
            folder = self.folder_list[i]
            # exchange_country_name = folder.split('/')[-2]
            files_in_directory = self.get_file_names_in_directory(folder)
            number_of_files = len(files_in_directory)

            if ((number_of_files - 1) > int(number_of_files * self.percentage_of_data_to_plot)):
                print("number_of_files - 1: ", number_of_files - 1)
                print("int(number_of_files * self.percentage_of_data_to_plot): ",
                      int(number_of_files * self.percentage_of_data_to_plot))
                list_of_file_ids_for_plotting = random.sample(range(0, number_of_files - 1),
                                                              int(number_of_files * self.percentage_of_data_to_plot))
            else:
                list_of_file_ids_for_plotting = np.arange(number_of_files - 1)

            print(list_of_file_ids_for_plotting)
            # self.line_printer.print_line()
            for file_counter in tqdm(range(number_of_files)):
                load_file_path = self.path + folder + "/"
                save_file_path = self.save_path + folder + "/"
                file_name = files_in_directory[file_counter]
                file_data = self.file_loader.load_file(load_file_path, file_name)

                singe_file_statistics = {}
                singe_file_statistics['integrity_check'] = self.statistic_generator. \
                    check_integrity_of_data(file_data, load_file_path, file_name, file_name.replace('.csv', '', ),
                                            folder)

                file_data_length = len(file_data)

                if self.use_sentence_length_for_data_splitting:
                    date_step_size = self.sentence_length + self.future_length
                    number_of_chunks = file_data_length // date_step_size
                    print("Each part of plot has ", date_step_size, ' Bars')

                else:
                    date_step_size = file_data_length // self.data_splits_for_plotting
                    number_of_chunks = self.data_splits_for_plotting
                    print("Each part of plot has ", date_step_size, ' Bars')

                for chunk_id in tqdm(range(number_of_chunks)):
                    start_range = chunk_id * date_step_size
                    end_range = chunk_id * date_step_size + date_step_size
                    chunk_data = file_data.iloc[start_range:end_range]

                    # Statistics
                    print('Loading Statistics for ', file_name, ' part ', chunk_id)
                    singe_file_statistics[
                        'part_' + str(chunk_id + 1)] = self.statistic_generator.generate_chunk_statistics(
                        chunk_data)

                    usability_result = singe_file_statistics['part_' + str(chunk_id + 1)]['good_for_trading']

                    print("Plotting Part ", chunk_id, " of ", file_name)
                    usability_text = ''
                    if not usability_result:
                        usability_text = usability_text + 'NOT_'
                    usability_text = usability_text + 'Usable_'

                    plot_title = "Part_" + str(chunk_id + 1) + "_" + usability_text + str(date_step_size) + \
                                 "_bars_" + file_name
                    if singe_file_statistics['integrity_check']['has_enough_data'] and \
                            (file_counter in list_of_file_ids_for_plotting):
                        self.plotter.plot_values(chunk_data[self.column_name_for_plotting], save_file_path, plot_title,
                                                 x_step_size=len(chunk_data) // 10)

                print('Loading Statistics for ', file_name, ' Total')
                singe_file_statistics['total'] = self.statistic_generator.generate_chunk_statistics(file_data)

                # statistic_results = np.append(statistic_results,
                #                               self.statistic_generator.flatten(singe_file_statistics))

                statistic_results.append(self.statistic_generator.flatten(singe_file_statistics))
                if (file_counter+1) % MAXIMUM_STATISTIC_FILE_SIZE ==0:
                    self.save_statistic_data(statistic_results, folder + "_" + str(file_counter) + "_.csv")
            # if (all_data_counter + 1) % STATISTIC_SAVING_INTERVAL == 0:
            # self.line_printer.print_line(text=str(all_data_counter + 1))
            # self.save_statistic_data(statistic_results, str(i) + '_Statistics.csv')
            # some folders might not have any files
            if number_of_files != 0:
                self.save_statistic_data(statistic_results, folder+".csv")
            # We could not save all the data into a single dictionary and then save it all at the same time because we
            # ran out of memory and the program crashed. So had to save each folder's data separately and then clear
            # the dictionary memory
            # statistic_results.clear()

        # self.save_statistic_data(statistic_results, '0_Final_Statistics.csv')

    def save_statistic_data(self, statistic_results, file_name):
        # try:
        result_df = pd.DataFrame(statistic_results)
        result_df.rename(columns={'integrity_check_symbol': 'symbol', 'integrity_check_file_name': 'file_name',
                                  'integrity_check_file_path': 'file_path', 'integrity_check_exchange': 'exchange'},
                         inplace=True)
        # print("result_df.column: ", result_df.columns)
        # print("result_df.iloc[0]", result_df.iloc[0])
        result_df.set_index(result_df['symbol'], drop=True, inplace=True)
        result_df.drop('symbol', axis=1, inplace=True)
        print("saving Statistics result ", file_name, " in ", self.path)
        result_df.to_csv(self.path + file_name)
    # except:
    #     print("could not save statistics_results")


mode = 'train'
if mode == 'test':
    _percentage_of_data_to_plot = 1
    data_path = "test_data/"
    save_path = 'test_charts/'
else:
    _percentage_of_data_to_plot = 0.05
    data_path = "Yahoo_Finance_Stock_Data/"
    save_path = 'charts/'

plot_parameters = {'percentage_of_data_to_plot': _percentage_of_data_to_plot,
                   'use_different_path_to_save_data': True,
                   'mean_length': 20,
                   'column_name_for_plotting': 'close',
                   'plot_title': 'Comparing Price and Normalized Price',
                   'min_no_of_sentences_for_integrity_check': 10,
                   'sentence_length': 128,
                   'future_length': 32,
                   'use_sentence_length_for_data_splitting': False,
                   'data_splits_for_plotting': 3,
                   'save_path': save_path,
                   'path': data_path,
                   'file_formats_to_load': 'csv',
                   'save_plots': True,
                   'verbose': False,
                   'clip': True,
                   'percentage_for_normalization': 0.75,
                   'clip_max': 1.5,
                   'clip_min': -0.5,
                   'bin_size': 0.1
                   }

starting_time = timeit.default_timer()

my_plotter = PlotData(**plot_parameters)

print("Start time :", starting_time)
my_plotter.plotter_all_data()
print("Time difference :", timeit.default_timer() - starting_time)

# WORK for tomorrow:
# Add max_date gap for the data
# Also add if volume data is available
