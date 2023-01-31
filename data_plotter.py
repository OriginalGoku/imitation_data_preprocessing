from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from data_normalizer import DataNormalizer
import numpy as np
import os


class Plotter:
    def __init__(self, mean_length, plot_title, save_plots=True, verbose=False, clip=True,
                 percentage_for_normalization=0.75,
                 clip_max=1.5, clip_min=-0.5, bin_size=0.1):
        """
        This function saves/plots the data
        :param mean_length: length of the mean for plotting data mean
        :param plot_title: The title used on top of all the plots
        :param save_plots: if True, it will only save the plots (not display them)
        :param verbose:
        :param clip:
        :param percentage_for_normalization:
        :param clip_max:
        :param clip_min:
        :param bin_size:
        """
        self.percentage_for_normalization = percentage_for_normalization
        self.mean_length = mean_length
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.bin_size = bin_size
        self.data_normalizer = DataNormalizer(clip, percentage_for_normalization, clip_max, clip_min, bin_size)
        self.verbose = verbose
        self.save_plots = save_plots
        self.plot_title = plot_title

    def generate_normalized_mean_data(self, source_):  # , start_position_, end_position_):
        # Originl value
        orig_val_ = source_  # .iloc[start_position_:end_position_]
        # Original Normalized
        orig_normalized = self.data_normalizer.normalize_data(orig_val_, self.verbose)
        # Original Mean
        orig_mean = orig_val_.rolling(self.mean_length).mean()
        # Original Mean Normalized
        orig_mean_normalized = self.data_normalizer.normalize_data(orig_mean.dropna(), self.verbose)
        # Original-Mean
        orig_minus_mean = orig_val_ - orig_mean
        orig_minus_mean.dropna(inplace=True)
        # Original-Mean Normalized
        orig_minus_mean_normalized = self.data_normalizer.normalize_data(orig_minus_mean, self.verbose)

        return orig_val_, orig_normalized, orig_mean, orig_mean_normalized, orig_minus_mean, orig_minus_mean_normalized

    def plot_values(self, source, path_to_save_image, file_name: str, x_step_size=20, line_width=2):
        """
        :param line_width: chart line width
        :param source: original source data for plotting. this must be a column of data where the index is a
        pandas date/time object
        :param path_to_save_image: folder name where the image should be saved. This is not the entire path. The first
        part of the path should be passed as a class parameter during initialization
        :param file_name: file name for the information being plotted
        :param path_to_save_image: the path where the file is located. This is useful for finding the file
        :param x_step_size: step size for plotting the data on the x axis (date)
        """
        # fig, ax = plt.subplots(6, 2, figsize=(50, 40))
        fig, ax = plt.subplots(5, 2, figsize=(50, 40))
        plt.rcParams['lines.linewidth'] = line_width

        start_date = np.datetime_as_string(source.index.values[0], unit='D')
        end_date = np.datetime_as_string(source.index.values[-1], unit='D')

        statistics = "Clipping range from " + str(self.clip_min) + " to " + str(
            self.clip_max) + "\nPercentage of Data used for Normalization: %" + str(
            100 * self.percentage_for_normalization) + "\nBin Size: " + str(self.bin_size)

        full_plot_name = self.plot_title + 'for File Name: ' + file_name + '\n' + statistics + '\n' + start_date + \
                         " to " + end_date
        fig.suptitle(full_plot_name)
        original, original_normalized, original_mean, original_mean_normalized, original_minus_mean, \
        original_minus_mean_normalized = self.generate_normalized_mean_data(source)
        original_normalized_round = self.data_normalizer.my_round(original_normalized, 2)
        original_mean_normalized_round = self.data_normalizer.my_round(original_mean_normalized, 2)
        original_minus_mean_normalized_round = self.data_normalizer.my_round(original_minus_mean_normalized, 2)
        values_to_plot = [original, original_normalized, original_normalized_round,
                          original_mean, original_mean_normalized, original_mean_normalized_round,
                          original_minus_mean, original_minus_mean_normalized, original_minus_mean_normalized_round]

        plot_titles = ["Original Data", "Original Normalized Data", "Original Normalized Data Round",
                       "Original Mean", "Original Mean Normalized", "Original Mean Normalized Round",
                       "Residual (Original-Mean)", "Residual (Original-Mean) Normalized",
                       "Residual (Original-Mean) Normalized Round"]
        counter = 0

        # First 3 rows plotting original data and mean side by side
        colors = ['green', 'purple']
        for col in range(2):
            for row in range(3):
                # print(col, " - ", row, ' -> Counter: ', plot_titles[counter])
                ax[row][col].plot(values_to_plot[counter], color=colors[col])
                start, end = ax[row][col].get_xlim()
                ax[row][col].xaxis.set_ticks(np.arange(start, end, x_step_size))
                # todo: fix this part so the date is shown only as month and year or month and date
                # ax[0][0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%m-%d'))
                ax[row][col].set_title(plot_titles[counter])
                counter += 1

        # Drawing Residuals
        residual_row = [3, 3, 4]
        residual_col = [0, 1, 1]

        for coordinate in range(3):
            ax[residual_row[coordinate]][residual_col[coordinate]].plot(values_to_plot[counter], color='red')
            start, end = ax[residual_row[coordinate]][residual_col[coordinate]].get_xlim()
            ax[residual_row[coordinate]][residual_col[coordinate]].xaxis.set_ticks(np.arange(start, end, x_step_size))
            # todo: fix this part so the date is shown only as month and year or month and date
            # ax[0][0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%m-%d'))
            ax[residual_row[coordinate]][residual_col[coordinate]].set_title(plot_titles[counter])
            counter += 1

        # Drawing RMSE's in one chart
        # 1->2
        # 4->5
        # 7->8
        rms = []
        for i in range(1, 8, 3):
            rms.append(self.data_normalizer.calculate_rmse(values_to_plot[i], values_to_plot[i + 1]))
        # rms = self.data_normalizer.calculate_rmse(values_to_plot[counter], values_to_plot[counter - 1])
        rms_locations = [1, 2, 3]
        ax[4][0].vlines(x=rms_locations, ymin=[0, 0, 0], ymax=rms, colors='teal', ls='--', lw=2,
                        label='RMSE Original, Mean, Original-Mean')
        trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)

        ax[4][0].text(0.0, 1.0, 'RMSE [Original: '+str(round(rms[0],4))+' - Mean: '+str(round(rms[1],4))+
                      '  -  Original-Mean: '+str(round(rms[0],4))+']', transform=ax[4][0].transAxes + trans,
                      fontsize='medium', verticalalignment='top', fontfamily='serif',
                      bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

        # ax[row][col].hlines(rms, 0, 1, colors='C1', linestyles='dashed', label='RMSE')
        # ax[row][col].set_title("Root Mean Square Error for Rounded Data")

        # Original working Code:
        # counter = 0
        # for row in range(6):
        #     for col in range(2):
        #         if row % 2 == 0:
        #             ax[row][col].plot(values_to_plot[counter])
        #             start, end = ax[row][col].get_xlim()
        #             ax[row][col].xaxis.set_ticks(np.arange(start, end, x_step_size))
        #             # todo: fix this part so the date is shown only as month and year or month and date
        #             # ax[0][0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%m-%d'))
        #             ax[row][col].set_title(plot_titles[counter])
        #             counter += 1
        #         else:
        #             if col == 1:
        #                 ax[row][col].plot(values_to_plot[counter])
        #                 start, end = ax[row][col].get_xlim()
        #                 ax[row][col].xaxis.set_ticks(np.arange(start, end, x_step_size))
        #                 # ax[0][0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%m-%d'))
        #                 ax[row][col].set_title(plot_titles[counter])
        #                 counter += 1
        #             else:
        #                 rms = self.data_normalizer.calculate_rmse(values_to_plot[counter], values_to_plot[counter - 1])
        #                 ax[row][col].hlines(rms, 0, 1, colors='C1', linestyles='dashed', label='RMSE')
        #                 ax[row][col].set_title("Root Mean Square Error for Rounded Data")

        fig.tight_layout()

        if not os.path.isdir(path_to_save_image):
            os.makedirs(path_to_save_image)

        fig.savefig(path_to_save_image + file_name + '.png')
        if self.save_plots:
            plt.close()
