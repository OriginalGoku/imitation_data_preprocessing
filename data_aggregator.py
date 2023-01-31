import pandas as pd
import numpy as np
import warnings
import os

from line_printer import LinePrinter
from default_input_data_for_all_classes import Constants


class DataAggregator:

    def __init__(self, sentence_length: int, length_to_look_into_future_for_rewards: int,
                 min_volatility_for_explosive_move_filtering: float, include_volatility_in_output,
                 generate_binary_output,
                 price_column_name: str, date_column_name: str, rounding_precision: int, debug_folder_path,
                 percentage_of_same_data_in_a_row_to_keep_the_data=0.2, time_difference_between_trades=None,
                 tolerance_percent_from_0_to_take_a_trade=0, debug_mode=False, verbose=False):

        # todo: add a filter for penny stocks

        """

        This class provides tools to aggregate historic and future data into a single row.
        It also sets action and rewards for the entire dataset. This system caps reward for explosive moves to the volatility
        level of the last explosive_move_look_back_period bars.
        # todo: It might be worth studying scenarios where optimal exit points can be determined rather than just
        # capping the profit based on volatility of the last few bars or the close of the last bar of length_to_look_into_future_for_rewards

        :param sentence_length: the length of historic data to aggregate into one row
        :param length_to_look_into_future_for_rewards: number of bars to look into the future for calculating rewards and actions.
        this calculation start at the next bar and go all the way up to and including length_to_look_into_future_for_rewards
        :param min_volatility_for_explosive_move_filtering: this is the minimum volatility that the system will consider before categorizing the next movement as explosive
        This is used to fileter situation where the symbol moves in a very narrow band and then makes a move that is still very small but compared to the narrow band it might look large
        :param include_volatility_in_output: if True, it will include one column in the output dataframe for volatility of the observed state
        :param generate_binary_output: if True, it will generate a binary output for actions. if the closing price of
        the last column of future dataframe is above the tolerance_percent_from_0_to_take_a_trade action will be 1
        otherwise, action is 0
        :param date_column_name: column name for date
        :param price_column_name: which column will be used for all the calculations (usually it is close)
        :param rounding_precision: precision level to store the data
        :param tolerance_percent_from_0_to_take_a_trade: This value determines how much the price can move above and below 0 for it to be still considered for action -1 and 1
        by default this value is set at 0 meaning that for +1 and -1 action, the price must stay absolutely above and below 0 respectively.
        Must be very careful with this value and its advised to keep it below 0.01
        :param min_volatility_to_accept_row_of_data_for_final_output: This value determines if a row of aggregated data will be kept or disregarded.
        This is the final output of this class and the min_std_to_accept_row_of_data_for_final_output is used to weed out data that has extremely low movement over the entire sentence
        The min std value of LOW Volatility S&P500 SPLV was 0.1 and thats why the default value was set to 0.09 to make sure a low volatility index can also be accepted
        :param time_difference_between_trades: this value is used to calculate how frequent trades can be made for the statistics generated from each file.
        The calculation is as follow:
        once a trade is entered, the system will not take any trades for the next number of time_difference_between_trades
        if left None, the system will use length_to_look_into_future_for_rewards as default.
        It is advised to leave this unchanged
        :param debug_folder_path: this format should not include a / at the end
        :param debug_mode: if debug_mode is True, then all dataframes will be saved in the debug_folder

        :return: # todo: make it optional to include volatility in the output of this file

        """
        if sentence_length % 2 == 0:
            warnings.warn("Your sentence length is " + str(
                sentence_length) + ". Please note that the output of this class always has volatility as a column in "
                                   "the output. If the output of this class is going to be used for systems such as "
                                   "BERT, it is advised to set the sentence length to " + str(sentence_length - 1))

        if tolerance_percent_from_0_to_take_a_trade > 0.01:
            warnings.warn("You have set tolerance_percent_from_0_to_take_a_trade to "
                          + str(tolerance_percent_from_0_to_take_a_trade) +
                          " This value means that price could move more than" +
                          str(tolerance_percent_from_0_to_take_a_trade) +
                          "% and the system will still take a -1 or +1 trade accordingly. It is highly advised to "
                          "keep this value below 0.01")

        if not time_difference_between_trades:
            self.time_difference_between_trades = length_to_look_into_future_for_rewards
        else:
            self.time_difference_between_trades = time_difference_between_trades

        if debug_folder_path[-1] == '/':
            debug_folder_path = debug_folder_path[:-1]

        if not os.path.exists(debug_folder_path):
            os.makedirs(debug_folder_path)

        self.debug_folder = debug_folder_path + "/"
        self.debug_mode = debug_mode
        self.sentence_length = sentence_length
        self.length_to_look_into_future_for_rewards = length_to_look_into_future_for_rewards
        self.min_volatility_for_explosive_move_filtering = min_volatility_for_explosive_move_filtering
        self.include_volatility_in_output = include_volatility_in_output
        self.generate_binary_output = generate_binary_output
        self.rounding_precision = rounding_precision
        self.price_column_name = price_column_name
        self.date_column_name = date_column_name
        self.line_printer = LinePrinter()
        self.tolerance_percent_from_0_to_take_a_trade = tolerance_percent_from_0_to_take_a_trade
        self.percentage_of_same_data_in_a_row_to_keep_the_data = percentage_of_same_data_in_a_row_to_keep_the_data
        self.action_dictionary = Constants().action_dictionary
        self.verbose = verbose

    def gather_history(self, data_column):
        """
        :param data_column: dataframe column to collect last self.sentence_length bars. The data must have date as index
        :return: a new dataframe where each row is self.sentence_length and the index is self.date_column_name
        """
        # todo: check that _data has date as index

        # Transpose data
        sequence_data = data_column.T.tolist()
        the_end_ = len(sequence_data) - self.sentence_length + 1
        chunks = [sequence_data[x:x + self.sentence_length] for x in range(0, the_end_)]

        # Make a dataframe from the chunks
        chunk_df = pd.DataFrame(chunks)
        # Reverse the order of the columns name to place them from oldest to newest
        chunk_df.columns = chunk_df.columns[::-1]
        # Add a negative sign to the column names to indicate how many bars back we got the data from
        # this method adds a - sign to columns 0 (the current row's column)
        # so we rename it to 0 after
        chunk_df = chunk_df.add_prefix("-")
        chunk_df.rename(columns={"-0": "0"}, inplace=True)

        # in order to get the dates back, we merge the chunks with the original data Since chunk data only starts
        # after self.sentence_length bars, then we only merge _data starting at that position and -1 is to include
        # the current row in rese_index we keep drop=False since in our data, index is the date and we don't want to
        # lose that information
        merged = pd.merge(data_column[self.sentence_length - 1:].
                          reset_index(drop=False), chunk_df, right_index=True, left_index=True)
        merged.rename(columns={'index': 'date'}, inplace=True)
        # Drop the column name since the historic data at column 0 is the same as the current row's column_name data

        merged.drop(self.price_column_name, axis=1, inplace=True)
        merged.set_index('date', inplace=True, drop=True)
        print(merged.columns)

        return merged, round(merged.std(axis=1) / merged.mean(axis=1), self.rounding_precision)

    def calculate_future_returns(self, data_column):
        """
        This function calculates the return for the number of bars into the future
        :param data_column: the columns of the dataframe to calculate future returns

        """
        future_returns = pd.DataFrame(index=data_column.index)
        # Calculates return for the next number_of_bars_into_the_future
        for row in range(1, self.length_to_look_into_future_for_rewards + 1):
            col_ = str(row) + '_bar_into_future'
            future_returns[col_] = round((data_column.shift(-row) -
                                          data_column) /
                                         data_column,
                                         self.rounding_precision)
        future_returns.index.name = 'date'

        return future_returns

    def calculate_volatility(self, historic_data):
        """
        initially we were using the formula (max-min)/min but this value might be much larger than 1 and then we need
        to normalize it again.
        Eventually, we decided to change the denominator to max in order to keep the value between 0 and 1
        :param historic_data: the historic dataframe generate by self.gather_history()
        :return: (max_ - min_) / max_
        """
        min_ = historic_data.min(axis=1, numeric_only=True)
        max_ = historic_data.max(axis=1, numeric_only=True)
        vol = round((max_ - min_) / max_, self.rounding_precision)

        vol_df = pd.DataFrame(vol)
        vol_df.columns = ['volatility']
        vol_df.index.name = 'date'
        return vol_df

    def set_actions_rewards_for_explosive_moves(self, future_high_prices, future_low_prices, volatility):

        """
        :param volatility: the volatility of the historic data
        :param future_low_prices: the low price column of future values. each row of data must have self.length_to_look_into_future_for_rewards columns
        :param future_high_prices: the high price column of future values. each row of data must have self.length_to_look_into_future_for_rewards columns

        """
        high = future_high_prices.copy()
        low = future_low_prices.copy()

        # todo: optimize this code by making it modular (so we can use same code for both short and long)
        # todo: Check a crazy scenario where the price could hit both positive and negative explosive target

        # Calculating actions for the long side
        high['max'] = high.max(axis=1)

        high['volatility'] = volatility * (1 - self.tolerance_percent_from_0_to_take_a_trade)

        # calculate appropriate masks

        # Mask for when future high prices are less than the volatility target but price movement is above the minimum volatility
        long_min_volatility_hit_mask = (high['max'] > self.min_volatility_for_explosive_move_filtering)

        # Mask for when future high prices hit the volatility target and the volatility target is above the minimum volatility
        long_volatility_target_hit_mask = (
                (high['max'] > high['volatility'].to_numpy()) & (
                high['volatility'].to_numpy() > self.min_volatility_for_explosive_move_filtering))

        # Set appropriate actions
        high.loc[long_min_volatility_hit_mask, 'action'] = int(self.action_dictionary['GOOD_LONG'])
        high.loc[long_volatility_target_hit_mask, 'action'] = int(self.action_dictionary['EXPLOSIVE_LONG'])

        # Set appropriate rewards
        high.loc[long_min_volatility_hit_mask, 'reward'] = self.min_volatility_for_explosive_move_filtering
        high.loc[long_volatility_target_hit_mask, 'reward'] = round(
            high.loc[long_volatility_target_hit_mask, 'volatility'], self.rounding_precision)

        # Calculating actions for the long side
        low['min'] = low.min(axis=1)

        low['volatility'] = - volatility * (1 - self.tolerance_percent_from_0_to_take_a_trade)

        # calculate appropriate masks

        # Mask for when future high prices are less than the volatility target but price movement is above the
        # minimum volatility
        short_min_volatility_hit_mask = (low['min'] < -self.min_volatility_for_explosive_move_filtering)

        # Mask for when future high prices hit the volatility target and the volatility target is above the minimum
        # volatility The writing of the volatility>min_volatility_for_explosive_move_filtering looks similar to the
        # code for comparing long_volatility_target_hit_mask but it is correct since
        # -volatility<-min_volatility_for_explosive_move_filtering =
        # volatility>min_volatility_for_explosive_move_filtering
        short_volatility_target_hit_mask = ((low['min'] < low['volatility'].to_numpy())
                                            & (low['volatility'].to_numpy() <
                                               - self.min_volatility_for_explosive_move_filtering))

        # Set appropriate actions
        low.loc[short_min_volatility_hit_mask, 'action'] = int(self.action_dictionary['GOOD_SHORT'])
        low.loc[short_volatility_target_hit_mask, 'action'] = int(self.action_dictionary['EXPLOSIVE_SHORT'])

        # Set appropriate rewards
        # Reward is kept positive since for our reinforcement learning agent, if it detects the short direction
        # correctly, it must receive a positive reward
        low.loc[short_min_volatility_hit_mask, 'reward'] = self.min_volatility_for_explosive_move_filtering
        low.loc[short_volatility_target_hit_mask, 'reward'] = -round(
            low.loc[short_volatility_target_hit_mask, 'volatility'], self.rounding_precision)

        high.fillna(0, inplace=True)
        low.fillna(0, inplace=True)

        final_action = high.action + low.action
        final_reward = high.reward + low.reward

        action_reward = pd.DataFrame([final_action, final_reward]).T
        action_reward.action = action_reward.action.astype(int)

        # for Debugging

        if self.debug_mode:
            low['final_action'] = final_action
            low['final_reward'] = final_reward
            # todo: find a way to name this file in a similar format with other files
            low.join(high, lsuffix='_low', rsuffix='_hig').to_csv(self.debug_folder + '/' + future_low_prices.name +
                                                                  'low_high_join.csv')

        return action_reward

    # def set_actions_rewards_for_high_low(self, future_high_prices, future_low_prices, future_close_prices, volatility):
    #     """
    #     This function sets the primary actions and rewards.
    #     action 1 is set if all prices in the future prices dataframe are positive meaning the price did not go below
    #     our starting price.
    #     action -1 is set if all prices are negative meaning the price did not go above our entry point
    #     action 0 is set if the price crosses our entry price
    #     :param future_prices: is a Dataframe of all the returns calculated by calculate_returns function.
    #     :return: This function modifies the original future_prices dataframe and returns an update dataframe including
    #      action and rewards
    #     """
    #
    #     action_reward_df = pd.DataFrame(index=future_close_prices.index)
    #     # the self.tolerance_percent_from_0_to_take_a_trade is used just in case we want to consider trades for
    #     # the time when price moves very slightly above or below 0
    #     all_positive = np.sign(future_low_prices) >= -self.tolerance_percent_from_0_to_take_a_trade
    #     all_negative = np.sign(future_high_prices) <= self.tolerance_percent_from_0_to_take_a_trade
    #     # Sets the reward to the last bar we are checking in the future
    #     #todo: fix this part so action is selected based on the ACTION definition
    #     # as is, this code sets action = 1 when all future prices are above the self.tolerance_percent_from_0_to_take_a_trade
    #     # and sets action to -1 when all future prices are below self.tolerance_percent_from_0_to_take_a_trade
    #     action_reward_df['action'] = all_positive.all(axis=1).astype(int) - all_negative.all(axis=1).astype(int)
    #     action_reward_df['reward'] = future_close_prices[future_close_prices.columns[-1]]
    #
    #     action_reward_df.loc[action_reward_df['action'] == -1, 'reward'] = \
    #         action_reward_df.loc[action_reward_df['action'] == -1, 'reward'] * -1
    #
    #     # generate a matrix for the positions where future prices have hit the volatility target
    #     ## if price went above volatility, then target is volatility, if price did not go above volatility but it went
    #     ## above minimum volatility then set target at min_volatility
    #
    #
    #
    #
    #
    #     return action_reward_df

    def set_binary_action_rewards(self, future_close_prices, std_ratio, std_multiplier=0.5):
        """
        This function sets actions and rewards based on the closing price of the last column of future_close_price
        if the value on the last col of future_close_prices > - self.tolerance_percent_from_0_to_take_a_trade
        then its long and future_close_prices < self.tolerance_percent_from_0_to_take_a_trade it is short
        # todo: This function needs to also check for the following scenarios
        # tolerance_percent_from_0_to_take_a_trade>price > - tolerance_percent_from_0_to_take_a_trade
        :param future_close_prices: the column of data that contains future close prices
        :param std_multiplier: the factor with which actions will be split.
        action = 1 if future_close_prices['last column'] > std_ratio*std_multiplier
        action = 0 if -std_ratio*std_multiplier < future_close_prices['last column'] <std_ratio*std_multiplier
        action = -1 if future_close_prices['last column'] < -std_ratio*std_multiplier
        :return: df with 'action' and 'reward' column. Reward is the value of the last column of future_close_prices

        """

        action_reward_df = pd.DataFrame(future_close_prices[future_close_prices.columns[-1]])
        action_reward_df.columns = ['last_close']
        action_reward_df['std_ratio'] = pd.DataFrame(std_ratio*std_multiplier, index=future_close_prices.index)


        action_1_map = action_reward_df['last_close'] >= action_reward_df['std_ratio']
        action_minus_1_map = action_reward_df['last_close'] <= - action_reward_df['std_ratio']
        action_0_map = ((- action_reward_df['std_ratio'] < action_reward_df['last_close']) &
                        (action_reward_df['last_close'] < action_reward_df['std_ratio'])
                        )

        action_reward_df.loc[action_1_map, 'action'] = 1
        action_reward_df.loc[action_0_map, 'action'] = 0
        action_reward_df.loc[action_minus_1_map, 'action'] = -1


        # print('action_reward_df[action_reward_df[action]==1): ', action_reward_df[action_reward_df['action'] == 1])
        # print('action_reward_df[action_reward_df[action]==0): ', action_reward_df[action_reward_df['action'] == 0])
        # print('action_reward_df[action_reward_df[action]==-1): ', action_reward_df[action_reward_df['action'] == -1])
        # print("convert type to int")
        # print((action_reward_df['action'].dropna()).astype(int))
        # print("8888888888")

        # print("action_reward before rename and drop")
        # print(action_reward_df)

        action_reward_df.rename(columns={'last_close': 'reward'}, inplace=True)
        action_reward_df.drop('std_ratio', axis=1, inplace=True)
        # action_reward_df['reward'] = future_close_prices[future_close_prices.columns[-1]]
        # print("Final action_reward")
        # print(action_reward_df)
        return action_reward_df

    def set_initial_actions_rewards(self, future_prices):
        """
        This function sets the primary actions and rewards.
        action 1 is set if all prices in the future prices dataframe are positive meaning the price did not go below
        our starting price.
        action -1 is set if all prices are negative meaning the price did not go above our entry point
        action 0 is set if the price crosses our entry price

        :param future_prices: is a Dataframe of all the returns calculated by calculate_returns function.

        :return: a dataframe with action and rewards column and same index as the future_prices
        """

        action_reward_df = pd.DataFrame(index=future_prices.index)
        # the self.tolerance_percent_from_0_to_take_a_trade is used just in case we want to consider trades for
        # the time when price moves very slightly above or below 0
        # if use_tolerance_for_target:
        all_positive = np.sign(future_prices) >= -self.tolerance_percent_from_0_to_take_a_trade
        all_negative = np.sign(future_prices) <= self.tolerance_percent_from_0_to_take_a_trade
        # else:
        #     all_positive = np.sign(future_prices) >= 0
        #     all_negative = np.sign(future_prices) <= 0
        # Sets the reward to the last bar we are checking in the future
        action_reward_df['action'] = all_positive.all(axis=1).astype(int) - all_negative.all(axis=1).astype(int)
        action_reward_df['reward'] = round(future_prices[future_prices.columns[-1]], self.rounding_precision)

        # if the action is -1, reward should still be set as positive so this part changes the sign of reward

        action_reward_df.loc[action_reward_df['action'] == -1, 'reward'] = \
            action_reward_df.loc[action_reward_df['action'] == -1, 'reward'] * -1

        action_reward_df.action = action_reward_df.action.astype(int)

        return action_reward_df

    def set_actions_rewards(self, future_high_prices, future_low_prices, future_close_prices, volatility):
        initial_actions_rewards = self.set_initial_actions_rewards(future_close_prices)

        explosive_actions_rewards = self.set_actions_rewards_for_explosive_moves(future_high_prices, future_low_prices,
                                                                                 volatility)

        explosive_actions_rewards.loc[explosive_actions_rewards['action'] == 0, 'reward'] = \
            initial_actions_rewards.loc[
                explosive_actions_rewards[explosive_actions_rewards['action'] == 0].index, 'reward']
        explosive_actions_rewards.loc[explosive_actions_rewards['action'] == 0, 'action'] = initial_actions_rewards.loc[
            explosive_actions_rewards[explosive_actions_rewards['action'] == 0].index, 'action']

        return explosive_actions_rewards

    def aggregate_symbol_data(self, data_frame):
        if self.verbose:
            print("Aggregating information for ", data_frame.name)

        file_name = data_frame.name
        folder_name = self.debug_folder + file_name
        history, std_ration = self.gather_history(data_frame.close)

        # print("STD: ", std_ration)

        volatility = self.calculate_volatility(history)

        # The return is only calculated from where the first row of historic data
        future_close_returns = self.calculate_future_returns(data_frame.loc[history.index[0]:, 'close'])
        future_close_returns.name = file_name

        if self.generate_binary_output:
            actions_rewards = self.set_binary_action_rewards(future_close_returns, std_ration)
            print(actions_rewards)
            print('len(actions_rewards): ', len(actions_rewards))

        else:
            future_high_returns = self.calculate_future_returns(data_frame.loc[history.index[0]:, 'high'])
            future_low_returns = self.calculate_future_returns(data_frame.loc[history.index[0]:, 'low'])

            future_high_returns.name = file_name
            future_low_returns.name = file_name

            actions_rewards = self.set_actions_rewards(
                future_high_returns,
                future_low_returns,
                future_close_returns, volatility)

        if self.include_volatility_in_output:
            aggregated_data = history.join(volatility)
        else:
            aggregated_data = history

        aggregated_data = aggregated_data.join(actions_rewards)

        aggregated_data.name = file_name

        aggregated_data_clean = self.check_data_integrity(aggregated_data, future_close_returns)

        if self.debug_mode:
            self.line_printer.print_text('Saving debug information for' + folder_name)
            history.to_csv(folder_name + "_history.csv")
            volatility.to_csv(folder_name + '_volatility.csv')
            if not self.generate_binary_output:
                ((future_close_returns.join(future_high_returns, lsuffix="close_", rsuffix='high_')).join(
                    future_low_returns, rsuffix='_low')).to_csv(folder_name + '_future_prices.csv')

            actions_rewards.to_csv(folder_name + '_action_reward.csv')
            aggregated_data.to_csv(folder_name + '_aggregated_data.csv')
            aggregated_data_clean.to_csv(folder_name + '_aggregated_data_after_integrity_check.csv')

        aggregated_data_clean.name = file_name

        if self.verbose:
            print("Total Row of data aggregated: ", len(aggregated_data_clean))

        aggregated_data_clean.dropna(inplace=True)
        aggregated_data_clean['action'] = aggregated_data_clean['action'].astype(int)
        print("8888888")
        print(aggregated_data_clean)
        print("77777777")
        print(shit)

        return aggregated_data_clean

    def filter_static_data(self, data_table, debug_file_name=''):
        """
        This function filters data that doesn't change
        :param debug_file_name: file_name for savin static data debug information
        :param data_table: This data could be the future data or history data
        :return: a column of a dataframe that shows acceptable rows in the data_table
        """

        test_for_static = data_table.T.diff().T

        test_for_static.where(test_for_static == 0, -1, inplace=True)

        test_for_static.replace(0, 1, inplace=True)
        test_for_static.replace(-1, 0, inplace=True)
        test_for_static['sum'] = test_for_static.sum(axis=1)
        test_for_static['is_acceptable'] = (test_for_static['sum'] / self.sentence_length) < \
                                           self.percentage_of_same_data_in_a_row_to_keep_the_data

        if self.debug_mode:
            test_for_static.to_csv(self.debug_folder + debug_file_name + "_filter_static_data.csv")

        if self.verbose:
            print(round(100 * (len(data_table) - len(test_for_static)) / len(data_table), self.rounding_precision),
                  '% of Rows for ' + debug_file_name + ' were dismissed out of total: ', len(data_table))
        return test_for_static['is_acceptable']

    def check_data_integrity(self, aggregated_data, future_close):
        # Check all values are not 0

        data_to_check = aggregated_data[aggregated_data.columns[:self.sentence_length]]
        data_containing_zero = data_to_check[(data_to_check == 0).any(axis=1)]

        data_without_zero = aggregated_data.drop(data_containing_zero.index)
        data_without_zero.name = aggregated_data.name

        # filter static data
        acceptable_df = pd.DataFrame(index=future_close.index)
        acceptable_df['history_is_acceptable'] = self.filter_static_data(
            data_without_zero[data_without_zero.columns[:self.sentence_length]], debug_file_name='_history_check')

        acceptable_df['future_is_acceptable'] = self.filter_static_data(
            data_without_zero[data_without_zero.columns[:self.sentence_length]], debug_file_name='_future_check')

        acceptable_df.loc[
            acceptable_df['history_is_acceptable'] & acceptable_df['future_is_acceptable'], 'row_is_acceptable'] = True

        # todo: check if we can remove == True
        final_df = aggregated_data.loc[acceptable_df[acceptable_df['row_is_acceptable'] == True].index]
        if self.verbose:
            print("Total row accepted from data_aggregator check_data_integrity function: ", len(final_df))

        return final_df

# not implemented yet
# def save_trade_statistics(self, final_df, len_original_data):
#     """
#     This function calculates the number of unique trades, % of unique trades compared to total trades, as well as the total possible reward.
#     :param final_df:
#     :param len_original_data: we need the length of the original data to be able to calculate % of trades
#     :return:
#     """
#     statistics = pd.DataFrame()
#     # time_difference_df = pd.DataFrame()
#     counter = 1
#     final_df['date'] = final_df.index
#     final_df['time_delta'] = final_df['date'].diff()
#
#     # .iloc[4]).days
#
#     print(type(final_df['time_delta']))
#     for key, value in ACTIONS.items():
#         # print(value)
#         trade_times = final_df['time_delta'][final_df['action']==value].sort_index().rolling(self.time_difference_between_trades, min_periods=1).sum()
#         # print(len(trade_times)
#         # statistics[key] = len(final_df[(trade_times>self.time_difference_between_trades)])
#         print(trade_times)
#         counter+=1
#         if counter % 3 ==0:
#             break
#     print(statistics)
# 1,2,3 rewards
# % of 2 trades
# % of 3 trades
# % of 1 trades
# total_rewards

# 31 - 0.1
# 7 -> 0.028
# data_loader = LoadYahooSymbol()

# file with volatility
# file_to_load = 'RAJOIL.BO.csv'
# folder = 'Yahoo_Finance_Stock_Data/BSE/'

# file with many data equal to each other
# file_to_load = 'UNQTYMI-ZERO.BO.csv'
# folder = 'Test_Data/'

# file_to_load = 'SPLV_From_Csv.csv'
# folder = 'Test_Data/'
#
# # with many high=low data: 'DHOF.AX.csv'
# # 'Yahoo_Finance_Stock_Data/ASX/'
#
# test_data = data_loader.load_file(folder, file_to_load)
# input_data = {
#     'sentence_length': 7,
#     'length_to_look_into_future_for_rewards': 3,
#     'min_volatility_for_explosive_move_filtering': 0.03,
#     'price_column_name': 'close',
#     'date_column_name': 'date',
#     'rounding_precision': 4,
#     'tolerance_percent_from_0_to_take_a_trade': 0,
#     'debug_mode': True
# }
# data_aggregator = DataAggregator(**input_data)
# final_ds = data_aggregator.aggregate_symbol_data(test_data, False)
# # data_aggregator.check_data_integrity(final_ds)
