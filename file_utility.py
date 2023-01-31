import os
from line_printer import LinePrinter

class FileUtility:
    def __init__(self, source_data_path, save_destination_path, file_formats_to_load='csv', file_format_to_save='csv'
                 ,verbose=False):
        """
        This class loads all sub directories as well as all files within a directory
        The sub directories are only loaded up to one level, not nested
        :param source_data_path: main path where sub directories will be loaded
        :param save_destination_path: destination where data should be saved
        :param file_formats_to_load: the file format that the FileUtility must load while looking into a directory
        The file format does not need a "." so for example to load all csv files, file_format = 'csv'
        """
        print("Generating File Utility")

        # if save_destination_path[-1]!='/':
        #     raise Warning('save_destination_path must end with /')
        # if source_data_path[-1]!='/':
        #     raise Warning('source_data_path must end with /')

        self.source_data_path = source_data_path
        if not os.path.isdir(save_destination_path):
            os.makedirs(save_destination_path)

        self.save_destination_path = save_destination_path
        self.file_formats_to_load = file_formats_to_load.replace('.', '')
        self.file_format_to_save = file_format_to_save.replace('.', '')

        if not ((self.file_format_to_save == 'csv') | (self.file_format_to_save == 'xlsx')):
            raise Warning("File format to save must be either csv or xlsx")

        self.verbose = verbose
        self.line_printer = LinePrinter()

    def load_all_sub_directories(self):
        all_folders = os.listdir(self.source_data_path)
        if self.verbose:
            print('Loading All Sub Folders for: ', self.source_data_path)
            print("all_folders: ", all_folders)
        folder_list = []

        for file in all_folders:
            if os.path.isdir(self.source_data_path +'/'+ file):
                folder_list.append(file)
        if self.verbose:
            print('Loaded: ', len(folder_list), ' Folders')
            self.line_printer.print_text('Folder List')
            print(folder_list)

        return folder_list

    def load_file_names_in_directory(self, dir_):
        """
        Get all files in a directory with a specific extension specified at class level (self.file_formats_to_load)

        :param dir_: the directory to check
        :return: [] of files with self.file_formats_to_load extension in the specified directory
        """
        print('Loading file names in folder: ', dir_)
        all_files = os.listdir(self.source_data_path + '/' +dir_)
        file_list = []
        print(self.source_data_path + '/' + dir_)
        for file in all_files:
            if os.path.isfile(self.source_data_path + "/" + dir_ + "/" + file):
                if file.split('.')[-1] == self.file_formats_to_load:
                    file_list.append(file)

        if self.verbose:
            self.line_printer.print_text('Files in '+dir_)
            print(file_list)

        return file_list

    def save_data(self, data, folder_name, file_name):
        self.line_printer.print_text('I am save_data from file_utility')
        # print(data.name)
        print('folder_name: ', folder_name)
        print('file_name:', file_name)
        if not os.path.isdir(self.save_destination_path + "/" + folder_name):
            os.makedirs(self.save_destination_path + "/" + folder_name)
        if self.file_format_to_save == 'csv':
            data.to_csv(self.save_destination_path + "/" + folder_name + "/" + file_name + '.csv')
        elif self.file_format_to_save == 'xlsx':
            data.to_excel(self.save_destination_path + "/" + folder_name + "/" + file_name + '.xlsx')


