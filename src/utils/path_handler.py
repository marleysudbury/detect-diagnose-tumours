# Demonstration of how folders and files are accessed
# Important for command line parameter passing

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import os
import sys


class PathHandler:
    def __init__(self, provided_path):
        self.provided_path = provided_path
        interpreted_path = os.path.split(provided_path)
        self.dir = None
        self.file_name = None
        potential_path = os.path.join(
            interpreted_path[0],
            interpreted_path[1],
            # '\\'
        )
        self.valid = True

        if os.path.isdir(potential_path):
            # The tail of the provided path is a folder
            # This will only be the case if all files are being
            # classified within the folder
            self.dir = potential_path

        if os.path.isfile(potential_path):
            # The tail of the provided path is a file
            # This will only be the case if a specific filed
            # is being classified
            self.dir = interpreted_path[0]
            self.file_name = interpreted_path[1]

        if self.dir == self.file_name == None:
            # The provided path is neither a file nor a folder
            print("Provided path is not valid")
            self.valid = False

    def file(self):
        # Return true if current path specifies file
        if self.file is not None:
            return True
        else:
            return False

    def folder(self):
        # Return true if current path specifies folder
        if self.file_name is None and self.dir is not None:
            return True
        else:
            return False

    def iterate_files(self):
        list_of_files = []
        # path = os.path.join(self.dir, self.file_name)
        for root, dirs, files in os.walk(self.dir, topdown=False):
            for name in files:
                if name[-3:] in ['tif', 'svs']:
                    list_of_files.append(os.path.join(root, name))
        return list_of_files
