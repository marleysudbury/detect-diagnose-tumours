# Demonstration of how folders and files are accessed
# Important for command line parameter passing

# Written my Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import os
import sys

class PathHandler:
    def PathHandler(provided_path):
        # provided_path = sys.argv[1]
        # provided_path = provided_path
        print(provided_path)
        interpreted_path = os.path.split(provided_path)
        print(interpreted_path)
        dir = None
        file = None
        print(os.path.join(interpreted_path[0],interpreted_path[1]))
        if os.path.isdir(os.path.join(interpreted_path[0],interpreted_path[1],'\\')):
            # The tail of the provided path is a folder
            # This will only be the case if all files are being
            # classified within the folder
            print("dir")
            dir = os.path.join(interpreted_path[0], interpreted_path[1])
            yourpath = dir
            print(dir)
            for root, dirs, files in os.walk(os.path.join(yourpath), topdown=False):
                for name in files:
                    print(name)

        if os.path.isfile(os.path.join(interpreted_path[0],interpreted_path[1])):
            # The tail of the provided path is a file
            # This will only be the case if a specific filed
            # is being classified
            print("file")
            dir = interpreted_path[0]
            file = interpreted_path[1]

        if dir == file == None:
            # The provided path is neither a file nor a folder
            print("not valid")

    def file():
        # Return true if current path specifies file
        pass

    def folder():
        # Return true if current path specifies folder
        pass

    def interate_files():
        # Iterate over all the files in the folder
        pass

