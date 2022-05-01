import glob
import os


def load_pnoa_filenames(base_folder, tile_file):
    """
    Localiza tiles del pnoa a partir de fichero
    :param base_folder:
    :return:
    """
    lines = open(tile_file).read().splitlines()
    files = set()
    for line in lines:
        if line:  # not empty
            # is a file
            fabs = "{}/{}"
            if os.path.exists(fabs) and os.path.isfile(fabs):
                files.add(fabs)
            else:
                nested_files = glob.glob("{}/{}/*.tif".format(base_folder, line))
                if len(nested_files) > 0:
                    # it has nested tif
                    files.update(nested_files)
                else:
                    # it has nested folders with tifs
                    nested_files = glob.glob("{}/{}/**/*.tif".format(base_folder, line))
                    files.update(nested_files)
    lst_files = list(files)
    lst_files.sort()
    return lst_files