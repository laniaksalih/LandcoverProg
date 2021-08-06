from LandcoverProg import LandcoverProg
import argparse
import os

parser = argparse.ArgumentParser(description = '...')
parser.add_argument('-f', type=str, help='The file you wish to classify')
parser.add_argument('-m', type=str, help='The model name as it appears in model_files.json')
args = parser.parse_args()

classification = LandcoverProg()


ws = classification.load_workspace()
print('Showing datastores...')
classification.show_datastores(ws)
account_key, datastore_name, file_share_name, account_name, file_path= classification.load_data_share_info()


registered = classification.register_data_share(account_key, ws, datastore_name, file_share_name, account_name)

datastore = classification.get_data_share(ws, datastore_name)
dataset_mount = classification.start_mount(datastore, file_path)
print('Mount started')
dataset_mount_folder = classification.create_dataset_mount_folder(dataset_mount)
print('Dataset mount folder created')
files_list = classification.walk_directory(dataset_mount_folder)
print('Showing list of file paths')
file_path = dataset_mount_folder + os.path.sep + args.f

try:
    extent = classification.define_extent(file_path)

    input_raster = classification.get_data_from_extent(file_path, extent)

    model_name = args.m

    output_raster = classification.pred_tile(input_raster, model_name)

    class_list = classification.load_classes_json()

    color_list = classification.load_colors_json()

    output_hard, output_raster, img_hard = classification.image_post_processing(input_raster, output_raster, color_list)

    print('Writing to .tif file')
    file_path = classification.write_to_geotiff(file_path, output_hard, output_raster, img_hard)
    print('Moving file to fileshare')
    classification.move_file_to_fileshare(file_path)
finally:
    classification.stop_mount(dataset_mount)


