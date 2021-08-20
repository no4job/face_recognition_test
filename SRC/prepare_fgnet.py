import os
import shutil
import re

image_dataset_fgnet_test = {
    "PROBE_PATH" : "H:/face_recognition/FGNET",
    "PROBE_IMAGE_PATH" : "H:/face_recognition/FGNET/FGNET_test",
    "PROBE_IMAGE_FILE_LIST" : None,
    "DATA_SET_NAME":"FGNET_test",
    "BASE_DATASET":"FGNET",
    "SLICE_DATA_SET": False,
    "START_IDX":0,
    "END_IDX":0,
}
SOURCE_FGNET_IMAGES_PATH = "H:/face_recognition/FGNET/FGNET/images"
image_dataset  = image_dataset_fgnet_test
idx = 0

target_img_folder = image_dataset["PROBE_IMAGE_PATH"]
os.makedirs(target_img_folder, exist_ok=True)

for source_img_filename in os.listdir(SOURCE_FGNET_IMAGES_PATH):
    source_img_file =  os.path.join(SOURCE_FGNET_IMAGES_PATH,source_img_filename)
    n_parts = source_img_filename.split(".")[0].split("A")
    n_parts[1] = re.sub("[^0-9]","",n_parts[1])
    target_img_filename = "{}_person{}_{}_{}.{}".format(idx, n_parts[0],n_parts[1],"f",source_img_filename.split(".")[1])
    target_img_file =  os.path.join(target_img_folder,target_img_filename)
    shutil.copyfile(source_img_file, target_img_file)
    idx+=1
