import os
import shutil
import re
from glob import iglob
from sklearn.utils import shuffle

image_dataset_lfw_gender_test = {
    "PROBE_PATH" : "H:/face_recognition/LFW_gender",
    "PROBE_IMAGE_PATH" : "H:/face_recognition/LFW_gender/LFW_gender_test",
    "PROBE_IMAGE_FILE_LIST" : None,
    "DATA_SET_NAME":"LFW_gender_test",
    "BASE_DATASET":"LFW_gender",
    "SLICE_DATA_SET": False,
    "START_IDX":0,
    "END_IDX":0,
}

image_dataset_lfw_all_gender_test = {
    "PROBE_PATH" : "H:/face_recognition/LFW_gender",
    "PROBE_IMAGE_PATH" : "H:/face_recognition/LFW_gender/LFW_all_gender_test",
    "PROBE_IMAGE_FILE_LIST" : None,
    "DATA_SET_NAME":"LFW_all_gender_test",
    "BASE_DATASET":"LFW_gender",
    "SLICE_DATA_SET": False,
    "START_IDX":0,
    "END_IDX":0,
}
SOURCE_IMAGES_PATH = "H:/face_recognition/lfw_gender/lfw_funneled_all/lfw_funneled"
MALE_FILES = "H:/face_recognition/lfw_gender/gender_labels/male_names.txt"
FEMALE_FILES = "H:/face_recognition/lfw_gender/gender_labels/female_names.txt"

# image_dataset  = image_dataset_lfw_gender_test
image_dataset  = image_dataset_lfw_all_gender_test

male = None
female = None
with open(MALE_FILES) as f:
    male= [s.strip() for s in f.readlines()[:]]
with open(FEMALE_FILES) as f:
    female= [s.strip() for s in f.readlines()[:]]

# if len(male)>len(female):
#     male =  shuffle(male, random_state=0)[:len(female)]
# else:
#     female =  shuffle(female, random_state=0)[:len(male)]




file_list = [f for f in iglob(SOURCE_IMAGES_PATH+"/**", recursive=True) if os.path.isfile(f)]
# for subdir, dirs, files in os.walk(SOURCE_IMAGES_PATH):
#     for file in files:
#         print(os.path.join(subdir, file))

idx = 0
target_img_folder = image_dataset["PROBE_IMAGE_PATH"]
os.makedirs(target_img_folder, exist_ok=True)

for source_img_file in file_list:
    # source_img_file =  os.path.join(SOURCE_IMAGES_PATH,source_img_filename)
    source_img_filename = os.path.split(source_img_file)[1]
    subject_name = source_img_filename.split(".")[0]
    subject_name = re.sub("[_0-9]+","",subject_name)
    if source_img_filename in male:
        gender = "m"
    elif source_img_filename in female:
        gender = "f"
    else:
        print("gender label not found for {}".format(source_img_filename))
        continue
    target_img_filename = "{}_{}_{}_{}.{}".format(idx, subject_name,"0",gender,source_img_filename.split(".")[1])
    target_img_file =  os.path.join(target_img_folder,target_img_filename)
    shutil.copyfile(source_img_file, target_img_file)
    idx+=1
