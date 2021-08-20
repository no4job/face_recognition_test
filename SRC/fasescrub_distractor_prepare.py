import os

if __name__ == "__main__":
    image_dataset_facescrub_3530 = {
        "PROBE_PATH" : "H:/face_recognition/megaface_test",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/facescrub_images",
        "PROBE_IMAGE_LABELS" : "H:/face_recognition/megaface_test/facescrub3530/label.txt",
        "PROBE_IMAGE_PAIRS_INDEX": None,
        "PROBE_IMAGE_PAIRS_FILES": None,
        "PROBE_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/facescrub3530/list.txt",
        "DISTRACTOR_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/megaface_images",
        "DISTRACTOR_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/megaface_distractor/id_10_clean_list.txt",
        "DATA_SET_NAME":"facescrub_3530",
        "BASE_DATASET":"megaface_test",
        "SLICE_DATA_SET": False,
        "START_IDX":0,
        "END_IDX":0,
        "USE_PAIRS":False,
        "USE_DISTRACTORS":True
    }
    image_dataset = image_dataset_facescrub_3530
    DISTRACTOR_ALL_IMAGES_FILE_LIST = "H:/face_recognition/megaface_test/raw/megaface_lst"
    DISTRACTOR_IMAGE_SLICED_SET = "H:/face_recognition/megaface_test/megaface_distractor/id_@set_size@_clean"
    DISTRACTOR_IMAGE_PATH =  "H:/face_recognition/megaface_test/raw/megaface_images"
    all_images_file_list = []
    with open(DISTRACTOR_ALL_IMAGES_FILE_LIST) as f:
        all_images_file_list = [s.strip() for s in f.readlines()[:]]
    for size in [10,100,1000,10000,100000,1000000]:
        print("size:"+str(size))
        with open(DISTRACTOR_IMAGE_SLICED_SET.replace("@set_size@",str(size))) as f:
            idx_list = [int(s.strip()) for s in f.readlines()[:]]
            file_list = []
            for idx in idx_list:
                image_file  = os.path.join(DISTRACTOR_IMAGE_PATH,all_images_file_list[idx].strip())
                if os.path.exists(image_file):
                    file_list.append(all_images_file_list[idx].strip())
                else:
                    print("not found:"+all_images_file_list[idx].strip())


        with open(DISTRACTOR_IMAGE_SLICED_SET.replace("@set_size@",str(size))+"_list.txt","w") as f:
            f.write("\n".join(file_list))
exit(0)
