import numpy as np
import os
# from shutil import copy
import shutil

def LFW_loader(image_dataset,index_file_path,data_folder_path):
    # lfw_test_pairs_file = os.path.join(source_dataset_path,"pairsDevTest.txt")
    # lfw_probe_image_path = os.path.join(source_dataset_path,"pairsDevTest.txt")
    # with open(source_dataset_path) as f:

    # parse the index file to find the number of pairs to be able to allocate
    # the right amount of memory before starting to decode the jpeg files
    with open(index_file_path, 'rb') as index_file:
        split_lines = [ln.decode().strip().split('\t') for ln in index_file]
    pair_specs = [sl for sl in split_lines if len(sl) > 2]
    n_pairs = len(pair_specs)
    # iterating over the metadata lines for each pair to find the filename to
    # decode and load in memory
    target = np.zeros(n_pairs, dtype=int)
    file_paths = list()
    for i, components in enumerate(pair_specs):
        if len(components) == 3:
            target[i] = 1
            pair = (
                (components[0], int(components[1]) - 1),
                (components[0], int(components[2]) - 1),
            )
        elif len(components) == 4:
            target[i] = 0
            pair = (
                (components[0], int(components[1]) - 1),
                (components[2], int(components[3]) - 1),
            )
        else:
            raise ValueError("invalid line %d: %r" % (i + 1, components))
        for j, (name, idx) in enumerate(pair):
            try:
                person_folder = os.path.join(data_folder_path, name)
            except TypeError:
                person_folder = os.path.join(data_folder_path, str(name, 'UTF-8'))
            filenames = list(sorted(os.listdir(person_folder)))
            file_path = os.path.join(person_folder, filenames[idx])
            file_paths.append(file_path)

    image_file_list_abs= sorted(list(set(file_paths)))
    image_file_list_rel = [os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.split(f)[1]).replace("\\","/") for f in image_file_list_abs]
    os.makedirs(image_dataset["PROBE_PATH"], exist_ok=True)
    with open(image_dataset["PROBE_IMAGE_FILE_LIST"], "w") as outfile:
        outfile.write("\n".join(image_file_list_rel))
    img_ = ""
    idx = -1
    labels = []
    for img in [os.path.split(img)[0] for img in image_file_list_rel]:
        if img != img_:
            idx+=1
            img_= img
            labels.append (idx)
        else:
            labels.append(idx)
    with open(image_dataset["PROBE_IMAGE_LABELS"], "w") as outfile:
        outfile.write(str(len(labels))+" "+str(max(labels)+1)+"\n")
        # outfile.write("\n".join(map(lambda x: str(labels),labels)))
        outfile.write("\n".join(map(str,labels)))

    file_paths_rel = [os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.split(f)[1]).replace("\\","/") for f in file_paths]
    idx = 0
    file_paths_rel_dict = {}
    for f in image_file_list_rel:
        file_paths_rel_dict[f] = idx
        idx+=1
    # file_paths_rel_dict = dict(zip(file_paths_rel,range(len(file_paths_rel))))

    pairs_index =   list(zip(map(lambda t: str(file_paths_rel_dict.get(t)),file_paths_rel[0::2]),
                             map(lambda t: str(file_paths_rel_dict.get(t)),file_paths_rel[1::2])))
    pairs_files = list(zip(file_paths_rel[0::2],file_paths_rel[1::2]))
    with open(image_dataset["PROBE_IMAGE_PAIRS_INDEX"], "w") as outfile:
        outfile.write("\n".join(map(lambda t: t[0]+"\t"+t[1] ,pairs_index)))
    with open(image_dataset["PROBE_IMAGE_PAIRS_FILES"], "w") as outfile:
        outfile.write("\n".join(map(lambda t: t[0]+"\t"+t[1] ,pairs_files)))
    os.makedirs(image_dataset["PROBE_IMAGE_PATH"], exist_ok=True)
    for img in image_file_list_rel:
        source_img_file =  os.path.join(data_folder_path,img)
        target_img_folder =  os.path.join(image_dataset["PROBE_IMAGE_PATH"],os.path.split(img)[0])
        os.makedirs(target_img_folder, exist_ok=True)
        target_img_fle =  os.path.join(image_dataset["PROBE_IMAGE_PATH"],img)
        shutil.copyfile(source_img_file, target_img_fle)
    return




if __name__ == "__main__":
    image_dataset_lfw_funneled_test = {
        "PROBE_PATH" : "H:/face_recognition/lfw_test",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/lfw_test/lfw_funneled",
        "PROBE_IMAGE_LABELS" : "H:/face_recognition/lfw_test/label.txt",
        "PROBE_IMAGE_PAIRS_INDEX": "H:/face_recognition/lfw_test/pairs_idx.txt",
        "PROBE_IMAGE_PAIRS_FILES": "H:/face_recognition/lfw_test/pairs_files.txt",
        "PROBE_IMAGE_FILE_LIST" : "H:/face_recognition/lfw_test/list.txt",
        "DISTRACTOR_IMAGE_PATH" : "",
        "DISTRACTOR_IMAGE_FILE_LIST" : None,
        "DISTRACTOR_IMAGE_TEST_SET" : None,
        "DATA_SET_NAME":"lfw_funneled_test",
        "SLICE_DATA_SET": False,
        "START_IDX":0,
        "END_IDX":0
    }
    image_dataset = image_dataset_lfw_funneled_test
    index_file_path = "H:/Видеоаналитика/LFW/lfw_home/pairsDevTest.txt"
    data_folder_path = "H:/Видеоаналитика/LFW/lfw_home/lfw_funneled"
    LFW_loader(image_dataset,index_file_path,data_folder_path)

    # image_dataset = image_dataset_lfw_funneled_test
    # split_lines = []
    # with open(image_dataset["PROBE_IMAGE_PAIRS_INDEX"], 'rb') as index_file:
    #     split_lines = [ln.decode().strip().split('\t') for ln in index_file]
    # pairs = [[int(p[0]),int(p[1])]  for p in split_lines]
    pass
exit(0)
