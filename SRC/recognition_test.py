from functools import cmp_to_key

from tqdm import tqdm
from deepface.commons import functions, distance as dst
import matplotlib.pyplot as plt

import os
import time
from deepface.basemodels import ArcFace
import deepface
from deepface import DeepFace
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import cv2
import glob
from datetime import datetime
import json
import csv
import auc
from sklearn import metrics

from operator import itemgetter, attrgetter
import math
import sys

# from scipy import interpolate

# PROBE_IMAGE_PATH =  "H:/face_recognition/megaface_test/raw/facescrub_images"
# PROBE_IMAGE_FILE_LIST =  "H:/face_recognition/megaface_test/raw/facescrub_lst"
# PROBE_IMAGE_LABELS = "H:/face_recognition/megaface_test/facescrub3530/label.txt"
# PROBE_IMAGE_FILE_LIST = "H:/face_recognition/megaface_test/facescrub3530/list.txt"

# DISTRACTOR_IMAGE_PATH =  "H:/face_recognition/megaface_test/raw/megaface_images"
# DISTRACTOR_IMAGE_FILE_LIST =  "H:/face_recognition/megaface_test/raw/megaface_lst"
# DISTRACTOR_IMAGE_TEST_SET =  "H:\face_recognition\megaface_test\megaface_distractor"

# EMBEDINGS_FILE = "H:/face_recognition/megaface_test/embeddings/emb_arcface__facescrub_3530.npy"
EMBEDINGS_PATH = "H:/face_recognition/@@base_dataset@@/embeddings"
# ***EMBEDINGS_PATH = "H:/face_recognition/megaface_test/embeddings"
# CMP_DETAILS_FILE = "H:/face_recognition/megaface_test/cmp_details/cmp_details.npy"
VERIFICATION_PATH = "H:/face_recognition/@@base_dataset@@/verification"
#***VERIFICATION_PATH = "H:/face_recognition/megaface_test/verification"

ALL_MODELS =["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
ALL_METRICS = ['cosine','euclidean', 'euclidean_l2']
TRASHOLD_MATRIX ={
    'VGG-Face':ALL_METRICS,'OpenFace':ALL_METRICS,'Facenet':ALL_METRICS,'Facenet512':ALL_METRICS,
    'DeepFace': ALL_METRICS,'DeepID': 	ALL_METRICS, 'Dlib':ALL_METRICS,'ArcFace': ALL_METRICS
}


PERSON_ATTRIBUTES_PATH = "H:/face_recognition/@@base_dataset@@/person_attributes"

def build_representation(model,model_name,  image_dataset, align = False, detect_face = False,enforce_detection = True, detector_backend = 'opencv'):
    print("{} expects ".format(model_name),model.layers[0].input_shape[0][1:]," inputs")
    try:
        print("and it represents faces as ", model.layers[-1].output_shape[1:]," dimensional vectors")
    except AttributeError as e:
        print(str(e))

    target_size = model.layers[0].input_shape[0][1:3]
    embeddings = []
    probe = []
    with open(image_dataset["PROBE_IMAGE_FILE_LIST"]) as f:
        probe= [os.path.join(image_dataset["PROBE_IMAGE_PATH"],s.strip()) for s in f.readlines()[:]]
    if image_dataset["SLICE_DATA_SET"]:
        probe_ = probe[image_dataset["START_IDX"]:image_dataset["END_IDX"]]
    else:
        probe_ = probe
    for p in probe_:
        # ***img = functions.load_image(p)
        # plt.imshow(img/255)
        # plt.show()
        # ***img_ = preprocess_face_(img, target_size=target_size,enforce_detection=enforce_detection , detector_backend = detector_backend, align = align, detect_face = detect_face)
        img_ = preprocess_face_(p, target_size=target_size,enforce_detection=enforce_detection , detector_backend = detector_backend, align = align, detect_face = detect_face)
        # plt.imshow(img_[0])
        # plt.show()
        embeddings.append(model.predict(img_)[0])
    return embeddings

def plt_img(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.show()

def preprocess_face_(img, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False, align = True, detect_face = True):

    #img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img = functions.load_image(img)
    # plt_img(img)
    base_img = img.copy()

    if detect_face:
        img, region = functions.detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection, align = align)

        #--------------------------

        if img.shape[0] == 0 or img.shape[1] == 0:
            if enforce_detection == True:
                raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
            else: #restore base image
                img = base_img.copy()

        #--------------------------
    else:
        if img.shape[0] == 0 or img.shape[1] == 0:
            if enforce_detection == True:
                raise ValueError("Detected face shape is ", img.shape)
        img = base_img.copy()
    #post-processing
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #---------------------------------------------------
    #resize image to expected shape

    # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

    # First resize the longer side to the target size
    #factor = target_size[0] / max(img.shape)

    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    img = cv2.resize(img, dsize)

    # Then pad the other side to the target size by adding black pixels
    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]
    if grayscale == False:
        # Put the base image in the middle of the padded image
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
    else:
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    #double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    #---------------------------------------------------
    # plt_img(img)

    img_pixels = image.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 #normalize input in [0, 1]

    if return_region == True:
        return img_pixels, region
    else:
        return img_pixels

def get_eval_stat_details(all_details):
    # label_1 = [details["label_1"] for details in  all_details]
    all_eval_stat = {}
    tp = []
    tn = []
    fp = []
    fn = []
    for details in  all_details:
        true_verification = details["true_verification"]
        verification = details["verification"]
        # label_1 = details["label_1"]
        # idx_1 = details["idx_1"]
        for idx in [details["idx_1"],details["idx_2"]]:
            # eval_stat = all_eval_stat.get(label_1,{"tp":0,"tn":0,"fp":0,"fn":0})
            eval_stat = all_eval_stat.get(idx,{"tp":0,"tn":0,"fp":0,"fn":0})
            if true_verification == verification and true_verification == 1:
                eval_stat["tp"]+=1
            if true_verification == verification and true_verification == 0:
                eval_stat["tn"]+=1
            if true_verification != verification and true_verification == 1:
                eval_stat["fn"]+=1
            if true_verification != verification and true_verification == 0:
                eval_stat["fp"]+=1
            # all_eval_stat[label_1] = eval_stat
            all_eval_stat[idx] = eval_stat
    # for label in all_eval_stat:
    #     tp.append(all_eval_stat[label]["tp"])
    #     tn.append(all_eval_stat[label]["tn"])
    #     fp.append(all_eval_stat[label]["fp"])
    #     fn.append(all_eval_stat[label]["fn"])
    for idx in all_eval_stat:
        tp.append(all_eval_stat[idx]["tp"])
        tn.append(all_eval_stat[idx]["tn"])
        fp.append(all_eval_stat[idx]["fp"])
        fn.append(all_eval_stat[idx]["fn"])

    # return [tp,tn,fp,fn]
    return {"tp":tp, "tn":tn, "fp":fp, "fn":fn}

def get_eval_stat_details_(all_details):
    # label_1 = [details["label_1"] for details in  all_details]
    all_eval_stat = {}
    tp = []
    tn = []
    fp = []
    fn = []
    for details in  all_details:
        true_verification = details["true_verification"]
        verification = details["verification"]
        # label_1 = details["label_1"]
        idx_1 = details["idx_1"]
        # eval_stat = all_eval_stat.get(label_1,{"tp":0,"tn":0,"fp":0,"fn":0})
        eval_stat = all_eval_stat.get(idx_1,{"tp":0,"tn":0,"fp":0,"fn":0})
        if true_verification == verification and true_verification == 1:
            eval_stat["tp"]+=1
        if true_verification == verification and true_verification == 0:
            eval_stat["tn"]+=1
        if true_verification != verification and true_verification == 1:
            eval_stat["fn"]+=1
        if true_verification != verification and true_verification == 0:
            eval_stat["fp"]+=1
        # all_eval_stat[label_1] = eval_stat
        all_eval_stat[idx_1] = eval_stat
    # for label in all_eval_stat:
    #     tp.append(all_eval_stat[label]["tp"])
    #     tn.append(all_eval_stat[label]["tn"])
    #     fp.append(all_eval_stat[label]["fp"])
    #     fn.append(all_eval_stat[label]["fn"])
    for idx in all_eval_stat:
        tp.append(all_eval_stat[idx]["tp"])
        tn.append(all_eval_stat[idx]["tn"])
        fp.append(all_eval_stat[idx]["fp"])
        fn.append(all_eval_stat[idx]["fn"])

    # return [tp,tn,fp,fn]
    return {"tp":tp, "tn":tn, "fp":fp, "fn":fn}

def tst(true_verification_,distance_,th,idx_,idx_1_,idx_2_):
    tp_=[]
    tn_=[]
    fp_=[]
    fn_=[]
    c = -1
    for idx in idx_:
        # print(c)
        c+=1
        tp=tn=fp=fn=0
        for i in idx:
            idx_1=idx_1_[i]
            idx_2=idx_2_[i]
            true_verification = true_verification_[i]
            distance = -distance_[i]
            if distance <  th :
                verification = 1
            else:
                verification = 0
            if true_verification == verification and true_verification == 1:
                tp+=1
            if true_verification == verification and true_verification == 0:
                tn+=1
            if true_verification != verification and true_verification == 1:
                fn+=1
            if true_verification != verification and true_verification == 0:
                fp+=1
            # all_eval_stat[label_1] = eval_stat
            if (c==34 or c==35 ) and i==119425:
                print(true_verification,verification, distance,th, fn)
        if (c==34 or c==35 ):
            print(fn)
        tp_.append(tp)
        tn_.append(tn)
        fp_.append(fp)
        fn_.append(fn)
        # return [tp,tn,fp,fn]
        # return {"tp":tp_, "tn":tn_, "fp":fp_, "fn":fn_}
    result =  clf_metrics_mean(tp_, tn_, fp_, fn_)
    print (result)
    pass
    return



def mean_roc_curve(true_verification,distance,idx_1,idx_2 ):
    # idx_1_ = set(idx_1)
    idx_ =[[] for i in range(len(set(idx_1+idx_2)))]
    # idx_1_ = [[]] * len(set(idx_1))
    for idx in range(len (idx_1)):
        idx_[idx_1[idx]].append(idx)
        idx_[idx_2[idx]].append(idx)
        pass

    tst(true_verification,distance,4.1591468986978,idx_,idx_1,idx_2)

    base_fpr = np.linspace(0, 1, 101)
    base_fpr_ = base_fpr.tolist()
    base_fpr_.append(0.441738458404305)
    base_fpr = np.array(sorted(base_fpr_))

    tprs = []
    # for i in range(len(idx_)):
    #     true_verification_ = [true_verification[i_]  for i_ in  idx_[i]]
    #     distance_ = [distance[i_]  for i_ in  idx_[i]]
    for idx in idx_:
        true_verification_ = [true_verification[i_]  for i_ in  idx ]
        distance_ = [distance[i_]  for i_ in  idx]

        fpr, tpr, thresh = metrics.roc_curve(true_verification_, distance_, drop_intermediate =False)
        # print (base_fpr, fpr, tpr)
        tpr = np.interp(base_fpr, fpr, tpr)
        print(base_fpr[45],fpr[45],tpr[45])
        # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    return [list(base_fpr),list(mean_tprs)]



def verification_eval_pairs(image_dataset,embeddings, distance_metric, threshold,distractors):
    split_lines = []
    with open(image_dataset["PROBE_IMAGE_PAIRS_INDEX"], 'rb') as index_file:
        split_lines = [ln.decode().strip().split('\t') for ln in index_file]
    pairs = [[int(p[0]),int(p[1])]  for p in split_lines]
    with open(image_dataset["PROBE_IMAGE_FILE_LIST"]) as f:
        probe= [os.path.join(image_dataset["PROBE_IMAGE_PATH"],s.strip()) for s in f.readlines()[:]]
    with open(image_dataset["PROBE_IMAGE_LABELS"]) as f:
        labels= [s.strip() for s in f.readlines()[1:]]

    all_details = []
    tp = tn = fp = fn = 0
    for pair in pairs:
        idx_1 = pair[0]
        emb_1 = embeddings[idx_1]
        idx_2 = pair[1]
        emb_2 = embeddings[idx_2]
        # if idx_1 < idx_2:
        details = {}
        details["idx_1"] = idx_1
        details["idx_2"] = idx_2
        details["img_1"] = probe[idx_1]
        details["img_2"] = probe[idx_2]
        label_1 = labels[idx_1]
        details["label_1"] = label_1
        label_2 = labels[idx_2]
        details["label_2"] = label_2
        if distance_metric == "cosine":
            distance = dst.findCosineDistance(emb_1, emb_2)
        elif distance_metric == "euclidean":
            distance = dst.findEuclideanDistance(emb_1, emb_2)
        elif distance_metric == 'euclidean_l2':
            distance = dst.findEuclideanDistance(dst.l2_normalize(emb_1), dst.l2_normalize(emb_2))
        else:
            raise ValueError("Invalid distance_metric passed - ", distance_metric)

        details["distance"] = distance
        # details["threshold"] = threshold
        if distance < threshold:
            verification = 1
        else:
            verification = 0
        details["verification"] = verification
        if label_1 == label_2:
            true_verification = 1
        else:
            true_verification = 0
        details["true_verification"] = true_verification
        details["verification"] = verification
        if true_verification == verification and true_verification == 1:
            tp+=1
        if true_verification == verification and true_verification == 0:
            tn+=1
        if true_verification != verification and true_verification == 1:
            fn+=1
        if true_verification != verification and true_verification == 0:
            fp+=1
        all_details.append(details)

    return {"eval_stat":{"tp":tp, "tn":tn, "fp":fp, "fn":fn}, "eval_details":all_details }
def cmp_all_all_details(a,b):

    # if int(a["label_1"]) > int(b["label_1"]):
    #     return 1
    # elif int(a["label_1"]) < int(b["label_1"]):
    #     return -1
    # else:

    if a["idx_1"] > b["idx_1"]:
        return 1
    elif a["idx_1"] < b["idx_1"]:
        return -1
    else:
        if  a["label_2"] == "distractor" and b["label_2"] != "distractor" :
            return 1
        elif a["label_2"] != "distractor" and b["label_2"] == "distractor":
            return -1
        # elif  (a["label_2"] == "distractor" and b["label_2"] == "distractor")  or int(a["label_2"]) == int(b["label_2"]):
        else:
            if a["idx_2"] > b["idx_2"]:
                return 1
            elif a["idx_2"] < b["idx_2"]:
                return -1
            else:
                return 0
        # elif  int(a["label_2"]) > int(b["label_2"]):
        #     return 1
        # elif int(a["label_2"]) < int(b["label_2"]):
        #     return -1
        # else:
            return None




def verification_eval(image_dataset, embeddings, distance_metric, threshold,distractor_embeddings:None):
    start_idx = 0
    end_idx = None
    probe_list = None
    distractors = None
    distractors_list = None
    gallery_labels = None
    probe_embeddings = embeddings
    emb_types = None
    if image_dataset["SLICE_DATA_SET"]:
        start_idx = image_dataset["START_IDX"]
        end_idx = image_dataset["END_IDX"]
    else:
        end_idx = len(embeddings)
        # embeddings = embeddings[start_idx:end_idx]
    with open(image_dataset["PROBE_IMAGE_FILE_LIST"]) as f:
        probe_list= [os.path.join(image_dataset["PROBE_IMAGE_PATH"],s.strip()) for s in f.readlines()[:]]
    if image_dataset["USE_DISTRACTORS"]:
        with open(image_dataset["DISTRACTOR_IMAGE_FILE_LIST"]) as f:
            distractors_list= [os.path.join(image_dataset["DISTRACTOR_IMAGE_PATH"],s.strip()) for s in f.readlines()[:]]

    with open(image_dataset["PROBE_IMAGE_LABELS"]) as f:
        probe_labels= [s.strip() for s in f.readlines()[1:]]

    all_details = []
    tp = tn = fp = fn = 0
    if image_dataset["USE_DISTRACTORS"]:
        emb_types = ["probe","distractor"]
    else:
        emb_types = ["probe"]
    if image_dataset["USE_DISTRACTORS"]:
        total_aprox = math.comb(int(len(embeddings)/len(set(probe_labels[start_idx:end_idx]))),2)*len(set(probe_labels[start_idx:end_idx]))+len(embeddings)*len(distractor_embeddings)
    else:
        total_aprox = math.comb(int(len(embeddings)/len(set(probe_labels[start_idx:end_idx]))),2)*len(set(probe_labels[start_idx:end_idx]))
    pbar = tqdm(total=total_aprox)
    for emb_type in emb_types:
        if emb_type=="probe":
            gallery_embeddings = embeddings
            gallery_list = probe_list
            gallery_labels= probe_labels
        else:
            gallery_embeddings = distractor_embeddings
            gallery_list= distractors_list
            gallery_labels= ["distractor"]*len(distractors_list)
        idx_1 = start_idx
        for emb_1 in probe_embeddings[:]:
            idx_2 = start_idx
            for emb_2 in gallery_embeddings[:]:
                if (idx_1 < idx_2 and emb_type == "probe" and probe_labels[idx_1] == probe_labels[idx_2]) or \
                        emb_type == "distractor":
                    details = {}
                    details["idx_1"] = idx_1
                    details["idx_2"] = idx_2
                    details["img_1"] = probe_list[idx_1]
                    details["img_2"] = gallery_list[idx_2]
                    label_1 = probe_labels[idx_1]
                    details["label_1"] = label_1
                    label_2 = gallery_labels[idx_2]
                    details["label_2"] = label_2
                    details["img_2_type"] = emb_type
                    if distance_metric == "cosine":
                        distance = dst.findCosineDistance(emb_1, emb_2)
                    elif distance_metric == "euclidean":
                        distance = dst.findEuclideanDistance(emb_1, emb_2)
                    elif distance_metric == 'euclidean_l2':
                        distance = dst.findEuclideanDistance(dst.l2_normalize(emb_1), dst.l2_normalize(emb_2))
                    else:
                        raise ValueError("Invalid distance_metric passed - ", distance_metric)

                    details["distance"] = distance
                    # details["threshold"] = threshold
                    if distance < threshold:
                        verification = 1
                    else:
                        verification = 0
                    details["verification"] = verification
                    if label_1 == label_2:
                        true_verification = 1
                    else:
                        true_verification = 0

                    details["true_verification"] = true_verification
                    details["verification"] = verification

                    if true_verification == verification and true_verification == 1:
                        tp+=1
                    if true_verification == verification and true_verification == 0:
                        tn+=1
                    if true_verification != verification and true_verification == 1:
                        fn+=1
                    if true_verification != verification and true_verification == 0:
                        fp+=1
                    all_details.append(details)
                    pbar.update(1)
                idx_2+=1
            idx_1+=1
    pbar.close()
    all_details_key = cmp_to_key(cmp_all_all_details)
    all_details_ = sorted(all_details, key=all_details_key)
    # print("\n".join(str(details) for details in  all_details_) )
    # print(all_details_ )
    return {"eval_stat":{"tp":tp, "tn":tn, "fp":fp, "fn":fn}, "eval_details":all_details_ }


def clf_metrics(tp, tn, fp, fn):
    result = {}
    result["tp"]=tp
    result["tn"]=tn
    result["fp"]=fp
    result["fn"]=fn
    result["total"] = tp+fp+fn+tn
    result["accuracy"] = 0 if tp+tn == 0 else  (tp+tn)/(tp+fp+fn+tn)
    result["precision"] = 0 if tp == 0 else tp/(tp+fp)
    result["recall"] = 0 if tp == 0 else  tp/(tp+fn)
    result["fpr"] = 0 if fp == 0 else fp/(tn+fp) #fpr = fmr = far
    result["fnr"] = 0 if fn == 0 else fn/(tp+fn) #fnr = fnmr
    result["tpr"] = 0 if tp == 0 else tp/(tp+fn) #tpr = tmr = tar
    result["tnr"] = 0 if tn == 0 else tn/(tn+fp) #tnr = tnmr

    return result

def clf_metrics_mean(tp, tn, fp, fn):
    result_ = {"tp":[],"tn":[],"fp":[],"fn":[],"total":[],"accuracy":[],"precision":[],"recall":[]
        ,"fpr":[],"fnr":[],"tpr":[],"tnr":[]}
    tp_ = tp
    tn_ = tn
    fp_ = fp
    fn_ = fn
    for idx in range(len(tp_)):
        tp = tp_[idx]
        tn = tn_[idx]
        fp = fp_[idx]
        fn = fn_[idx]
        result_["tp"].append(tp)
        result_["tn"].append(tn)
        result_["fp"].append(fp)
        result_["fn"].append(fn)
        result_["total"].append (tp+fp+fn+tn)
        result_["accuracy"].append( 0 if tp+tn == 0 else  (tp+tn)/(tp+fp+fn+tn))
        result_["precision"].append(0 if tp == 0 else tp/(tp+fp))
        result_["recall"].append(0 if tp == 0 else  tp/(tp+fn))
        result_["fpr"].append(0 if fp == 0 else fp/(tn+fp)) #fpr = fmr = far
        result_["fnr"].append(0 if fn == 0 else fn/(tp+fn)) #fnr = fnmr
        result_["tpr"].append(0 if tp == 0 else tp/(tp+fn)) #tpr = tmr = tar
        result_["tnr"].append(0 if tn == 0 else tn/(tn+fp)) #tnr = tnmr
    result={}
    for metric in ["accuracy","precision","recall","fpr","fnr","tpr","tnr"]:
        result[metric] = sum (result_[metric])/len (result_[metric])
    for metric in ["tp","tn","fp","fn","total"]:
        result[metric] = sum (result_[metric])
    return result




def print_classification_metrics(verification_result):
    print ("total = {}".format(verification_result["total"]))
    print ("tp={},tn={}".format(verification_result["tp"],verification_result["tn"]))
    print ("fp={},fn={}".format(verification_result["fp"],verification_result["fn"]))
    print ("accuracy = {}, precision = {}, recall= {}".format(verification_result["accuracy"],verification_result["precision"], verification_result["recall"]))
    print ("tpr = tmr = tar = {} , fpr = fmr = far = {}".format(verification_result["tpr"],verification_result["fpr"]))
    print ("tnr = tnmr = {} , fnr = fnmr =  {}, ".format(verification_result["tnr"],verification_result["fnr"]))

def rgsr_metrics(true_y, eval_y):
    # mse = metrics.mean_squared_error(true_y, eval_y)
    mape = metrics.mean_absolute_percentage_error(true_y, eval_y)
    mae = metrics.mean_absolute_error(true_y, eval_y)
    result = {}
    # result["mse"]=mse
    result["mape"]=mape
    result["mae"]=mae

    return result

def print_rgsr_metrics(rgsr_result, samples):
    print ("mape:{}, mae:{}, samples:{}".format(rgsr_result["mape"],rgsr_result["mae"],
                                               samples))


def localize_floats(row):
    for el in row:
        row[el] = str(row[el]).replace('.', ',') if isinstance(row[el], float) else row[el]

def person_attributes_eval(image_dataset, actions = None):
    f_names = [f for f in os.listdir(image_dataset["PROBE_IMAGE_PATH"]) if
                 os.path.isfile(os.path.join(image_dataset["PROBE_IMAGE_PATH"], f))]
    if image_dataset["SLICE_DATA_SET"]:
        start_idx = image_dataset["START_IDX"]
        end_idx = image_dataset["END_IDX"]
    else:
        start_idx = 0
        end_idx = len(f_names)

    name = []
    id = []
    label = []
    true_age = []
    true_gender = []
    eval_age = []
    eval_gender = []
    err = []

    # idx = 0
    # idx2 = 0
    pbar = tqdm(total=end_idx)
    for f_name in f_names[start_idx:end_idx]:
        # idx+=1
        # if idx<2400:continue
        # img_ = os.path.join(image_dataset["PROBE_IMAGE_PATH"],f_name)
        # img__ =img_.replace("\\","/")
        # img1 = cv2.imread(img__)
        # img2 = plt.imread(img__)
        # img2=img2[...,::-1]
        # cmp = np.array_equal(img1,img2)
        # strt = time.time()
        # for f_name in f_names:
        #     eval_attributes  = DeepFace.analyze(os.path.join(image_dataset["PROBE_IMAGE_PATH"],f_name),
        #                         actions = ['age', 'gender'], enforce_detection = False, detector_backend = 'opencv')
        #     idx2+=1
        #     end = time.time()
        #     tm= end - strt
        #     print(int(tm),idx2)


        img = os.path.join(image_dataset["PROBE_IMAGE_PATH"],f_name)
        try:
            eval_attributes  = DeepFace.analyze(img, actions = actions, enforce_detection = False, detector_backend = 'opencv', prog_bar =  False)
        except:
            img = plt.imread(img)
            try:
                eval_attributes  = DeepFace.analyze(img, actions = actions, enforce_detection = False, detector_backend = 'opencv', prog_bar =  False)
            except:
                err.append(f_name)
                continue
        f_name_spl = f_name.split(".")[0].split("_")
        name.append(f_name)
        id.append(int(f_name_spl[0]))
        label.append(f_name_spl[1])
        if "age" in actions:
            true_age.append(float(f_name_spl[2]))
            eval_age.append(eval_attributes["age"])
        if "gender" in actions:
            true_gender.append(f_name_spl[3])
            if eval_attributes["gender"].lower() == "woman":
                eval_gender.append("f")
            else:
                eval_gender.append("m")

        pbar.update(1)
    pbar.close()
    attributes = {"name":name,"id":id,"label":label,"errors":err}
    if "age" in actions:
        attributes = {**attributes,"true_age":true_age,"eval_age":eval_age}
    if "gender" in actions:
        attributes = {**attributes,"true_gender":true_gender,"eval_gender":eval_gender}
        # attributes = {"name":name,"id":id,"label":label,"true_age":true_age,"true_gender":true_gender,"eval_age":eval_age,
        #           "eval_gender":eval_gender,"errors":err }
    # print("\n".join([str(i) for i in attributes["eval_age"]]))
    return attributes

def report_person_attributes(attributes,person_attributes_timestamp, save_report_to_csv = False):
    report_age =  not (attributes.get("true_age",None) is None)
    report_gender =  not (attributes.get("true_gender",None) is None)
    true_age = None
    eval_age = None
    true_gender = None
    eval_gender = None

    if report_age:
        true_age = attributes["true_age"]
        eval_age = attributes["eval_age"]
    if report_gender:
        true_gender = attributes["true_gender"]
        eval_gender = attributes["eval_gender"]
    rgsr_metrics_info =[]
    clf_metrics_info =[]
    age_ranges = None
    if report_age:
        print("age metrics for all ages")
        mtr = rgsr_metrics(true_age, eval_age)
        print_rgsr_metrics(mtr,len(true_age))
        rgsr_metrics_info.append({"age_range":"all", **mtr,"samples":len(true_age)})
        # print ("mse: {}, mape:{}, mae:{}".format(mse,mape, mae))
        age_ranges = [[0,5],[6,10],[11,15],[16,20],[21,25],[26,30],[31,40],[41,50],[51,60],[61,70],[71,80],[81,90],[91,100],[101,200]]
        for r in age_ranges:
            attributes_idx = [idx for idx in range(len(true_age)) if true_age[idx]>=r[0] and true_age[idx] <=r[1]]
            if len(attributes_idx)==0:
                continue
            true_age_ = [true_age[idx] for idx in  attributes_idx]
            eval_age_ = [eval_age[idx] for idx in  attributes_idx]
            print("\nage metrics for age range {}".format(r))
            mtr = rgsr_metrics(true_age_, eval_age_)
            print_rgsr_metrics(mtr,len(true_age_))
            rgsr_metrics_info.append({"age_range":str(r), **mtr,"samples":len(true_age_)})


    if report_gender:
        tn, fp, fn, tp = metrics.confusion_matrix(true_gender, eval_gender, labels=["m","f"]).ravel()
        mtr = clf_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
        clf_metrics_info.append({"age_range":"all", **mtr,"total":len(true_gender)})

        print("\ngender metrics for all ages")
        print_classification_metrics(mtr)
        if report_age:
            for r in age_ranges:
                attributes_idx = [idx for idx in range(len(attributes["true_age"])) if attributes["true_age"][idx]>=r[0] and attributes["true_age"][idx] <=r[1]]
                true_gender = [attributes["true_gender"][idx] for idx in  attributes_idx]
                eval_gender = [attributes["eval_gender"][idx] for idx in  attributes_idx]
                tn, fp, fn, tp = metrics.confusion_matrix(true_gender, eval_gender, labels=["m","f"]).ravel()
                print("\ngender metrics for age range {}".format(r))
                tn, fp, fn, tp = metrics.confusion_matrix(true_gender, eval_gender, labels=["m","f"]).ravel()
                mtr = clf_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
                clf_metrics_info.append({"age_range":str(r), **mtr,"total":len(true_gender)})
                print_classification_metrics(mtr)


    if save_report_to_csv:
        # fieldnames = [
        #     "age_range",
        #     "mape", "mae",
        #     "person_attributes_timestamp",'dataset_name','enforce_detection','detector_backend']
        person_attributes_path = os.path.join(PERSON_ATTRIBUTES_PATH,image_dataset["DATA_SET_NAME"])
        person_age_report_file = os.path.join(person_attributes_path,
                                              "person_age_report_{}.csv".format(person_attributes_timestamp))
        if report_age:
            fieldnames = [
                "age_range",
                "mape", "mae", "samples",
                "person_attributes_timestamp"]
            with open(person_age_report_file, 'w', newline='') as csvfile:
                # writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='excel-tab')
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                info_={}
                for r in rgsr_metrics_info:
                    info_={**r}
                    info_["person_attributes_timestamp"] = person_attributes_timestamp
                    localize_floats(info_)
                    writer.writerow(info_)
        # fieldnames = [
        #               "age_range","tp","tn","fp","fn","total","accuracy","precision","recall","fpr","fnr","tpr","tnr",
        #               "person_attributes_timestamp",'dataset_name','enforce_detection','detector_backend']
        if report_gender:
            fieldnames = [
                "age_range",
                "tp","tn","fp","fn","total","accuracy","precision","recall","fpr","fnr","tpr","tnr",
                "person_attributes_timestamp","notes"]
            person_gender_report_file = os.path.join(person_attributes_path,
                                                     "person_gender_report_{}.csv".format(person_attributes_timestamp))
            with open(person_gender_report_file, 'w', newline='') as csvfile:
                # writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='excel-tab')
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                info_={}
                for r in clf_metrics_info:
                    info_={**r}
                    info_["notes"]="positive label - female"
                    info_["person_attributes_timestamp"] = person_attributes_timestamp
                    localize_floats(info_)
                    writer.writerow(info_)

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

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
        "DISTRACTOR_DATA_SET_NAME": None,
        "BASE_DATASET":"megaface_test",
        "SLICE_DATA_SET": False,
        "START_IDX":0,
        "END_IDX":0,
        "USE_PAIRS":False,
        "USE_DISTRACTORS":False
    }
    image_dataset_facescrub_3530_sliced = {
        "PROBE_PATH" : "H:/face_recognition/megaface_test",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/facescrub_images",
        "PROBE_IMAGE_LABELS" : "H:/face_recognition/megaface_test/facescrub3530/label.txt",
        "PROBE_IMAGE_PAIRS_INDEX": None,
        "PROBE_IMAGE_PAIRS_FILES": None,
        "PROBE_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/facescrub3530/list.txt",
        "DISTRACTOR_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/megaface_images",
        "DISTRACTOR_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/megaface_distractor/id_10_clean_list.txt",
        "DATA_SET_NAME":"facescrub_3530_sliced",
        "DISTRACTOR_DATA_SET_NAME": None,
        "BASE_DATASET":"megaface_test",
        "SLICE_DATA_SET": True,
        "START_IDX":0,
        "END_IDX":10,
        "USE_PAIRS":False,
        "USE_DISTRACTORS":False
    }
    image_dataset_facescrub_3530_sliced_100 = {
        "PROBE_PATH" : "H:/face_recognition/megaface_test",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/facescrub_images",
        "PROBE_IMAGE_LABELS" : "H:/face_recognition/megaface_test/facescrub3530/label.txt",
        "PROBE_IMAGE_PAIRS_INDEX": None,
        "PROBE_IMAGE_PAIRS_FILES": None,
        "PROBE_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/facescrub3530/list.txt",
        "DISTRACTOR_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/megaface_images",
        "DISTRACTOR_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/megaface_distractor/id_10_clean_list.txt",
        "DATA_SET_NAME":"facescrub_3530_sliced_100",
        "DISTRACTOR_DATA_SET_NAME": None,
        "BASE_DATASET":"megaface_test",
        "SLICE_DATA_SET": True,
        "START_IDX":0,
        "END_IDX":100,
        "USE_PAIRS":False,
        "USE_DISTRACTORS":False
    }
    image_dataset_facescrub_3530_sliced_100_detect = {
        "PROBE_PATH" : "H:/face_recognition/megaface_test",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/facescrub_images",
        "PROBE_IMAGE_LABELS" : "H:/face_recognition/megaface_test/facescrub3530/label.txt",
        "PROBE_IMAGE_PAIRS_INDEX": None,
        "PROBE_IMAGE_PAIRS_FILES": None,
        "PROBE_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/facescrub3530/list.txt",
        "DISTRACTOR_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/megaface_images",
        "DISTRACTOR_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/megaface_distractor/id_10_clean_list.txt",
        "DATA_SET_NAME":"facescrub_3530_sliced_100_detect",
        "DISTRACTOR_DATA_SET_NAME": None,
        "BASE_DATASET":"megaface_test",
        "SLICE_DATA_SET": True,
        "START_IDX":0,
        "END_IDX":100,
        "USE_PAIRS":False,
        "USE_DISTRACTORS":False
    }
    image_dataset_facescrub_3530_sliced_100_detect_retinaface = {
        "PROBE_PATH" : "H:/face_recognition/megaface_test",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/facescrub_images",
        "PROBE_IMAGE_LABELS" : "H:/face_recognition/megaface_test/facescrub3530/label.txt",
        "PROBE_IMAGE_PAIRS_INDEX": None,
        "PROBE_IMAGE_PAIRS_FILES": None,
        "PROBE_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/facescrub3530/list.txt",
        "DISTRACTOR_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/megaface_images",
        "DISTRACTOR_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/megaface_distractor/id_10_clean_list.txt",
        "DATA_SET_NAME":"facescrub_3530_sliced_100_detect_retinaface",
        "DISTRACTOR_DATA_SET_NAME": None,
        "BASE_DATASET":"megaface_test",
        "SLICE_DATA_SET": True,
        "START_IDX":0,
        "END_IDX":100,
        "USE_PAIRS":False,
        "USE_DISTRACTORS":False
    }
    image_dataset_lfw_funneled_test = {
        "PROBE_PATH" : "H:/face_recognition/lfw_test",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/lfw_test/lfw_funneled",
        "PROBE_IMAGE_LABELS" : "H:/face_recognition/lfw_test/label.txt",
        "PROBE_IMAGE_PAIRS_INDEX": "H:/face_recognition/lfw_test/pairs_idx.txt",
        "PROBE_IMAGE_PAIRS_FILES": "H:/face_recognition/lfw_test/pairs_files.txt",
        "PROBE_IMAGE_FILE_LIST" : "H:/face_recognition/lfw_test/list.txt",
        "DISTRACTOR_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/megaface_images",
        "DISTRACTOR_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/megaface_distractor/id_10_clean_list.txt",
        "DATA_SET_NAME":"lfw_funneled_test",
        "DISTRACTOR_DATA_SET_NAME": None,
        "BASE_DATASET":"lfw_test",
        "SLICE_DATA_SET": False,
        "START_IDX":0,
        "END_IDX":0,
        "USE_PAIRS":True,
        "USE_DISTRACTORS":False
    }

    image_dataset_megaface_distractor_id_1000_clean = {
        "PROBE_PATH" : "H:/face_recognition/megaface_test",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/megaface_images",
        "PROBE_IMAGE_LABELS" : None,
        "PROBE_IMAGE_PAIRS_INDEX": None,
        "PROBE_IMAGE_PAIRS_FILES": None,
        "PROBE_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/megaface_distractor/id_1000_clean_list.txt",
        "DISTRACTOR_IMAGE_PATH" : None,
        "DISTRACTOR_IMAGE_FILE_LIST" : None,
        "DATA_SET_NAME":"megaface_distractor_id_1000_clean",
        "DISTRACTOR_DATA_SET_NAME": None,
        "BASE_DATASET":"megaface_test",
        "SLICE_DATA_SET": False,
        "START_IDX":0,
        "END_IDX":0,
        "USE_PAIRS":False,
        "USE_DISTRACTORS":False
    }
    image_dataset_agedb_test = {
        "PROBE_PATH" : "H:/face_recognition/AgeDB",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/AgeDB/AgeDB_test",
        "PROBE_IMAGE_FILE_LIST" : None,
        "DATA_SET_NAME":"AgeDB_test",
        "BASE_DATASET":"AgeDB",
        "SLICE_DATA_SET": False,
        "START_IDX":0,
        "END_IDX":10,
    }


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
    image_dataset_facescrub_3530_sliced_distractor_10 = {
        "PROBE_PATH" : "H:/face_recognition/megaface_test",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/facescrub_images",
        "PROBE_IMAGE_LABELS" : "H:/face_recognition/megaface_test/facescrub3530/label.txt",
        "PROBE_IMAGE_PAIRS_INDEX": None,
        "PROBE_IMAGE_PAIRS_FILES": None,
        "PROBE_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/facescrub3530/list.txt",
        "DISTRACTOR_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/megaface_images",
        "DISTRACTOR_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/megaface_distractor/id_10_clean_list.txt",
        "DATA_SET_NAME":"facescrub_3530_sliced",
        "DISTRACTOR_DATA_SET_NAME": "megaface_distractor_id_10_clean",
        "BASE_DATASET":"megaface_test",
        "SLICE_DATA_SET": True,
        "START_IDX":0,
        "END_IDX":10,
        "USE_PAIRS":False,
        "USE_DISTRACTORS":True
    }
    image_dataset_facescrub_3530_sliced_100_distractor_10 = {
        "PROBE_PATH" : "H:/face_recognition/megaface_test",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/facescrub_images",
        "PROBE_IMAGE_LABELS" : "H:/face_recognition/megaface_test/facescrub3530/label.txt",
        "PROBE_IMAGE_PAIRS_INDEX": None,
        "PROBE_IMAGE_PAIRS_FILES": None,
        "PROBE_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/facescrub3530/list.txt",
        "DISTRACTOR_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/megaface_images",
        "DISTRACTOR_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/megaface_distractor/id_10_clean_list.txt",
        "DATA_SET_NAME":"facescrub_3530_sliced_100",
        "DISTRACTOR_DATA_SET_NAME":  "megaface_distractor_id_10_clean",
        "BASE_DATASET":"megaface_test",
        "SLICE_DATA_SET": True,
        "START_IDX":0,
        "END_IDX":100,
        "USE_PAIRS":False,
        "USE_DISTRACTORS":True
    }
    image_dataset_facescrub_3530_distractor_10000 = {
        "PROBE_PATH" : "H:/face_recognition/megaface_test",
        "PROBE_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/facescrub_images",
        "PROBE_IMAGE_LABELS" : "H:/face_recognition/megaface_test/facescrub3530/label.txt",
        "PROBE_IMAGE_PAIRS_INDEX": None,
        "PROBE_IMAGE_PAIRS_FILES": None,
        "PROBE_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/facescrub3530/list.txt",
        "DISTRACTOR_IMAGE_PATH" : "H:/face_recognition/megaface_test/raw/megaface_images",
        "DISTRACTOR_IMAGE_FILE_LIST" : "H:/face_recognition/megaface_test/megaface_distractor/id_10000_clean_list.txt",
        "DATA_SET_NAME":"facescrub_3530",
        "DISTRACTOR_DATA_SET_NAME":  "megaface_distractor_id_10000_clean",
        "BASE_DATASET":"megaface_test",
        "SLICE_DATA_SET": False,
        "START_IDX":0,
        "END_IDX":0,
        "USE_PAIRS":False,
        "USE_DISTRACTORS":True
    }
    # image_dataset = image_dataset_agedb_test
    # image_dataset = image_dataset_lfw_gender_test
    image_dataset = image_dataset_lfw_all_gender_test
    # image_dataset = image_dataset_fgnet_test
    # image_dataset = image_dataset_lfw_funneled_test
    # image_dataset = image_dataset_facescrub_3530
    # image_dataset = image_dataset_megaface_distractor_id_1000_clean
    # image_dataset = image_dataset_facescrub_3530_sliced
    # image_dataset = image_dataset_facescrub_3530_sliced_distractor_10
    # image_dataset = image_dataset_facescrub_3530_sliced_100
    # image_dataset = image_dataset_facescrub_3530_sliced_100_distractor_10
    # image_dataset = image_dataset_facescrub_3530_distractor_10000
    # image_dataset = image_dataset_facescrub_3530_sliced_100_detect
    # image_dataset = image_dataset_facescrub_3530_sliced_100_detect_retinaface

    EMBEDINGS_PATH = EMBEDINGS_PATH.replace("@@base_dataset@@",image_dataset["BASE_DATASET"])
    VERIFICATION_PATH = VERIFICATION_PATH.replace("@@base_dataset@@",image_dataset["BASE_DATASET"])
    PERSON_ATTRIBUTES_PATH = PERSON_ATTRIBUTES_PATH.replace("@@base_dataset@@",image_dataset["BASE_DATASET"])


    build_representation_flag = False
    verification_flag = False
    api_verfication_recheck = False
    api_benchmark = False
    plot_roc_curve = False
    person_attributes = True

    if build_representation_flag:
        build_representation_model_name_list = ("VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace")
        # build_representation_model_name_list = ["VGG-Face"]
        # build_representation_model_name_list = ["Dlib"]
        # build_representation_model_name_list = ["ArcFace"]
        rebuild_representation = True
        # representation_timestamp = datetime.now().strftime("%d-%b-%Y_%H_%M_%S_%f)")
        representation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        for model_name in build_representation_model_name_list:
            embedings_path =  os.path.join(EMBEDINGS_PATH,image_dataset["DATA_SET_NAME"],model_name)
            embedings_file =   os.path.join(embedings_path,"emb.npy")
            if not os.path.exists(embedings_file) or rebuild_representation :
                if not os.path.exists(embedings_file):
                    print ("\nBuild image representation with model ", model_name)
                else:
                    print ("\nRebuild image representation with model ", model_name)
                model = DeepFace.build_model(model_name)
                start = time.time()
                param = {"model": model,"model_name":model_name,  "image_dataset":image_dataset, "align": True, "detect_face": True, "enforce_detection":False, "detector_backend":"opencv"}
                # param = {"model": model,"model_name":model_name,  "image_dataset":image_dataset, "align": False, "detect_face": False, "detector_backend":None}
                # param = {"model": model,"model_name":model_name,  "image_dataset":image_dataset, "align": True, "detect_face": True, "enforce_detection":False, "detector_backend":"retinaface"}
                # param = {"model": model,"model_name":model_name,  "image_dataset":image_dataset, "align": True, "detect_face": True, "enforce_detection":False, "detector_backend":"opencv"}
                embedings = build_representation(**param)
                end = time.time()
                print("Representation building time",end - start)
                os.makedirs(embedings_path, exist_ok=True)
                np.save(embedings_file,embedings)
                info = {**param,"representation_timestamp":representation_timestamp,
                             # "start":time.strftime('%Y-%m-%d %H:%M:%S %f', time.localtime(start)),
                        "start":datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S.%f'),
                        "end":datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S.%f'),"duration":end - start}
                del info["model"]
                info_file =   os.path.join(embedings_path,"info.json")
                with open(info_file, "w") as f:
                    json.dump( info,  f, indent=2)
            else:
                print ("Image representation with model {} is already built".format(model_name))

# exit (0)
    if verification_flag:
        # verification_model_name_list = ("VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace")
        verification_model_name_list = ["ArcFace"]
        # verification_model_name_list = ["Dlib"]
        # distance_metric_list = ['cosine', 'euclidean', 'euclidean_l2']
        distance_metric_list = ['cosine']
        # distance_metric_list = ['euclidean']
        save_verification_results = True
        repeat_verification = True
        load_verification_results = False
        verification_file_name = "vrf_2021-08-06_04_15_42.npy"
        dt = datetime.now()
        # verification_timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        verification_timestamp = dt.strftime("%Y-%m-%d_%H_%M_%S")
        verification_timestamp_ = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        all_models_all_distance_verification_info =[]
        json_file =""
        for model_name in verification_model_name_list:
            all_distance_metrics_verification_results =[]
            verification_time = []
            # verification_path =  os.path.join(VERIFICATION_PATH,model_name)
            if image_dataset["USE_DISTRACTORS"]:
                verification_path =  os.path.join(VERIFICATION_PATH,
                            image_dataset["DATA_SET_NAME"]+"_"+image_dataset["DISTRACTOR_DATA_SET_NAME"],model_name)
            else:
                verification_path =  os.path.join(VERIFICATION_PATH,image_dataset["DATA_SET_NAME"],model_name)
            verification_file_pattern =   os.path.join(verification_path,"vrf*.npy")
            # ***embedings_path =  os.path.join(EMBEDINGS_PATH,model_name)
            embedings_path =  os.path.join(EMBEDINGS_PATH,image_dataset["DATA_SET_NAME"],model_name)
            embedings_file =   os.path.join(embedings_path,"emb.npy")
            if not glob.glob(verification_file_pattern) or repeat_verification :
                embeddings = np.load(embedings_file, allow_pickle=True).tolist()
                distractor_embeddings = None
                if image_dataset["USE_DISTRACTORS"]:
                    distractors_path = os.path.join(EMBEDINGS_PATH,image_dataset["DISTRACTOR_DATA_SET_NAME"],model_name)
                    distractors_file =   os.path.join(distractors_path,"emb.npy")
                    distractor_embeddings = np.load(distractors_file, allow_pickle=True).tolist()
                # probe = []
                for distance_metric in distance_metric_list:
                    print ("\nNew verification:\nmodel_name: {} , distance_metric: {}".format(model_name, distance_metric))
                    if TRASHOLD_MATRIX.get(model_name, None) is None or \
                            distance_metric not in  TRASHOLD_MATRIX.get(model_name, None):
                        raise ValueError("Trashold for model {}, distance_metric {} not defined".format(model_name,distance_metric))
                    threshold = dst.findThreshold(model_name, distance_metric)
                    param = {"image_dataset":image_dataset, "embeddings": embeddings, "distance_metric":distance_metric,
                             "threshold":threshold,"distractor_embeddings":distractor_embeddings}
                    # result_ = verification_eval(image_dataset, embeddings, distance_metric, threshold)
                    start = time.time()
                    if image_dataset["USE_PAIRS"]:
                        result_ = verification_eval_pairs(**param)
                    else:
                        result_ = verification_eval(**param)
                    end = time.time()
                    verification_time.append({"verification_timestamp":verification_timestamp_,
                                        "start":datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S.%f'),
                                        "end":datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S.%f'),
                                        "duration":end - start})
                    print("Verification time",end - start)
                    # verification_result = {"model_name":model_name,"distance_metric":distance_metric, "threshold":threshold,"result":result_}
                    verification_info = {"model_name":model_name,"distance_metric":distance_metric, "threshold":threshold}
                    verification_result = {"result":result_}

                    # eval_stat_details = get_eval_stat_details(verification_result["result"]["eval_details"])
                    # verification_metrics = clf_metrics_mean (**eval_stat_details)
                    verification_metrics = clf_metrics(**verification_result["result"]["eval_stat"])

                    print_classification_metrics(verification_metrics)
                    all_distance_metrics_verification_results.append({"verification_info":verification_info,
                        "verification_result":verification_result,"verification_metrics":verification_metrics})
                if save_verification_results:
                    verification_file =   os.path.join(verification_path,"vrf_{}.npy".format(verification_timestamp))
                    # ***print("Saved object size:{}".format(get_size(all_distance_metrics_verification_results)))
                    os.makedirs(verification_path, exist_ok=True)
                    np.save(verification_file,all_distance_metrics_verification_results)
                    info_file =   os.path.join(embedings_path,"info.json")
                    with open(info_file, "r") as f:
                        image_representation_info = json.load(f)
                    for result in all_distance_metrics_verification_results:
                        del result["verification_result"]
                    for i in range (len(verification_time)):
                        all_distance_metrics_verification_results[i]["verification_time"]=verification_time[i]
                    info = {"image_representation":image_representation_info,"verification_result":all_distance_metrics_verification_results}
                    all_models_all_distance_verification_info.append(info)
                    info_file =   os.path.join(verification_path,"info_{}.json".format(verification_timestamp))
                    json_file = info_file
                    with open(info_file, "w") as f:
                        json.dump( info,  f, indent=2)
                    print("Result saved in {}".format(verification_file))

            elif load_verification_results:
                verification_file =   os.path.join(verification_path,verification_file_name)
                # all_distance_metrics_verification_results = []
                all_distance_metrics_verification_results =np.load(verification_file, allow_pickle=True).tolist()
                # x=1
                # pass
                for result in all_distance_metrics_verification_results:
                    verification_info = result["verification_info"]
                    verification_metrics = result["verification_metrics"]
                    print ("\nSaved verification:\nmodel_name: {} , distance_metric: {}, threshold:  {}"
                           .format(verification_info["model_name"], verification_info["distance_metric"], verification_info["threshold"]))
                    print_classification_metrics(verification_metrics)

        if save_verification_results:
            fieldnames = ['json','model_name',"distance_metric","threshold",
                          "tp","tn","fp","fn","total","accuracy","precision","recall","fpr","fnr","tpr","tnr",
                          "verification_timestamp","verification_duration",
                          'dataset_name','align','detect_face','detector_backend',"representation_timestamp","representation_duration"]
            # all_models_all_distance_metrics_path = os.path.join(VERIFICATION_PATH,image_dataset["DATA_SET_NAME"])
            if image_dataset["USE_DISTRACTORS"]:
                all_models_all_distance_metrics_path =  os.path.join(VERIFICATION_PATH,
                                                  image_dataset["DATA_SET_NAME"]+"_"+image_dataset["DISTRACTOR_DATA_SET_NAME"])
            else:
                all_models_all_distance_metrics_path =  os.path.join(VERIFICATION_PATH,image_dataset["DATA_SET_NAME"])

            all_models_all_distance_metrics_file = os.path.join(all_models_all_distance_metrics_path,
                "all_models_all_distance_metrics_{}.csv".format(verification_timestamp))
            with open(all_models_all_distance_metrics_file, 'w', newline='') as csvfile:
                # writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='excel-tab')
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                info_={}
                for mdl in all_models_all_distance_verification_info:
                    for info in mdl["verification_result"]:
                        info_={}
                        head, tail = os.path.split(json_file)
                        info_['json']= tail
                        info_={**info_,**info["verification_info"]}
                        info_={**info_,**info["verification_metrics"]}
                        info_["verification_timestamp"]=info["verification_time"]["verification_timestamp"]
                        info_["verification_duration"]=info["verification_time"]["duration"]
                        info_['dataset_name']=mdl["image_representation"]["image_dataset"]["DATA_SET_NAME"]
                        info_['align']=mdl["image_representation"]["align"]
                        info_["detect_face"]=mdl["image_representation"]["detect_face"]
                        info_["detector_backend"]=mdl["image_representation"]["detector_backend"]
                        info_["representation_timestamp"]=mdl["image_representation"]["representation_timestamp"]
                        info_["representation_duration"]=mdl["image_representation"]["duration"]
                        localize_floats(info_)
                        # writer.writerow(localize_floats(info_))
                        writer.writerow(info_)

    if api_verfication_recheck:
        """
        img_file = "H:/face_recognition/megaface_test/raw/facescrub_images/Adam_Brody/Adam_Brody_241.png"
        model_name = "ArcFace"
        # img = functions.load_image(img_file)
        img = img_file
        img_my = preprocess_face_(img, target_size=(112,112), enforce_detection=False , detector_backend = 'opencv', align = True, detect_face = True)
        img_api = functions.preprocess_face(img, target_size=(112,112), enforce_detection=False , detector_backend = 'opencv', align = True)
        eq = np.array_equal(img_my, img_api)
        model = DeepFace.build_model(model_name)
        emb_my= model.predict(img_my)[0].tolist()
        emb_api = DeepFace.represent(img_path = img, model_name = model_name, model = model,enforce_detection = False,
                                        detector_backend = "opencv", align = True)
        embedings_path =  os.path.join(EMBEDINGS_PATH,image_dataset["DATA_SET_NAME"],model_name)
        embedings_file =   os.path.join(embedings_path,"emb.npy")
        embeddings = np.load(embedings_file, allow_pickle=True).tolist()

        exit (0)
        """


        # api_verfication_model_name_list = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
        api_verfication_model_name_list = ["ArcFace"]
        # distance_metric_list = ['cosine', 'euclidean', 'euclidean_l2']
        # distance_metric_list = ['euclidean_l2']
        distance_metric_list = ['cosine']
        verification_file_name = "vrf_2021-08-15_12_50_57.npy"
        for model_name in api_verfication_model_name_list:
            # verification_path =  os.path.join(VERIFICATION_PATH,image_dataset["DATA_SET_NAME"],model_name)
            if image_dataset["USE_DISTRACTORS"]:
                verification_path =  os.path.join(VERIFICATION_PATH,
                                                  image_dataset["DATA_SET_NAME"]+"_"+image_dataset["DISTRACTOR_DATA_SET_NAME"],model_name)
            else:
                verification_path =  os.path.join(VERIFICATION_PATH,image_dataset["DATA_SET_NAME"],model_name)
            verification_file =  os.path.join(verification_path,verification_file_name)
            all_distance_metrics_verification_results = np.load(verification_file, allow_pickle=True).tolist()
            embedings_path =  os.path.join(EMBEDINGS_PATH,image_dataset["DATA_SET_NAME"],model_name)
            embedings_file =   os.path.join(embedings_path,"emb.npy")
            embeddings = np.load(embedings_file, allow_pickle=True).tolist()
            for result in all_distance_metrics_verification_results:
                verification_info = result["verification_info"]
                if verification_info["distance_metric"] not in distance_metric_list:
                    continue
                verification_metrics = result["verification_metrics"]
                print ("\nSaved verification:\nmodel_name: {} , distance_metric: {}, threshold:  {}"
                       .format(verification_info["model_name"], verification_info["distance_metric"], verification_info["threshold"]))
                print_classification_metrics(verification_metrics)
                eval_details = result["verification_result"]["result"]["eval_details"]
                for eval in eval_details:

                    if eval["label_1"] == eval["label_2"]:
                        true_verification = 1
                    else:
                        true_verification = 0
                    if true_verification != eval["verification"] and true_verification == 1:
                        img_1 = eval["img_1"]
                        img_2 = eval["img_2"]
                        result = DeepFace.verify(img_1, img_2, model_name = model_name, distance_metric = verification_info["distance_metric"])
                        pass

    if api_benchmark:
        # verification_model_name_list = ("VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace")
        # verification_model_name_list = ["ArcFace"]
        # verification_model_name_list = ["Dlib"]
        verification_model_name_list = ["VGG-Face"]
        distance_metric_list = ['cosine', 'euclidean', 'euclidean_l2']
        # distance_metric_list = ['cosine']
        start_idx = 0
        probe = []
        labels = []
        with open(image_dataset["PROBE_IMAGE_FILE_LIST"]) as f:
            probe= [os.path.join(image_dataset["PROBE_IMAGE_PATH"],s.strip()) for s in f.readlines()[:]]
        end_idx = len(probe)
        if image_dataset["SLICE_DATA_SET"]:
            start_idx = image_dataset["START_IDX"]
            end_idx = image_dataset["END_IDX"]
        with open(image_dataset["PROBE_IMAGE_LABELS"]) as f:
            labels= [s.strip() for s in f.readlines()[1:]]
        pair_list =[]
        label_list =[]
        idx_1 = start_idx
        for prb_1 in probe[start_idx:end_idx]:
            idx_2 = start_idx
            for prb_2 in probe[start_idx:end_idx]:
                if idx_1 < idx_2:
                    pair_list.append([prb_1,prb_2])
                    label_list.append([labels[idx_1],labels[idx_2]])
                idx_2+=1
            idx_1+=1
        for model_name in verification_model_name_list:
            for distance_metric in distance_metric_list:
                print ("\nNew verification (API):\nmodel_name: {} , distance_metric: {}".format(model_name, distance_metric))
                start = time.time()
                result = DeepFace.verify(pair_list, model_name = model_name, distance_metric = distance_metric,align=False,enforce_detection=False)
                end = time.time()
                print("Verification time",end - start)
                idx = 0
                tp = tn = fp = fn = 0
                for vrf in result:
                    verification = result[vrf]["verified"]
                    label_1 = label_list[idx][0]
                    label_2 = label_list[idx][1]
                    if label_1 == label_2:
                        true_verification = 1
                    else:
                        true_verification = 0
                    if true_verification == verification and true_verification == 1:
                        tp+=1
                    if true_verification == verification and true_verification == 0:
                        tn+=1
                    if true_verification != verification and true_verification == 1:
                        fn+=1
                    if true_verification != verification and true_verification == 0:
                        fp+=1
                    idx+=1
                print_classification_metrics(clf_metrics(tp, tn, fp, fn))

    if plot_roc_curve:
        log_scale = True
        # verification_model_name_list = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
        verification_model_name_list = ["ArcFace"]
        # verification_model_name_list = ["Dlib"]
        # distance_metric_list = ['cosine', 'euclidean', 'euclidean_l2']
        # distance_metric_list = ['euclidean', 'euclidean_l2','cosine']
        # distance_metric_list = ['euclidean_l2','cosine']
        # distance_metric_list = ['euclidean', 'cosine']
        # distance_metric_list = ['euclidean']
        distance_metric_list = ['cosine']
        # distance_metric_list = ['euclidean_l2']
        # verification_file_name = "vrf_2021-08-07_21_18_55.npy"
        # verification_file_name = "vrf_2021-08-12_14_16_47.npy"
        # ***verification_file_name = "vrf_2021-08-15_12_50_57.npy"
        verification_file_name = "vrf_2021-08-07_21_18_55.npy"

        # ***verification_file_name = "vrf_2021-08-13_13_39_55.npy"


        # verification_file_name = "vrf_2021-08-06_22_04_40.npy"

        all_model_all_distahce_metric_auc_roc = {}
        for model in verification_model_name_list:
            all_model_all_distahce_metric_auc_roc[model] = {}
            # verification_path =  os.path.join(VERIFICATION_PATH,image_dataset["DATA_SET_NAME"],model)
            if image_dataset["USE_DISTRACTORS"]:
                verification_path =  os.path.join(VERIFICATION_PATH,
                                                  image_dataset["DATA_SET_NAME"]+"_"+image_dataset["DISTRACTOR_DATA_SET_NAME"],model)
            else:
                verification_path =  os.path.join(VERIFICATION_PATH,image_dataset["DATA_SET_NAME"],model)
            verification_file =  os.path.join(verification_path,verification_file_name)
            all_distance_metrics_verification_results =np.load(verification_file, allow_pickle=True).tolist()
            for distance_metric in distance_metric_list:
                for verification_result in all_distance_metrics_verification_results:
                    if verification_result["verification_info"]["distance_metric"] != distance_metric:
                        continue
                    else:
                        all_model_all_distahce_metric_auc_roc[model][distance_metric] = {}
                        idx_1 = []
                        idx_2 = []
                        distance = []
                        true_verification = []
                        for result in verification_result["verification_result"]["result"]["eval_details"]:
                            idx_1.append(result["idx_1"])
                            idx_2.append(result["idx_2"])
                            distance.append(result["distance"])
                            if result["label_1"]== \
                                    result["label_2"]:
                                true_verification.append(1)
                            else:
                                true_verification.append(0)
                        all_model_all_distahce_metric_auc_roc[model][distance_metric]["idx_1"]=idx_1
                        all_model_all_distahce_metric_auc_roc[model][distance_metric]["idx_2"]=idx_2
                        all_model_all_distahce_metric_auc_roc[model][distance_metric]["distance"]=distance
                        all_model_all_distahce_metric_auc_roc[model][distance_metric]["true_verification"]=true_verification
                        all_model_all_distahce_metric_auc_roc[model][distance_metric]["threshold"]=\
                            verification_result["verification_info"]["threshold"]
                        all_model_all_distahce_metric_auc_roc[model][distance_metric]["verification_metrics"]= \
                            verification_result["verification_metrics"]
                        # print("_1_",verification_result["verification_metrics"])
                        pass

        NUM_COLORS = 20

        # print("_2_",verification_result["verification_metrics"])

        # ***cm = plt.get_cmap('gist_rainbow')
        cm = plt.get_cmap('tab10')
        fig = plt.figure(figsize=(8, 6), dpi=200)
        ax = fig.add_subplot(111)
        #*** ax.set_prop_cycle(color=[cm(1.*5*i/NUM_COLORS) for i in range(NUM_COLORS)])
        # ax.set_prop_cycle(color=[cm(1.*1*i/NUM_COLORS) for i in range(NUM_COLORS)])
        ax.set_prop_cycle(color=[cm(1+i) for i in range(NUM_COLORS)])
        if log_scale:
            plt.xscale('log')
        # color_idx = 0

        test_estimator = auc.CustomClassifier__()
        for model in all_model_all_distahce_metric_auc_roc:
            for distance_metric in all_model_all_distahce_metric_auc_roc[model]:
                distance = all_model_all_distahce_metric_auc_roc[model][distance_metric]["distance"]
                true_verification = all_model_all_distahce_metric_auc_roc[model][distance_metric]["true_verification"]
                # d = np.reshape(np.array(distance),(1,len(distance)))
                distance_ = test_estimator.predict_proba(-np.array(distance))
                # distance_ = test_estimator.predict_proba(d)
                idx_1 = all_model_all_distahce_metric_auc_roc[model][distance_metric]["idx_1"]
                idx_2 = all_model_all_distahce_metric_auc_roc[model][distance_metric]["idx_2"]

                # fpr, tpr = mean_roc_curve(true_verification,distance_[:,1],idx_1,idx_2)
                fpr, tpr, thresh = metrics.roc_curve(true_verification, distance_[:,1], drop_intermediate =False)
                auc = metrics.auc(fpr, tpr)
                print("AUC:", auc)
                # print("\n".join(map(str,list(zip(fpr.tolist(), tpr.tolist(), thresh.tolist())))))

                # print("_3_",verification_result["verification_metrics"])

                # plt.plot(fpr, tpr, label='{}@{}'.format(model,distance_metric), color='g')

                # graph = ax.plot(fpr, tpr, label='{}@{}'.format(model,distance_metric))
                fpr_ = fpr[(fpr[:] > 1e-03) & (fpr[:] < 1e-01)]
                tpr_ = tpr[(fpr[:] > 1e-03) & (fpr[:] < 1e-01)]
                graph = ax.plot(fpr_, tpr_, label='{}@{}'.format(model,distance_metric))
                # graph
                th_tuned = all_model_all_distahce_metric_auc_roc[model][distance_metric]["threshold"]
                fpr_th_tuned = all_model_all_distahce_metric_auc_roc[model][distance_metric]["verification_metrics"]["fpr"]
                tpr_th_tuned = all_model_all_distahce_metric_auc_roc[model][distance_metric]["verification_metrics"]["tpr"]
                graph = ax.plot(fpr_th_tuned,tpr_th_tuned, marker="o", color='b')
                graph = ax.text(fpr_th_tuned,tpr_th_tuned-0.03, "tpr:{:.4f}\nfpr:{:.5f}\nth:{:.2f}".
                                format(tpr_th_tuned,fpr_th_tuned,th_tuned), color='b',
                                verticalalignment='top')

        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid()
        plt.legend()
        plt.show()

    if person_attributes:
        eval_person_attributes = True
        load_person_attributes = False
        # actions = ["age"]
        # actions = ["age","gender"]
        actions = ["gender"]

        if eval_person_attributes:
            print ("Person attributes evaluation")
            attributes = person_attributes_eval(image_dataset,actions = actions)
            dt = datetime.now()
            person_attributes_timestamp = dt.strftime("%Y-%m-%d_%H_%M_%S")
            person_attributes_path = os.path.join(PERSON_ATTRIBUTES_PATH,image_dataset["DATA_SET_NAME"])
            person_attributes_file =   os.path.join(person_attributes_path,"attr_{}.npy".format(person_attributes_timestamp))
            os.makedirs(person_attributes_path, exist_ok=True)
            np.save(person_attributes_file,attributes)
            report_person_attributes(attributes,person_attributes_timestamp,save_report_to_csv = True)

        if load_person_attributes:
            # person_attributes_file_name = "attr_2021-08-16_15_30_40.npy"
            person_attributes_file_name = "attr_2021-08-09_16_48_04.npy"
            person_attributes_path = os.path.join(PERSON_ATTRIBUTES_PATH,image_dataset["DATA_SET_NAME"])
            person_attributes_file =   os.path.join(person_attributes_path,person_attributes_file_name)
            attributes = np.load(person_attributes_file, allow_pickle=True).tolist()
            # person_attributes_timestamp = person_attributes_file_name.replace("attr_","").replace(".npy","")
            # report_person_attributes(attributes,person_attributes_timestamp,save_report_to_csv = True )

            # plt.hist(attributes["true_age"], alpha=0.5, label = "true age")
            plt.hist(attributes["eval_age"], alpha=0.5, label = "evaluated age")
            plt.legend(loc='upper right')
            # plt.hist(sorted(attributes["true_age"]), sorted(attributes["eval_age"]), alpha=0.5)
            plt.show()
            # plt.hist(data_2, alpha=0.5)
            # plt.scatter(attributes["true_age"],attributes["eval_age"], s= 3)
            # plt.hist(sorted(attributes["true_age"]), sorted(attributes["eval_age"]), alpha=0.5)
            # plt.show()

            print("")



exit (0)

