from deepface.commons import functions, distance as dst
import matplotlib.pyplot as plt
from deepface.basemodels import ArcFace

# from deepface import DeepFace
# result  = DeepFace.verify("img1.jpg", "img2.jpg")
# print("Is verified: ", result["verified"])
#
if __name__ == "__main__":
    model = ArcFace.loadModel()
    model_name="ArcFace"
    # model.load_weights("arcface_weights.h5")
    print("ArcFace expects ",model.layers[0].input_shape[0][1:]," inputs")
    print("and it represents faces as ", model.layers[-1].output_shape[1:]," dimensional vectors")
    target_size = model.layers[0].input_shape[0][1:3]
    img1_path = "../IMG/img1.jpg"
    img2_path = "../IMG/img2.jpg"
    img3_path = "../IMG/img3.jpg"
    img4_path = "../IMG/img4.jpg"
    img5_path = "../IMG/img5.jpg"
    img6_path = "../IMG/img6.jpg"
    detector_backend = 'opencv'
    # detector_backend = 'retinaface'

    img1 = functions.preprocess_face(img1_path, target_size = target_size, detector_backend = detector_backend)

    # img1 = functions.detect_face(img1, detector_backend = detector_backend)

    img2 = functions.preprocess_face(img2_path, target_size = target_size, detector_backend = detector_backend)
    # img3 = functions.preprocess_face(img3_path, target_size = target_size, detector_backend = detector_backend)
    # img4 = functions.preprocess_face(img4_path, target_size = target_size, detector_backend = detector_backend)
    # img5 = functions.preprocess_face(img5_path, target_size = target_size, detector_backend = detector_backend)
    # img6 = functions.preprocess_face(img6_path, target_size = target_size, detector_backend = detector_backend)
    fig = plt.figure(figsize = (20, 20))

    ax1 = fig.add_subplot(1,6,1)
    plt.axis('off')
    plt.imshow(img1[0][:,:,::-1])

    ax1 = fig.add_subplot(1,6,2)
    plt.axis('off')
    plt.imshow(img2[0][:,:,::-1])
    #
    # ax1 = fig.add_subplot(1,6,3)
    # plt.axis('off')
    # plt.imshow(img3[0][:,:,::-1])
    #
    # ax1 = fig.add_subplot(1,6,4)
    # plt.axis('off')
    # plt.imshow(img4[0][:,:,::-1])
    #
    # ax1 = fig.add_subplot(1,6,5)
    # plt.axis('off')
    # plt.imshow(img5[0][:,:,::-1])
    #
    # ax1 = fig.add_subplot(1,6,6)
    # plt.axis('off')
    # plt.imshow(img6[0][:,:,::-1])

    plt.show()
    img1_embedding = model.predict(img1)[0]
    img2_embedding = model.predict(img2)[0]
    print(img1_embedding.shape)
    from deepface.commons import distance as dst

    # metric = 'euclidean'
    #
    # if metric == 'cosine':
    #     distance = dst.findCosineDistance(img1_embedding, img2_embedding)
    # elif metric == 'euclidean':
    #     distance = dst.findEuclideanDistance(img1_embedding, img2_embedding)
    # elif metric == 'euclidean_l2':
    #     distance = dst.findEuclideanDistance(dst.l2_normalize(img1_embedding), dst.l2_normalize(img2_embedding))

    # print(distance)


    distance = dst.findCosineDistance(img1_embedding, img2_embedding)
    threshold = dst.findThreshold(model_name, 'cosine')
    print('cosine:'+str(distance)+"  threshold:"+str(threshold))

    distance = dst.findEuclideanDistance(img1_embedding, img2_embedding)
    threshold = dst.findThreshold(model_name, 'euclidean')
    print('euclidean:'+str(distance)+"  threshold:"+str(threshold))

    distance = dst.findEuclideanDistance(dst.l2_normalize(img1_embedding), dst.l2_normalize(img2_embedding))
    # distance = dst.l2_normalize(dst.l2_normalize(img1_embedding), dst.l2_normalize(img2_embedding))
    threshold = dst.findThreshold(model_name, 'euclidean_l2')
    print('euclidean_l2:'+str(distance)+"  threshold:"+str(threshold))