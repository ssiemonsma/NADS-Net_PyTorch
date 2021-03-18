import torch
from torchvision import transforms
from model import NADS_Net
import cv2
import math
import time
import numpy as np
from config_reader import config_reader
import tkinter
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
from GUI import GUI
from numba import jit, njit, prange
import matplotlib.pyplot as plt

#Body joint pairs for Part Affinity Field
limbSeq = np.array([[4,3],[3,2],[2,1],[4,5],[5,6],[6,7],[4,8],[4,9]])
# x and y direction of part affinity field paired
mapIdx = np.array([[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]])

#This code is to add padding to the input image if the image height and weight is not a multiple of 32 or (2^5)
def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

# # # Original function
# def nonmaximum_suppression(heatmap, map_Gau):
#     all_peaks = []
#     peak_counter = 0
#
#     for part in range(9):
#         map_ori = heatmap[:, :, part]
#         map = map_Gau[:, :, part]
#         map_left = np.zeros(map.shape)
#         map_left[1:, :] = map[:-1, :]
#         map_right = np.zeros(map.shape)
#         map_right[:-1, :] = map[1:, :]
#         map_up = np.zeros(map.shape)
#         map_up[:, 1:] = map[:, :-1]
#         map_down = np.zeros(map.shape)
#         map_down[:, :-1] = map[:, 1:]
#         peaks_binary = np.logical_and.reduce(
#             (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > 0.1))
#         peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
#         peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
#         id = range(peak_counter, peak_counter + len(peaks))
#         peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
#         all_peaks.append(peaks_with_score_and_id)
#         peak_counter += len(peaks)
#
#     return all_peaks

# # Numba with no parallelization
# @njit(cache=True)
# def nonmaximum_suppression(heatmap, map_Gau):
#     all_peaks = []
#     peak_counter = 0
#
#     for part in range(9):
#         map_ori = heatmap[:, :, part]
#         map = map_Gau[:, :, part]
#         map_left = np.zeros(map.shape)
#         map_left[1:, :] = map[:-1, :]
#         map_right = np.zeros(map.shape)
#         map_right[:-1, :] = map[1:, :]
#         map_up = np.zeros(map.shape)
#         map_up[:, 1:] = map[:, :-1]
#         map_down = np.zeros(map.shape)
#         map_down[:, :-1] = map[:, 1:]
#         # peaks_binary = np.logical_and.reduce(
#         #     (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > 0.1))
#         peaks_binary = (map >= map_left) & (map >= map_right) & (map >= map_up) & (map >= map_down) & (map > 0.1)
#         peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
#         peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
#         # id = range(peak_counter, peak_counter + len(peaks))
#         id = np.arange(peak_counter, peak_counter + len(peaks))
#         peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
#         all_peaks.append(peaks_with_score_and_id)
#         peak_counter += len(peaks)
#
#     return all_peaks

# Numba with parallelization (Fastest yet!)
@njit(parallel=True, cache=True, fastmath=True)
def nonmaximum_suppression(heatmap, map_Gau):
    all_peaks = [np.zeros((2,4)) for _ in range(9)]  # We need placeholders of the proper type to keep Numba happy about the list

    for part in prange(9):
        map_ori = heatmap[:, :, part]
        map = map_Gau[:, :, part]
        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = (map >= map_left) & (map >= map_right) & (map >= map_up) & (map >= map_down) & (map > 0.1)

        peaks = np.nonzero(peaks_binary)
        # scores = map_ori[peaks[0], peaks[1]]
        scores = np.zeros(len(peaks[0]))  # Fix later so above line works with Numba
        # peaks_with_score_and_id = np.column_stack((peaks[1], peaks[0], map_ori[peaks], np.zeros(len(peaks[0]))))
        peaks_with_score_and_id = np.column_stack((peaks[1], peaks[0], scores, np.zeros(len(peaks[0]))))

        all_peaks[part] = peaks_with_score_and_id

    # Why is this step needed?
    peak_num = 0
    for i in range(9):
        for j in range(len(all_peaks[i])):
            all_peaks[i][j,3] = peak_num
            peak_num += 1

    return all_peaks

#
# @njit(parallel = True)
# def nonmaximum_suppression(heatmap, map_Gau):
#     map = heatmap[:,:,:9]
#
#     all_peaks = [np.zeros((2,5)) for _ in range(9)]
#
#     map_left = np.zeros(map.shape)
#     map_left[1:, :] = map[:-1, :, :]
#     map_right = np.zeros(map.shape)
#     map_right[:-1, :] = map[1:, :, :]
#     map_up = np.zeros(map.shape)
#     map_up[:, 1:] = map[:, :-1, :]
#     map_down = np.zeros(map.shape)
#     map_down[:, :-1] = map[:, 1:, :]
#
#     peaks_binary = (map >= map_left) & (map >= map_right) & (map >= map_up) & (map >= map_down) & (map > 0.1)
#     # peaks = np.argwhere(peaks_binary)
#     peaks = np.nonzero(peaks_binary)
#     peaks = np.column_stack((peaks[1], peaks[0], np.zeros(len(peaks[0])), np.arange(len(peaks[0])), peaks[2]))  # Fix scores later
#     # peaks = np.column_stack((np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)))
#     # peaks = [peaks[peaks[:,4] == i] for i in np.unique(peaks[:,4])]
#
#     # for i in prange(len(peaks)):
#     #     all_peaks[int(peaks[i][0,4])] = peaks[i]
#
#     for i in prange(9):
#         all_peaks[i] = peaks[peaks[:,4] == i]
#
#     # Why is this step needed?
#     peak_num = 0
#     for i in range(9):
#         for j in range(len(all_peaks[i])):
#             all_peaks[i][j,3] = peak_num
#             peak_num += 1
#
#     return all_peaks

# # Using Cupy
# def nonmaximum_suppression(heatmap, map_Gau):
#     map = cp.array(map_Gau[:,:,:9])
#
#     all_peaks = [None for _ in range(9)]
#
#     map_left = cp.zeros(map.shape)
#     map_left[1:, :] = map[:-1, :, :]
#     map_right = cp.zeros(map.shape)
#     map_right[:-1, :] = map[1:, :, :]
#     map_up = cp.zeros(map.shape)
#     map_up[:, 1:] = map[:, :-1, :]
#     map_down = cp.zeros(map.shape)
#     map_down[:, :-1] = map[:, 1:, :]
#
#     peaks_binary = (map >= map_left) & (map >= map_right) & (map >= map_up) & (map >= map_down) & (map > 0.1)
#     # peaks = np.argwhere(peaks_binary)
#     peaks = cp.nonzero(peaks_binary)
#     peaks = cp.column_stack((peaks[1], peaks[0], heatmap[peaks], cp.arange(len(peaks[0])), peaks[2]))
#     # peaks = np.column_stack((np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)))
#     # peaks = [peaks[peaks[:,4] == i] for i in np.unique(peaks[:,4])]
#
#     # for i in prange(len(peaks)):
#     #     all_peaks[int(peaks[i][0,4])] = peaks[i]
#
#     for i in range(9):
#         all_peaks[i] = cp.asnumpy(peaks[peaks[:,4] == i])
#
#     # Why is this step needed?
#     peak_num = 0
#     for i in range(9):
#         for j in range(len(all_peaks[i])):
#             all_peaks[i][j,3] = peak_num
#             peak_num += 1
#
#     return all_peaks

# Not working yet
# @njit(parallel = True)
# def nonmaximum_suppression(heatmap, map_Gau):
#     all_peaks = []
#     peak_counter = 0
#
#     for part in prange(9):
#         map_ori = heatmap[:, :, part]
#         map = map_Gau[:, :, part]
#         map_left = np.zeros(map.shape)
#         map_left[1:, :] = map[:-1, :]
#         map_right = np.zeros(map.shape)
#         map_right[:-1, :] = map[1:, :]
#         map_up = np.zeros(map.shape)
#         map_up[:, 1:] = map[:, :-1]
#         map_down = np.zeros(map.shape)
#         map_down[:, :-1] = map[:, 1:]
#         # peaks_binary = np.logical_and.reduce(
#         #     (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > 0.1))
#         peaks_binary = (map >= map_left) & (map >= map_right) & (map >= map_up) & (map >= map_down) & (map > 0.1)
#         peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
#         peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
#         # id = range(peak_counter, peak_counter + len(peaks))
#         id = np.arange(peak_counter, peak_counter + len(peaks))
#         peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
#         all_peaks.append(peaks_with_score_and_id)
#         peak_counter += len(peaks)
#
#     return all_peaks

# @njit(parallel = False, cache=True, fastmath=True)
# def process_PAF(paf, all_peaks, oriImg):
#     # connection_all = []
#     connection_all = [np.zeros((2, 5)) for _ in range(8)]  # We need placeholders of the proper type to keep Numba happy about the list
#
#     # special_k = []
#     special_k = np.zeros(8)
#     mid_num = 10
#
#     for k in prange(8):
#         score_mid = paf[:, :, mapIdx[k]]
#         candA = all_peaks[limbSeq[k][0] - 1]
#         candB = all_peaks[limbSeq[k][1] - 1]
#         nA = len(candA)
#         nB = len(candB)
#         indexA, indexB = limbSeq[k]
#         if (nA != 0 and nB != 0):
#             connection_candidate = []
#             for i in range(nA):
#                 for j in range(nB):
#                     vec = np.subtract(candB[j][:2], candA[i][:2])
#                     norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
#                     if norm == 0:
#                         continue
#                     vec = np.divide(vec, norm)
#
#                     # startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num)))
#                     startend = np.column_stack((np.linspace(candA[i][0], candB[j][0], mid_num), np.linspace(candA[i][1], candB[j][1], mid_num)))
#
#                     vec_x = np.array(
#                         [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
#
#                         for I in range(len(startend))])
#
#                     vec_y = np.array(
#                         [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
#                          for I in range(len(startend))])
#
#                     score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
#                     score_with_dist_prior = np.sum(score_midpts) / len(score_midpts) + min(
#                         0.5 * oriImg.shape[0] / norm - 1, 0)
#                     criterion1 = len(np.nonzero(score_midpts > 0.05)[0]) > 0.8 * len(score_midpts)  # 0.05 was params['thre2']
#                     criterion2 = score_with_dist_prior > 0
#
#                     if criterion1 and criterion2:
#                         connection_candidate.append([i, j, score_with_dist_prior,
#                                                      score_with_dist_prior + candA[i][2] + candB[j][2]])
#             # Numpy wouldn't allow caching when using Python's list sorter, so we needed a NumPy implementation
#             connection_candidate = np.array(connection_candidate)
#             sorted_indices = np.argsort(connection_candidate[:, 2])[::-1]   # Sort in reverse order
#             connection_candidate = connection_candidate[sorted_indices]
#             connection = np.zeros((0, 5))
#
#             for c in range(len(connection_candidate)):
#                 i, j, s = connection_candidate[c][0:3]
#                 test1 = j in list(connection[:, 4])
#                 test2 = i in list(connection[:, 3])
#                 if (not test1 and not test2):
#                     connection = np.vstack((connection, np.expand_dims(np.array([candA[int(i),3], candB[int(j),3], s, i, j]), 0)))  # Why does Numba think these were not ints in the finst place
#                     if (len(connection) >= min(nA, nB)):
#                         break
#
#             connection_all[k] = connection
#         else:
#             special_k[k] = 1
#
#     special_k = np.where(special_k)
#
#     return connection_all, special_k

@njit(parallel = False, cache=True, fastmath=True)
def process_PAF(paf, all_peaks, oriImg):
    # connection_all = []
    connection_all = [np.zeros((2, 5)) for _ in range(8)]  # We need placeholders of the proper type to keep Numba happy about the list

    # special_k = []
    special_k = np.zeros(8)
    mid_num = 10

    for k in prange(8):
        score_mid = paf[:, :, mapIdx[k]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    # startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                    startend = np.column_stack((np.linspace(candA[i][0], candB[j][0], mid_num), np.linspace(candA[i][1], candB[j][1], mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \

                        for I in range(len(startend))])

                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = np.sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > 0.05)[0]) > 0.8 * len(score_midpts)  # 0.05 was params['thre2']
                    criterion2 = score_with_dist_prior > 0

                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])
            # Numpy wouldn't allow caching when using Python's list sorter, so we needed a NumPy implementation
            connection_candidate = np.array(connection_candidate)
            sorted_indices = np.argsort(connection_candidate[:, 2])[::-1]   # Sort in reverse order
            connection_candidate = connection_candidate[sorted_indices]

            connection = []

            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if len(connection) > 0:
                    i_in_connection = i in [sublist[3] for sublist in connection]
                    j_in_connection = j in [sublist[4] for sublist in connection]

                if (len(connection) == 0) or (not i_in_connection and not j_in_connection):
                    connection.append([candA[int(i), 3], candB[int(j), 3], s, i, j])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all[k] = np.array(connection)
        else:
            special_k[k] = 1

    special_k = np.where(special_k)

    return connection_all, special_k

# @njit(parallel = False, cache=True, fastmath=True)
def assemble_skeletons(connection_all, candidate, subset, special_k):
    for k in range(len(mapIdx)):
        # if k not in special_k:
        if True:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = limbSeq[k] - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 7:
                    row = -1 * np.ones(11)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = np.sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur (any subset that has less than 4 components)
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    return subset

#Preprocess the input image and generate heatmaps from the Deep model
# @profile
def process(input_image, params, model_params):
    k_scaler = 1
    oriImg = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    #Scale for resizing the input image
    fx1 = (288/oriImg.shape[0])
    fy1 = (288/oriImg.shape[1])

    imageToTest = cv2.resize(oriImg, (0, 0), fx=fy1, fy=fy1, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])
    input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

    #Normalize image same as ResNet-50 normalization since we are transfer learning ResNet-50
    input_img[:, :, 0] -= 103.939
    input_img[:, :, 1] -= 116.779
    input_img[:, :, 2] -= 123.68

    start = time.time()
    #Insert input image and predict

    with torch.no_grad():
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_img = torch.unsqueeze(preprocess(np.flip(np.squeeze(input_img), 2)/255), 0)
        heatmap, paf, seatbelt = net(input_img.to(torch.device("cuda:7")))
    print ("prediction", time.time() - start)
    # plt.imshow(heatmap[0,0,:,:])
    # plt.show()

    heatmap = np.squeeze(heatmap.cpu().numpy())
    paf = np.squeeze(paf.cpu().numpy())
    seatbelt = np.squeeze(seatbelt.cpu().numpy())

    oriImg = cv2.resize(imageToTest, (0, 0), fx=1/k_scaler, fy=1/k_scaler, interpolation=cv2.INTER_CUBIC)

    #resize the heatmap and PAF to original size
    heatmap = cv2.resize(heatmap, (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    heatmap = cv2.resize(heatmap, (imageToTest.shape[1], imageToTest.shape[0]), interpolation=cv2.INTER_CUBIC)

    paf = cv2.resize(paf, (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    paf = cv2.resize(paf, (imageToTest.shape[1], imageToTest.shape[0]), interpolation=cv2.INTER_CUBIC)


#     #Non-maximum suppression to find the peak point from heatmaps
#     start = time.time()
#     map_Gau = cv2.GaussianBlur(heatmap,(5,5),2)
#
#     all_peaks = nonmaximum_suppression(heatmap, map_Gau) # Numba-accelerated
#     print("1", time.time() - start)
#
# ### Process part affinity field and find true connection between body joints
#     start = time.time()
#
#     connection_all, special_k = process_PAF(paf, all_peaks, oriImg)
#
#     subset = -1 * np.ones((0, 11))
#     candidate = np.array([item for sublist in all_peaks for item in sublist])
#
#     print("2", time.time() - start)
#
#     start = time.time()
#     subset = assemble_skeletons(connection_all, candidate, subset, special_k)
#     print ("3", time.time() - start)

    start = time.time()

    canvas = cv2.resize(input_image, (0, 0), fx=fy1, fy=fy1, interpolation=cv2.INTER_CUBIC)

    #Process Seatbelt Segmentation
    # Uncomment this section if you don't want to show Seatbelt segmentation

    seatbelt = seatbelt[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3]]
    seatbelt = cv2.resize(seatbelt, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_CUBIC)

    thres = 0.01
    seatbelt[seatbelt>thres] = 255
    seatbelt[seatbelt<=thres] = 0
    seatbelt= seatbelt.astype(int)
    p = (seatbelt == 0)

    canvas[:,:,0] = np.where(p, canvas[:,:,0], 0)
    canvas[:,:,1] = np.where(p, canvas[:,:,1], 0)
    canvas[:,:,2] = np.where(p, canvas[:,:,2], 255)

# Visualize the detected body joints and skeletons
#     keypoints=[]
#     for s in subset:
#         keypoint_indexes = s[:9]
#         person_keypoint_coordinates = []
#         for index in keypoint_indexes:
#             if index == -1:
#                 # "No candidate for keypoint"
#                 X, Y = 0, 0
#             else:
#                 X, Y = candidate[index.astype(int)][:2]
#             person_keypoint_coordinates.append((X, Y))
#         keypoints.append((person_keypoint_coordinates, 1 - 1.0 / s[9]))
#
#     kp_sks = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[3,9],[5,8],[8,9]]
#     kp_sks = np.array(kp_sks)-1
#     clrs = (124,252,0)
#     for ind, key in enumerate(keypoints):
#         keys = []
#         for i in key[0]:
#             keys.append(i[0])
#             keys.append(i[1])
#         x = np.array(keys[0::2])
#         y = np.array(keys[1::2])
#
#         for sk in kp_sks:
#             if np.all(x[sk]>0):
#                 cv2.line(canvas, (int(x[sk[0]]*k_scaler),int(y[sk[0]]*k_scaler)), (int(x[sk[1]]*k_scaler),int(y[sk[1]]*k_scaler)), clrs, 3)
#         for k in range(9):
#             cv2.circle(canvas,((int(x[k]*k_scaler)),(int(y[k]*k_scaler))) , 3, (0,128,0), thickness=-1)
#     print(canvas.shape)

    return canvas


if __name__ == '__main__':
    device = torch.device("cuda:7")
    net = NADS_Net(True, True, False).to(device)
    net.load_state_dict(torch.load('weights_training_with_new_segmentation_branch.pth'), strict=False)

    params, model_params = config_reader()

    num_frames_processed = 0
    total_processing_time = 0
    compiled_numba_functions = False

    class Pose(GUI):
        def __init__(self, window, window_title, video_source):
            GUI.__init__(self, window, window_title, video_source)

        def process(self, frame):
            global num_frames_processed
            global total_processing_time
            global compiled_numba_functions
            start = time.time()
            result = process(frame, params, model_params)
            processing_time = time.time() - start

            if compiled_numba_functions == True:
                total_processing_time += processing_time
                num_frames_processed += 1
                print("running processing fps: ", num_frames_processed/total_processing_time)
            else:
                compiled_numba_functions = True

            tt = 1.0/processing_time
            print("current processing fps: ", tt)

            return result, tt

        def open(self):
            inputFileName = filedialog.askopenfilename()
            self.window.destroy()
            Pose(tkinter.Tk(), "Tkinter and OpenCV", inputFileName)

        def update(self):
            ret, frame = self.vid.get_frame()
            tt = 30
            if ret:
                if self.value == 0:
                    out_frame, tt = self.No_process(frame)
                elif self.value == 1:
                    out_frame, tt = self.process(frame)
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(out_frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
                self.box.delete(1.0,'end')
                self.box.insert('insert', str(round(tt)))
            else:
                raise SystemExit(0)
                self.window.destroy()

            self.window.after(self.delay, self.update)

    #Change the input video file if necessary.
    Pose(tkinter.Tk(), "Tkinter and OpenCV", 'inputs/test1.mp4')
    #comment the above line and uncomment line of code below to run the file for Webcam
    # Pose(tkinter.Tk(), "Tkinter and OpenCV", 0) ##For webcam