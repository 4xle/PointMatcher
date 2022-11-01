import os.path as osp
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction
from utils.filesystem import icon_path
from pathlib import Path
import pickle
from tqdm import tqdm

import cv2 as cv
import numpy as np


class ComputeAndShowManualHomographyAction(QAction):
    """Keypoint image i&j shown if they don't already have files with their keypoints"""

    def __init__(self, parent):
        super(ComputeAndShowManualHomographyAction, self).__init__('Compute Manual\n Homography', parent)
        self.p = parent

        self.setIcon(QIcon(icon_path('fit-width')))
        self.setShortcut('Ctrl+H')
        self.triggered.connect(self.computeHomographyOfManualPoints)
        self.setEnabled(True)



    def load_keypoints_and_descriptors(self, basePath):
        kp_pkl_path, des_pkl_path = self.p.actions.autoKeypoint.gen_kp_des_paths(basePath)

        with kp_pkl_path.open("rb") as kpfile:
            kpdata = pickle.load(kpfile)

        with des_pkl_path.open("rb") as desfile:
            desdata = pickle.load(desfile)

        return kpdata, desdata



    def computeHomographyOfManualPoints(self, _value=False):
        view_id_i = self.p.matching.get_view_id_i()
        view_id_j = self.p.matching.get_view_id_j()


        # print(self.p.matching._matches)
        ptIds = [v for v in self.p.matching._matches.values()]
        # print(ptIds)

        iptIds = [x[0] for x in ptIds]
        jptIds = [x[1] for x in ptIds]

        ikps = self.p.matching.get_keypoints_i()
        jkps = self.p.matching.get_keypoints_j()

        # print(ikps)
        # print(jkps)


        iptCoords = [ikps[self.p.matching.find_keypoint_idx(self.p.matching._view_i['keypoints'], x)]['pos'] for x in iptIds]
        jptCoords = [jkps[self.p.matching.find_keypoint_idx(self.p.matching._view_j['keypoints'], x)]['pos'] for x in jptIds]

        # print(iptCoords)
        # print(jptCoords)

        pts1 = np.int32(iptCoords)
        pts2 = np.int32(jptCoords)

        # F,mask = cv.findFundamentalMat(pts1,pts2,cv.USAC_DEFAULT)
        M, mask = cv.findHomography(pts1,pts2)
        # print(M)
        matchesMask = mask.ravel().tolist()


        h,w,d = self.p.canvas.img_i.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)

        debugImage = cv.polylines(self.p.canvas.img_j, [np.int32(dst)],True,255,3,cv.LINE_AA)
        cv.imshow("test-manual", debugImage)
        cv.waitKey(1)
        # print(F)
        # print(mask)


class ComputeAndShowRootSIFTUSACHomographyAction(QAction):
    """Keypoint image i&j shown if they don't already have files with their keypoints"""

    def __init__(self, parent):
        super(ComputeAndShowRootSIFTUSACHomographyAction, self).__init__('Compute RootSIFT\n USAC Homography', parent)
        self.p = parent

        self.setIcon(QIcon(icon_path('fit-width')))
        self.setShortcut('Ctrl+H')
        self.triggered.connect(self.computeHomographyOfRootSIFTUSACPoints)
        self.setEnabled(True)



    def load_keypoints_and_descriptors(self, basePath):
        kp_pkl_path, des_pkl_path = self.p.actions.autoKeypoint.gen_kp_des_paths(basePath)

        with kp_pkl_path.open("rb") as kpfile:
            kpdata = pickle.load(kpfile)

        with des_pkl_path.open("rb") as desfile:
            desdata = pickle.load(desfile)

        return kpdata, desdata



    def computeHomographyOfRootSIFTUSACPoints(self, _value=False):

        self.p.actions.clearAllData.clearalldata()
        self.p.actions.autoMatch.automaticMatch()
        self.p.actions.hideUnmatched.hideunmatchedpoints()
        self.p.actions.computeManualPointHomography.computeHomographyOfManualPoints()
        # view_id_i = self.p.matching.get_view_id_i()
        # view_id_j = self.p.matching.get_view_id_j()


        # print(self.p.matching._matches)
        # ptIds = [v for v in self.p.matching._matches.values()]
        # print(ptIds)

        # iptIds = [x[0] for x in ptIds]
        # jptIds = [x[1] for x in ptIds]

        # ikps = self.p.matching.get_keypoints_i()
        # jkps = self.p.matching.get_keypoints_j()


        # iptCoords = [ikps[x]['pos'] for x in iptIds]
        # jptCoords = [jkps[x]['pos'] for x in jptIds]

        # print(iptCoords)
        # print(jptCoords)

        # pts1 = np.int32(iptCoords)
        # pts2 = np.int32(jptCoords)

        # # F,mask = cv.findFundamentalMat(pts1,pts2,cv.USAC_DEFAULT)
        # M, mask = cv.findHomography(pts1,pts2, cv.USAC_DEFAULT,5.0)
        # print(M)
        # matchesMask = mask.ravel().tolist()


        # h,w,d = self.p.canvas.img_i.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv.perspectiveTransform(pts,M)

        # debugImage = cv.polylines(self.p.canvas.img_j, [np.int32(dst)],True,255,3,cv.LINE_AA)
        # cv.imshow("test", debugImage)
        # cv.waitKey(1)


            # print(v)

        # i_path = Path(self.p.imageDir) / self.p.matching.get_filename(self.p.matching.get_view_id_i())
        # j_path = Path(self.p.imageDir) / self.p.matching.get_filename(self.p.matching.get_view_id_j())

        # ikp,ides = self.load_keypoints_and_descriptors(i_path)
        # jkp,jdes = self.load_keypoints_and_descriptors(j_path)

        # if self.p.actions.autoKeypoint.ikp is None:
        #     for pt in tqdm(ikp,desc="Adding top image keypoint dots..."):
        #         ix, iy = pt.pt
        #         self.p.matching.append_keypoint_in_view_i(ix,iy)
        #     self.p.actions.autoKeypointikp = ikp
        # self.p.canvas.update()


        # if self.p.actions.autoKeypoint.jkp is None:
        #     for pt in tqdm(jkp,desc="Adding bottom image keypoint dots..."):
        #         jx, jy = pt.pt
        #         # jy += self.p.canvas.img_i_h
        #         self.p.matching.append_keypoint_in_view_j(jx,jy)
        #     self.p.actions.autoKeypoint.jkp = jkp
        # self.p.canvas.update()


        # bf = cv.BFMatcher()
        # matches = bf.knnMatch(ides,jdes,k=2)
        # # matches=sorted(matches, key= lambda x:x.distance)
        # # Apply ratio test
        # good = []
        # pts1 = []
        # pts2 = []
        # # matchesMask = [[0, 0] for i in range(len(matches))]

        # for i, (m,n) in tqdm(enumerate(matches),desc="filtering good matches..."):
        #     if m.distance < 0.75*n.distance:
        #         good.append([m])
        #         pts1.append(ikp[m.queryIdx].pt)
        #         pts2.append(jkp[m.trainIdx].pt)

        # print(good)


        # for item in tqdm(good,desc="filling pt match arrays for inlier filtering..."):
        #     # print(item)
        #     for mat in item:
        #         # print([mat.queryIdx,mat.trainIdx])
        #         ix,iy = ikp[mat.queryIdx].pt
        #         jx,jy = jkp[mat.trainIdx].pt


        # pts1 = np.int32(pts1)
        # pts2 = np.int32(pts2)

        # # F,mask = cv.findFundamentalMat(pts1,pts2,cv.USAC_DEFAULT)
        # M, mask = cv.findHomography(pts1,pts2, cv.USAC_DEFAULT,5.0)
        # # print(M)
        # matchesMask = mask.ravel().tolist()


        # h,w,d = self.p.canvas.img_i.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv.perspectiveTransform(pts,M)

        # debugImage = cv.polylines(self.p.canvas.img_j, [np.int32(dst)],True,255,3,cv.LINE_AA)
        # cv.imshow("test", debugImage)
        # cv.waitKey(1)
        # print(F)
        # print(mask)


        # pts1 = pts1[mask.ravel() == 1]
        # pts2 = pts2[mask.ravel() == 1]
        # # print("Number of pts1 inlier points: ",+len(pts1))
        # # print("Number of pts2 inlier points: ", +len(pts2))

        # # pts1 = pts1[mask.ravel() == 1]
        # # pts2 = pts2[mask.ravel() == 1]

        # for a,b in zip(pts1,pts2):
        #     # print(a,b)
        #     ix,iy = a
        #     jx,jy = b


        #         # print([ix,iy,jx,jy])
        #         # print(self.p.canvas.img_i_h)
        #         # if self.p.actions.autoKeypoint.jkp is None:
        #         #     jy += self.p.canvas.img_i_h
        #         # print(self.p.matching.min_distance_in_view_i(ix,iy))
        #         # print(self.p.matching.min_distance_in_view_j(jx,jy))

        #     val, iid = self.p.matching.min_distance_in_view_i(ix,iy)
        #     val, jid = self.p.matching.min_distance_in_view_j(jx,jy)

        #     self.p.matching.append_match(iid,jid)

        # self.p.canvas.update()





        # view_id_j = self.p.matching.get_next_view(view_id_j)
        # self.p.changePair(view_id_i, view_id_j)
