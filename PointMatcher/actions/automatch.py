import os.path as osp
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction
from utils.filesystem import icon_path
from pathlib import Path
import pickle
from tqdm import tqdm

import cv2 as cv


class AutoMatchAction(QAction):
    """Keypoint image i&j shown if they don't already have files with their keypoints"""

    def __init__(self, parent):
        super(AutoMatchAction, self).__init__('Automatic Match', parent)
        self.p = parent

        self.setIcon(QIcon(icon_path('fit-width')))
        self.setShortcut('Ctrl+M')
        self.triggered.connect(self.automaticMatch)
        self.setEnabled(True)



    def load_keypoints_and_descriptors(self, basePath):
        kp_pkl_path, des_pkl_path = self.p.actions.autoKeypoint.gen_kp_des_paths(basePath)

        with kp_pkl_path.open("rb") as kpfile:
            kpdata = pickle.load(kpfile)

        with des_pkl_path.open("rb") as desfile:
            desdata = pickle.load(desfile)

        return kpdata, desdata



    def automaticMatch(self, _value=False):
        view_id_i = self.p.matching.get_view_id_i()
        view_id_j = self.p.matching.get_view_id_j()

        i_path = Path(self.p.imageDir) / self.p.matching.get_filename(self.p.matching.get_view_id_i())
        j_path = Path(self.p.imageDir) / self.p.matching.get_filename(self.p.matching.get_view_id_j())

        ikp,ides = self.load_keypoints_and_descriptors(i_path)
        jkp,jdes = self.load_keypoints_and_descriptors(j_path)

        if self.p.actions.autoKeypoint.ikp is None:
            for pt in tqdm(ikp,desc="Adding top image keypoint dots..."):
                ix, iy = pt.pt
                self.p.matching.append_keypoint_in_view_i(ix,iy)
            self.p.canvas.update()


        if self.p.actions.autoKeypoint.jkp is None:
            for pt in tqdm(jkp,desc="Adding bottom image keypoint dots..."):
                jx, jy = pt.pt
                # jy += self.p.canvas.img_i_h
                self.p.matching.append_keypoint_in_view_j(jx,jy)
            self.p.canvas.update()


        bf = cv.BFMatcher()
        matches = bf.knnMatch(ides,jdes,k=2)
        # matches=sorted(matches, key= lambda x:x.distance)
        # Apply ratio test
        good = []
        for m,n in tqdm(matches,desc="filtering good matches..."):
            if m.distance < 0.75*n.distance:
                good.append([m])

        # print(good)

        for item in tqdm(good,desc="filling automatch array..."):
            # print(item)
            for mat in item:
                # print([mat.queryIdx,mat.trainIdx])
                ix,iy = ikp[mat.queryIdx].pt
                jx,jy = jkp[mat.trainIdx].pt
                # print([ix,iy,jx,jy])
                # print(self.p.canvas.img_i_h)
                # if self.p.actions.autoKeypoint.jkp is None:
                #     jy += self.p.canvas.img_i_h
                # print(self.p.matching.min_distance_in_view_i(ix,iy))
                # print(self.p.matching.min_distance_in_view_j(jx,jy))

                val, iid = self.p.matching.min_distance_in_view_i(ix,iy)
                val, jid = self.p.matching.min_distance_in_view_j(jx,jy)

                self.p.matching.append_match(iid,jid)

        self.p.canvas.update()





        # view_id_j = self.p.matching.get_next_view(view_id_j)
        # self.p.changePair(view_id_i, view_id_j)
