import os.path as osp
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction
from utils.filesystem import icon_path
from pathlib import Path
import cv2 as cv
import copyreg
import pickle
from tqdm import tqdm
from collections import Counter

def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


class HideUnmatchedPointsAction(QAction):
    """Keypoint image i&j shown if they don't already have files with their keypoints"""

    def __init__(self, parent):
        super(HideUnmatchedPointsAction, self).__init__('Hide Unmatched', parent)
        self.p = parent

        self.setIcon(QIcon(icon_path('eye')))
        self.setShortcut('Ctrl+H')
        self.triggered.connect(self.hideunmatchedpoints)
        self.setEnabled(True)

        self.i_unmatchedkpidxs = Counter()
        self.j_unmatchedkpidxs = Counter()
        self.i_matchedkpidxs = Counter()
        self.j_matchedkpidxs = Counter()

    def hideunmatchedpoints(self):

        # get the indexes (which are also the ids) of points which have matches
        for matches in tqdm(self.p.matching._matches.values(),desc="walking match id values"):
            i_idx,j_idx = matches
            self.i_matchedkpidxs[i_idx] += 1
            self.j_matchedkpidxs[j_idx] += 1

        # loop through all the points testing for membership in matched id sets, and if they don't exist then
        # remove them

        for pt in tqdm(self.p.matching.get_keypoints_i(),desc="walking top keypoints..."):
            if pt['id'] in self.i_matchedkpidxs:
                continue
            else:
                self.i_unmatchedkpidxs[pt['id']] += 1

        for idx in tqdm(self.i_unmatchedkpidxs,desc="cleaning unmatched keypoints in i"):
            self.p.matching.remove_keypoint_in_view_i(idx)
        
        self.p.canvas.update()


        for pt in tqdm(self.p.matching.get_keypoints_j(),desc="walking bottom keypoints..."):
            if pt['id'] in self.j_matchedkpidxs:
                continue
            else:
                self.j_unmatchedkpidxs[pt['id']] += 1
                
        for idx in tqdm(self.j_unmatchedkpidxs,desc="cleaning unmatched keypoints in j"):
            self.p.matching.remove_keypoint_in_view_j(idx)

        self.p.canvas.update()
