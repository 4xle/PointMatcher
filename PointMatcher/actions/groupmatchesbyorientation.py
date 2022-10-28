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
from math import atan2,degrees

def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


class GroupMatchesByOrientationAction(QAction):
    """Keypoint image i&j shown if they don't already have files with their keypoints"""

    def __init__(self, parent):
        super(GroupMatchesByOrientationAction, self).__init__('Group Matches\nBy Orientation', parent)
        self.p = parent

        self.setIcon(QIcon(icon_path('eye')))
        self.setShortcut('Ctrl+L')
        self.triggered.connect(self.groupmatchesbyorientation)
        self.setEnabled(True)

        self.iBook = {}
        self.jBook = {}



    def groupmatchesbyorientation(self):

        for kp in self.p.matching.get_keypoints_i():
            self.iBook[kp['id']] = kp

        for kp in self.p.matching.get_keypoints_j():
            self.jBook[kp['id']] = kp

        print(self.p.matching._matches)

        for match_idx, match in self.p.matching._matches.items():
            i_idx,j_idx = match
            if i_idx is None:
                self.p.matching.remove_keypoint_in_view_j(j_idx)
                continue
            if j_idx is None:
                self.p.matching.remove_keypoint_in_view_i(i_idx)
                continue
        
            ikp_posx,ikp_posy = self.iBook[i_idx]['pos']
            jkp_posx,jkp_posy = self.jBook[j_idx]['pos']


            orientationHeadingFromJToI = -1 * degrees(atan2(ikp_posy - (jkp_posy+self.p.canvas.img_i_h),ikp_posx-jkp_posx))
            print(i_idx,j_idx,orientationHeadingFromJToI)

            # self.p.matching._matches['headingjtoi'] = orientationHeadingFromJToI



