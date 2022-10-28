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


class ClearAllDataAction(QAction):
    """Keypoint image i&j shown if they don't already have files with their keypoints"""

    def __init__(self, parent):
        super(ClearAllDataAction, self).__init__('Clear All Data', parent)
        self.p = parent

        self.setIcon(QIcon(icon_path('eye')))
        self.setShortcut('Ctrl+D')
        self.triggered.connect(self.clearalldata)
        self.setEnabled(True)

        self.i_kpids = Counter()
        self.j_kpids = Counter()

    def clearalldata(self):

        # self.p.matching._view_i = {}
        # self.p.matching._view_j = {}

        # self.p.matching._groups = {}
        # self.p.matching._matchcounter = Counter()
        # self.p.matching._matches = {}

        # self.p.matching.load_groups()
        # self.p.matching.load_viewlist()
        # self.p.matching.initialize_matchcounter()

        # self.p.matching.set_update()
        # self.p.matching.set_dirty()


        # get the indexes (which are also the ids) of points which have matches
        for matches in tqdm(self.p.matching._matches.values(),desc="walking match id values"):
            # print(matches)
            i_idx,j_idx  = matches
            self.i_kpids[i_idx] += 1

        for idx in tqdm(self.i_kpids,desc="cleaning i matches"):
            self.p.matching.remove_match_in_view_i(idx)

        for idx in tqdm(list(self.i_kpids.keys()),desc="cleaning i keypoints"):
            try:
                self.p.matching.remove_keypoint_in_view_i(idx)
            except KeyError:
                pass

        for matches in tqdm(self.p.matching._matches.values(),desc="walking match id values"):
            # print(matches)
            i_idx,j_idx  = matches
            self.j_kpids[j_idx] += 1

        for idx in tqdm(self.i_kpids,desc="cleaning j matches"):
            self.p.matching.remove_match_in_view_j(idx)


        for idx in tqdm(list(self.j_kpids.keys()),desc="cleaning j keypoints"):
            try:
                self.p.matching.remove_keypoint_in_view_j(idx)
            except KeyError:
                pass

        while len(self.p.matching.get_keypoints_i()) > 0:
            [self.p.matching.remove_keypoint_in_view_i(kp['id']) for kp in self.p.matching.get_keypoints_i()]

        while len(self.p.matching.get_keypoints_j()) > 0:
            [self.p.matching.remove_keypoint_in_view_j(kp['id']) for kp in self.p.matching.get_keypoints_j()]

        print(self.p.matching.get_keypoints_i())
        print(self.p.matching.get_keypoints_j())



            
        
        # self.p.canvas.update()
                
        # for idx in tqdm(self.j_kpids,desc="cleaning unmatched keypoints in j"):
        #     self.p.matching.remove_keypoint_in_view_j(idx)

        # self.p.canvas.update()
