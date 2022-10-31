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
from math import atan2,degrees,floor
from pprint import pprint
import statistics


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

        # print(self.p.matching._matches)

        self.matchOrientations = {}

        for match_idx, match in self.p.matching._matches.items():
            # pprint(match_idx)
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
            # print(i_idx,j_idx,orientationHeadingFromJToI)
            # print(self.p.matching._matches[match_idx])

            self.matchOrientations[match_idx] = orientationHeadingFromJToI


        sortedMatches = {k: v for k, v in sorted(self.matchOrientations.items(), key=lambda item: item[1])}
        # print(sortedMatches)

        stdev = statistics.pstdev(self.matchOrientations.values())
        # print(stdev)


        vMax = max(self.matchOrientations.values())
        vMin = min(self.matchOrientations.values())
        vDelta = vMax-vMin
        vChunkCount = floor(vDelta/stdev)
        vRangeSize = vDelta/vChunkCount

        # print([vMax,vMin,vDelta,vChunkCount,vRangeSize])

        orientationGroupRanges = [vMin + (c*vRangeSize) for c in range(0,vChunkCount+1)]
        # print(orientationGroupRanges)

        checkSets = [x for x in zip(orientationGroupRanges,orientationGroupRanges[1:])]
        # print(checkSets)

        checks = {k:v for k,v in enumerate(checkSets)}
        # print(checks)
        groupedMatches = {k:set() for k in checks.keys()}
        # print(groupedMatches)

        for k,v in sortedMatches.items():
            # print([k,v])
            for key, check in checks.items():
                # print([key, check])
                if check[0] <= v <= check[1]:
                    groupedMatches[key].add(k)
                    # print(groupedMatches)
                    break # inner loop short circuit

        # print(groupedMatches)

        for k,v in groupedMatches.items():
            for e in v:
                self.p.matching._matchorigroup[e] = k

        self.p.canvas.update()


        # pprint(self.p.matching._matchorigroup)


        # exit()



