import os.path as osp
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction
from utils.filesystem import icon_path
from pathlib import Path
import cv2 as cv
import copyreg
import pickle
from tqdm import tqdm

def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


class AutoKeypointAction(QAction):
    """Keypoint image i&j shown if they don't already have files with their keypoints"""

    def __init__(self, parent):
        super(AutoKeypointAction, self).__init__('Auto Keypoint', parent)
        self.p = parent

        self.setIcon(QIcon(icon_path('eye')))
        self.setShortcut('Ctrl+K')
        self.triggered.connect(self.automaticKeypoint)
        self.setEnabled(True)

        self.ikp = None
        self.ides = None
        self.jkp = None
        self.jdes = None


    def gen_kp_des_paths(self, basePath):
        kp_pkl_path = basePath.with_stem(basePath.stem + "_kp").with_suffix(".pkl")
        des_pkl_path = kp_pkl_path.with_stem(basePath.stem + "_des")
        return kp_pkl_path, des_pkl_path


    def save_keypoints_and_descriptors(self, kpdata, desdata, basePath,overwrite=False):
        kp_pkl_path, des_pkl_path = self.gen_kp_des_paths(basePath)

        with kp_pkl_path.open("wb") as kpfile:
            pickle.dump(kpdata,kpfile)

        with des_pkl_path.open("wb") as desfile:
            pickle.dump(desdata,desfile)


    def automaticKeypoint(self, _value=False):
        view_id_i = self.p.matching.get_view_id_i()
        view_id_j = self.p.matching.get_view_id_j()

        # get paths to files to write
        i_path = Path(self.p.imageDir) / self.p.matching.get_filename(self.p.matching.get_view_id_i())
        j_path = Path(self.p.imageDir) / self.p.matching.get_filename(self.p.matching.get_view_id_j())

        # create keypoint detector
        sift = cv.SIFT_create()

        # detect and compute keypoints
        # TODO: if file already exists, check for overwrite?
        self.ikp,self.ides = sift.detectAndCompute(self.p.canvas.img_i,None)
        self.save_keypoints_and_descriptors(self.ikp,self.ides,i_path)

        self.jkp,self.jdes = sift.detectAndCompute(self.p.canvas.img_j,None)
        self.save_keypoints_and_descriptors(self.jkp,self.jdes,j_path)

        for pt in tqdm(self.ikp,desc="Adding top image keypoint dots..."):
            ix, iy = pt.pt
            self.p.matching.append_keypoint_in_view_i(ix,iy)
        self.p.canvas.update()

        for pt in tqdm(self.jkp,desc="Adding bottom image keypoint dots..."):
            jx, jy = pt.pt
            # jy += self.p.canvas.img_i_h
            self.p.matching.append_keypoint_in_view_j(jx,jy)
        self.p.canvas.update()


        # kp2,des2 = sift.detectAndCompute(self.p.canvas.img_j)



        # view_id_j = self.p.matching.get_next_view(view_id_j)
        # self.p.changePair(view_id_i, view_id_j)
