#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import codecs
import distutils.spawn
import os
import os.path as osp
import platform
import re
import sys
import subprocess
import json
import numpy as np
import cv2
from functools import partial
from collections import defaultdict
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from libs.combobox import ComboBox
# from libs.resources import *
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.keypoints import Keypoints
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.newFileDialog import NewFileDialog
from libs.colorDialog import ColorDialog
from libs.toolBar import ToolBar
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem

__appname__ = 'PointMatcher'


class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # annotation file
        self.matching = None

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # Load string bundle for i18n
        self.stringBundle = StringBundle.getBundle()
        getStr = lambda strId: self.stringBundle.getString(strId)

        self.imageDir = None
        self.savePath = None
        self.img_i_w, self.img_i_h = None, None
        self.img_j_w, self.img_j_h = None, None

        # Whether we need to save or not.
        self.dirty = False

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Create and add combobox for showing unique labels in group
        self.comboBox = ComboBox(self)
        listLayout.addWidget(self.comboBox)

        self.pairListWidget = QListWidget()
        self.pairListWidget.itemDoubleClicked.connect(self.pairitemDoubleClicked)
        pairlistLayout = QVBoxLayout()
        pairlistLayout.setContentsMargins(0, 0, 0, 0)
        pairlistLayout.addWidget(self.pairListWidget)
        pairListContainer = QWidget()
        pairListContainer.setLayout(pairlistLayout)
        self.pairdock = QDockWidget(getStr('pairList'), self)
        self.pairdock.setObjectName(getStr('pairs'))
        self.pairdock.setWidget(pairListContainer)

        self.fileListWidgetI = QListWidget()
        self.fileListWidgetI.itemDoubleClicked.connect(self.fileitemDoubleClickedI)
        filelistLayoutI = QVBoxLayout()
        filelistLayoutI.setContentsMargins(0, 0, 0, 0)
        filelistLayoutI.addWidget(self.fileListWidgetI)
        fileListContainerI = QWidget()
        fileListContainerI.setLayout(filelistLayoutI)
        self.filedockI = QDockWidget(getStr('fileListI'), self)
        self.filedockI.setObjectName(getStr('files'))
        self.filedockI.setWidget(fileListContainerI)

        self.fileListWidgetJ = QListWidget()
        self.fileListWidgetJ.itemDoubleClicked.connect(self.fileitemDoubleClickedJ)
        filelistLayoutJ = QVBoxLayout()
        filelistLayoutJ.setContentsMargins(0, 0, 0, 0)
        filelistLayoutJ.addWidget(self.fileListWidgetJ)
        fileListContainerJ = QWidget()
        fileListContainerJ.setLayout(filelistLayoutJ)
        self.filedockJ = QDockWidget(getStr('fileListJ'), self)
        self.filedockJ.setObjectName(getStr('files'))
        self.filedockJ.setWidget(fileListContainerJ)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(settings.get(SETTING_DRAW_SQUARE, False))

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.pairdock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedockI)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedockJ)
        self.filedockI.setFeatures(QDockWidget.DockWidgetFloatable)
        self.filedockJ.setFeatures(QDockWidget.DockWidgetFloatable)

        # Actions
        action = partial(newAction, self)

        openDir = action(
            getStr('openDir'), self.openImageDir,
            'Ctrl+u', 'open', getStr('openDirDetail'))
        newFile = action(
            getStr('newFile'), self.newFile,
            'Ctrl+N', 'open', getStr('newFileDetail'))
        openFile = action(
            getStr('openFile'), self.openFile,
            'Ctrl+O', 'open', getStr('openFileDetail'))
        saveFile = action(
            getStr('saveFile'), self.saveFile,
            'Ctrl+S', 'save', getStr('saveFileDetail'), enabled=False)
        closeFile = action(
            getStr('closeFile'), self.closeFile,
            'Ctrl+W', 'close', getStr('closeFileDetail'))
        quitApp = action(
            getStr('quitApp'), self.close,
            'Ctrl+Q', 'quit', getStr('quitApp'))
        openNextPair = action(
            getStr('openNextPair'), self.openNextPair,
            'd', 'next', getStr('openNextPairDetail'))
        openPrevPair = action(
            getStr('openPrevPair'), self.openPrevPair,
            'a', 'prev', getStr('openPrevPairDetail'))

        editKeypointMode = action(
            getStr('editKeypoint'), self.setEditKeypointMode,
            'v', 'new', getStr('editKeypointDetail'), enabled=True)
        editMatchMode = action(
            getStr('editMatch'), self.setEditMatchMode,
            'e', 'edit', u'Move and edit Boxs', enabled=False)

        showInfo = action(
            getStr('info'), self.showInfoDialog,
            None, 'help', getStr('info'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            getStr('zoomin'), partial(self.addZoom, 10),
            'Ctrl++', 'zoom-in', getStr('zoominDetail'), enabled=False)
        zoomOut = action(
            getStr('zoomout'), partial(self.addZoom, -10),
            'Ctrl+-', 'zoom-out', getStr('zoomoutDetail'), enabled=False)
        zoomOrg = action(
            getStr('originalsize'), partial(self.setZoom, 100),
            'Ctrl+=', 'zoom', getStr('originalsizeDetail'), enabled=False)
        fitWindow = action(
            getStr('fitWin'), self.setFitWindow,
            'Ctrl+F', 'fit-window', getStr('fitWinDetail'),
            checkable=True, enabled=False)
        fitWidth = action(
            getStr('fitWidth'), self.setFitWidth,
            'Ctrl+Shift+F', 'fit-width', getStr('fitWidthDetail'),
            checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        shapeLineColor = action(getStr('shapeLineColor'), self.chshapeLineColor,
                                icon='color_line', tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), self.chshapeFillColor,
                                icon='color', tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        self.newFileDialog = NewFileDialog(self)

        # Store actions for further handling.
        self.actions = struct(
            openDir=openDir,
            newFile=newFile,
            openFile=openFile,
            saveFile=saveFile,
            closeFile=closeFile,
            editKeypointMode=editKeypointMode,
            editMatchMode=editMatchMode,
            shapeLineColor=shapeLineColor,
            shapeFillColor=shapeFillColor,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            zoomActions=zoomActions,
            fileMenuActions=(openDir, newFile, openFile, saveFile, closeFile, quitApp),
            editMenu=tuple(),
            onLoadActive=tuple())

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'))

        self.autoSaving = QAction(getStr('autoSaveMode'), self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(settings.get(SETTING_AUTO_SAVE, False))

        addActions(
            self.menus.file,
            (openDir, newFile, openFile, saveFile, closeFile, quitApp))
        addActions(
            self.menus.edit,
            (editKeypointMode, editMatchMode))
        addActions(
            self.menus.help,
            (showInfo,))
        addActions(
            self.menus.view,
            (self.autoSaving, None,
             zoomIn, zoomOut, zoomOrg, None,
             fitWindow, fitWidth))

        self.tools = self.toolbar('Tools')
        addActions(
            self.tools,
            (openDir, openFile, openNextPair, openPrevPair,
             zoomIn, zoom, zoomOut, fitWindow, fitWidth))

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.recentFiles = []
        self.maxRecent = 7
        self.zoom_level = 100
        self.fit_window = False

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

    def noShapes(self):
        return not self.itemsToShapes

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()
        self.comboBox.cb.clear()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def showInfoDialog(self):
        from libs.__init__ import __version__
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def setEditKeypointMode(self):
        self.actions.editKeypointMode.setEnabled(False)
        self.actions.editMatchMode.setEnabled(True)
        self.canvas.setEditKeypointMode()

    def setEditMatchMode(self):
        self.actions.editKeypointMode.setEnabled(True)
        self.actions.editMatchMode.setEnabled(False)
        self.canvas.setEditMatchMode()

    def pairitemDoubleClicked(self, item=None):
        idx = self.pairListWidget.currentIndex().row()
        if idx < len(self.matching['valid_pairs']):
            id_view_i = self.matching['valid_pairs'][idx][0]
            id_view_j = self.matching['valid_pairs'][idx][1]
            self.changePair(id_view_i, id_view_j)

    def fileitemDoubleClickedI(self, item=None):
        id_view_i = self.matching['views'][self.fileListWidgetI.currentIndex().row()]['id_view']
        id_view_j = self.matching['views'][self.fileListWidgetJ.currentIndex().row()]['id_view']
        self.changePair(id_view_i, id_view_j)

    def fileitemDoubleClickedJ(self, item=None):
        id_view_i = self.matching['views'][self.fileListWidgetI.currentIndex().row()]['id_view']
        id_view_j = self.matching['views'][self.fileListWidgetJ.currentIndex().row()]['id_view']
        self.changePair(id_view_i, id_view_j)

    def changePair(self, id_view_i, id_view_j):
        if len(self.matching['valid_pairs']) < self.pairListWidget.count():
            self.pairListWidget.takeItem(self.pairListWidget.count()-1)
        m = [p == [id_view_i, id_view_j] for p in self.matching['valid_pairs']]
        if any(m):
            idx = m.index(True)
            self.pairListWidget.setCurrentRow(idx)
        else:
            self.pairListWidget.addItem('None ({}, {})'.format(id_view_i, id_view_j))
            self.pairListWidget.setCurrentRow(self.pairListWidget.count()-1)
        idx_view_i = self.get_idx_view(id_view_i)
        idx_view_j = self.get_idx_view(id_view_j)
        self.fileListWidgetI.setCurrentRow(idx_view_i)
        self.fileListWidgetJ.setCurrentRow(idx_view_j)
        img_i = cv2.imread(osp.join(self.imageDir, self.matching['views'][idx_view_i]['filename']))
        img_j = cv2.imread(osp.join(self.imageDir, self.matching['views'][idx_view_j]['filename']))
        self.img_i_h, self.img_i_w, _ = img_i.shape
        self.img_j_h, self.img_j_w, _ = img_j.shape
        # set image
        img_h = self.img_i_h + self.img_j_h
        img_w = max(self.img_i_w, self.img_j_w)
        img = np.zeros(shape=(img_h, img_w, 3), dtype=np.uint8)
        img[:self.img_i_h, :self.img_i_w, :] = img_i
        img[self.img_i_h:, :self.img_j_w, :] = img_j
        qimg = QImage(img.flatten(), img_w, img_h, QImage.Format_BGR888)
        self.image = qimg
        self.canvas.setPixmap(QPixmap.fromImage(qimg))
        # set keypoints
        keypoints_i = Keypoints()
        for p in self.matching['views'][idx_view_i]['keypoints']:
            keypoints_i.append(p[0], p[1])
        keypoints_j = Keypoints()
        for p in self.matching['views'][idx_view_j]['keypoints']:
            keypoints_j.append(p[0], p[1])
        self.canvas.setKeypoints_i(keypoints_i)
        self.canvas.setKeypoints_j(keypoints_j)
        # set offset
        self.canvas.setOffset(self.img_i_h)
        self.canvas.repaint()

    def get_idx_view(self, id_view):
        return [v['id_view'] == id_view for v in self.matching['views']].index(True)

    def updateComboBox(self):
        # Get the unique labels and add them to the Combobox.
        itemsTextList = [str(self.labelList.item(i).text()) for i in range(self.labelList.count())]

        uniqueTextList = list(set(itemsTextList))
        # Add a null row for showing all the labels
        uniqueTextList.append("")
        uniqueTextList.sort()

        self.comboBox.update_items(uniqueTextList)

    def comboSelectionChanged(self, index):
        text = self.comboBox.cb.itemText(index)
        for i in range(self.labelList.count()):
            if text == "":
                self.labelList.item(i).setCheckState(2)
            elif text != self.labelList.item(i).text():
                self.labelList.item(i).setCheckState(0)
            else:
                self.labelList.item(i).setCheckState(2)

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            # Sync single class mode from PR#106
            if self.singleClassMode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.lastLabel = text
        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            generate_color = generateColorByText(text)
            shape = self.canvas.setLastLabel(text, generate_color, generate_color)
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def loadFile(self, filePath):
        with open(filePath, 'r') as f:
            self.matching = json.load(f)
        self.pairListWidget.clear()
        self.fileListWidgetI.clear()
        self.fileListWidgetJ.clear()
        for valid_pair in self.matching['valid_pairs']:
            self.pairListWidget.addItem('({}, {})'.format(valid_pair[0], valid_pair[1]))
        for view in self.matching['views']:
            self.fileListWidgetI.addItem('{} | {}'.format(view['id_view'], view['filename']))
            self.fileListWidgetJ.addItem('{} | {}'.format(view['id_view'], view['filename']))
        id_view_i = self.matching['valid_pairs'][0][0]
        id_view_j = self.matching['valid_pairs'][0][1]
        self.changePair(id_view_i, id_view_j)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)

        # Fix bug: An  index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        unicodeFilePath = os.path.abspath(unicodeFilePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                fileWidgetItem.setSelected(True)
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
                self.canvas.verified = self.labelFile.verified
            else:
                # Load image:
                # read data first and store for saving into label file.
                self.imageData = read(unicodeFilePath, None)
                self.labelFile = None
                self.canvas.verified = False

            if isinstance(self.imageData, QImage):
                image = self.imageData
            else:
                image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count()-1))
                self.labelList.item(self.labelList.count()-1).setSelected(True)

            self.canvas.setFocus(True)
            return True
        return False

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ''

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
        settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
        settings[SETTING_LABEL_FILE_FORMAT] = self.labelFileFormat
        settings.save()

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    def openImageDir(self, _value=False):
        if self.imageDir and os.path.exists(self.imageDir):
            defaultDir = self.imageDir
        else:
            defaultDir = '.'
        self.imageDir = ustr(
            QFileDialog.getExistingDirectory(
                self, '%s - Open Directory' % __appname__, defaultDir,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))

    def newFile(self, _value=False):
        if not self.mayContinue():
            return
        filename = self.newFileDialog.popUp()
        print(filename)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(ustr(self.filePath)) if self.filePath else '.'
        filters = 'matching file (*.json)'
        filename = QFileDialog.getOpenFileName(self, '%s - Choose matching file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        if self.savePath:
            self._saveFile(self.savePath)

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return

    def openNextPair(self, _value=False):
        if self.autoSaving.isChecked():
            if self.dirty is True:
                if self.savePath:
                    self._saveFile(self.savePath)
        if not self.mayContinue():
            return

    def openPrevPair(self, _value=False):
        if self.autoSaving.isChecked():
            if self.dirty is True:
                if self.savePath:
                    self._saveFile(self.savePath)
        if not self.mayContinue():
            return

    def _saveFile(self, path):
        with open(path, 'w') as f:
            json.dump(self.matching, f)

    def mayContinue(self):
        if not self.dirty:
            return True
        else:
            discardChanges = self.discardChangesDialog()
            if discardChanges == QMessageBox.No:
                return True
            elif discardChanges == QMessageBox.Yes:
                self.saveFile()
                return True
            else:
                return False

    def discardChangesDialog(self):
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        msg = u'You have unsaved changes, would you like to save them and proceed?'
        return QMessageBox.warning(self, u'Attention', msg, yes | no | cancel)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()


def read(filename, default=None):
    try:
        reader = QImageReader(filename)
        reader.setAutoTransform(True)
        return reader.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    argparser = argparse.ArgumentParser()
    argparser.add_argument("image_dir", nargs="?")
    argparser.add_argument("predefined_classes_file",
                           default=os.path.join(os.path.dirname(__file__), "data", "predefined_classes.txt"),
                           nargs="?")
    argparser.add_argument("save_dir", nargs="?")
    args = argparser.parse_args(argv[1:])
    # Usage : labelImg.py image predefClassFile saveDir
    win = MainWindow(args.image_dir,
                     args.predefined_classes_file,
                     args.save_dir)
    win.show()
    return app, win


def main():
    '''construct main app and run it'''
    app, _win = get_main_app(sys.argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
