#!/bin/env python3
import sys
from os import path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui


def image_resize(image, width=200, height=200, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# Video stream from provided camera
class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)

# Face Detection, PCA and Recognization


class FaceDetectionWidget(QtWidgets.QWidget):

    def __init__(self, haar_cascade_filepath, parent=None):
        super().__init__(parent)
        self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)
        self.save_data = False
        self.match_data = False
        self.face_data = []

    # Detecting face using pretrained haarclassifiers
    def detect_faces(self, image: np.ndarray):
        # haarclassifiers work better in black and white
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)

        faces = self.classifier.detectMultiScale(gray_image,
                                                 scaleFactor=1.3,
                                                 minNeighbors=4,
                                                 flags=cv2.CASCADE_SCALE_IMAGE,
                                                 minSize=self._min_size)
        return faces

    # Option to match current face
    def match_data_slot(self):
        self.match_data = True

    # Option to save training data
    def save_data_slot(self):
        self.save_data = True

    # Switch to choose save or match face rectangle of current image
    def image_data_slot(self, image_data):
        faces = self.detect_faces(image_data)
        for (x, y, w, h) in faces:
            if (self.save_data or self.match_data):
                face = cv2.cvtColor(image_resize(
                    image_data[y:y+h, x:x+w]), cv2.COLOR_BGR2GRAY)
                if self.save_data:
                    self.face_data.append(face)
                if self.match_data:
                    self.train_faces(query=face)

            cv2.rectangle(image_data,
                          (x, y),
                          (x+w, y+h),
                          self._red,
                          self._width)
        if (self.save_data):
            self.save_data = False
        if (self.match_data):
            self.match_data = False

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    # Training application with face data to recognize from
    def train_faces(self, query: np.ndarray):
        faceshape = self.face_data[0].shape
        facematrix = np.asarray([self.face_data[i].flatten()
                                for i in range(len(self.face_data))])
        print(facematrix)
        print(facematrix.shape)
        pca = PCA(n_components=5).fit(facematrix)
        eigenfaces = pca.components_

        fig, axes = plt.subplots(
            2, 2, sharex=True, sharey=True, figsize=(8, 10))
        for i in range(4):
            axes[i % 2][i //
                        2].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
        print("Showing the eigenfaces")
        plt.show()

        # Generate weights as a KxN matrix where K is the number of eigenfaces
        #                                    and N the number of samples
        weights = eigenfaces @ (facematrix - pca.mean_).T
        print("Shape of the weight matrix:", weights.shape)
        print(weights)

        query = query.reshape(-1)
        query_weight = eigenfaces @ (query - pca.mean_).T
        print("Shape of the query matrix:", query_weight.shape)
        print(query_weight)
        query_weights = np.ndarray(weights.shape)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                query_weights[i][j] = query_weight[i]
        print("Shape of the queries matrix:", query_weights.shape)

        euclidean_distance = np.linalg.norm(weights - query_weights, axis=0)
        best_match = np.argmin(euclidean_distance)

        # Visualize
        fig, axes = plt.subplots(
            1, 2, sharex=True, sharey=True, figsize=(8, 6))
        axes[0].imshow(query.reshape(faceshape), cmap="gray")
        axes[0].set_title("Query")
        axes[1].imshow(facematrix[best_match].reshape(faceshape), cmap="gray")
        axes[1].set_title("Best match")
        plt.show()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    # Presents the GUI of the app
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

# Main window of the application with options


class MainWidget(QtWidgets.QWidget):
    def __init__(self, haarcascade_filepath, parent=None):
        super().__init__(parent)
        fp = haarcascade_filepath
        self.face_detection_widget = FaceDetectionWidget(fp)

        self.record_video = RecordVideo()

        image_data_slot = self.face_detection_widget.image_data_slot
        save_data_slot = self.face_detection_widget.save_data_slot
        match_data_slot = self.face_detection_widget.match_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.face_detection_widget)
        self.run_button = QtWidgets.QPushButton('Start')
        layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.record_video.start_recording)

        self.train_button = QtWidgets.QPushButton('Train')
        layout.addWidget(self.train_button)
        self.train_button.clicked.connect(save_data_slot)

        self.match_button = QtWidgets.QPushButton('Match')
        layout.addWidget(self.match_button)
        self.match_button.clicked.connect(match_data_slot)

        self.setLayout(layout)

# Point of execution for the applicatiom


def main(haar_cascade_filepath):
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(haar_cascade_filepath)
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    cascade_filepath = path.join(script_dir,
                                 'haarcascade_frontalface_default.xml')

    cascade_filepath = path.abspath(cascade_filepath)
    main(cascade_filepath)
