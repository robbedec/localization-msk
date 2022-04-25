import cv2
import numpy as np

class FrameProcessor():
    def __init__(self, data_file, frame_shape):
        """
        - frame_shape: Tuple (W,H) of the camera 
        """
        with open(data_file, 'rb') as f:
            self.mtx = np.load(f)
            self.dist = np.load(f)
            self.rvecs = np.load(f)
            self.tvecs = np.load(f)
        
        refined_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, frame_shape, 1, frame_shape)
        self.refined_mtx = refined_mtx
        self.roi = roi
    
    def undistort(self, img):
        dst = cv2.undistort(src=img, cameraMatrix=self.mtx, distCoeffs=self.dist, newCameraMatrix=self.refined_mtx)

        # Crop image to ROI
        x, y, w, h = self.roi
        dst = dst[y:y+h, x:x+w]

        return dst

    @staticmethod
    def calibrate_camera(input_video, output, draw):
        """
        Follows the camera calibration guide from OpenCV
        https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        """

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*10,3), np.float32)
        objp[:,:2] = np.mgrid[0:10,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # Setup video stream
        cap = cv2.VideoCapture(input_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 2600)

        max_frame_n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            succes, img = cap.read()

            if not succes:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (10, 6))

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
                
                if draw:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (10, 6), corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                
                # Fastforward to new position
                current_frame_n = cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_frame_n = current_frame_n + (fps * 2)

                if new_frame_n > max_frame_n:
                    new_frame_n = max_frame_n
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_n)

        cv2.destroyAllWindows()

        # Returns the camera matrix, distortion coefficients, rotation and translation vectors 
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        with open(output, 'wb') as f:
            np.save(f, mtx)
            np.save(f, dist)
            np.save(f, rvecs)
            np.save(f, tvecs)

if __name__ == '__main__':
    # Code to create calibration files, should not execute if the two files exist in the data folder.
    #vid_path = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/gopro/calibration_M.mp4'
    #calib_file = '/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/gopro-M.npy'

    # vid_path = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/gopro/calibration_W.mp4'
    # calib_file = '/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/gopro-W.npy'

    # Create calibration file
    # FrameProcessor.calibrate_camera(vid_path, calib_file, True)


    # Sample usage: only use for videos taken with a gopro
    gopro_video = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/gopro/MSK_15.mp4'
    calib_file = '/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/gopro-M.npy'
    
    cap = cv2.VideoCapture(gopro_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frameproc = FrameProcessor(calib_file, (width, height))

    while True:
        succes, img = cap.read()

        undistorted_img = frameproc.undistort(img.copy())

        cv2.imshow('original',img)
        cv2.imshow('undistorted',undistorted_img)
        cv2.waitKey(1)