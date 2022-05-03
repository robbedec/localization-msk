import cv2
import numpy as np
import sys
import pywt

from util import resize_with_aspectratio

class FrameProcessor():
    def __init__(self, data_file, frame_shape):
        """
        Load the camera parameters from a given file.

        - data_file: path to the file that contains the python objects of
                     the camera matrix etc.
        - frame_shape: Tuple (W,H) of the camera frames.
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
        # The undistort may cause invalid pixels (closer to the edges).
        x, y, w, h = self.roi
        dst = dst[y:y+h, x:x+w]

        return dst

    @staticmethod
    def calibrate_camera(input_video, output, draw, manual_add=True):
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
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

                if draw:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (10, 6), corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                
                if input() == 'y' or not manual_add:
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                
                
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
    
    @staticmethod
    def sharpness_metric(img, print_metric=False):
        # https://en.wikipedia.org/wiki/Acutance

        # Haar transform is much faster on smaller images
        img_small = resize_with_aspectratio(img, width=400)

        # https://github.com/pedrofrodenas/blur-Detection-Haar-Wavelet
        # https://www.cs.cmu.edu/~htong/pdf/ICME04_tong.pdf
        per, blur_ext = blur_detect(img_small, 35)

        if print_metric:
            print(per, per < 0.001)

        return per < 0.001

def blur_detect(img, threshold):
    # Convert image to grayscale
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    M, N = Y.shape
    
    # Crop input image to be 3 divisible by 2
    Y = Y[0:int(M/16)*16, 0:int(N/16)*16]
    
    # Step 1, compute Haar wavelet of input image
    LL1,(LH1,HL1,HH1)= pywt.dwt2(Y, 'haar')
    # Another application of 2D haar to LL1
    LL2,(LH2,HL2,HH2)= pywt.dwt2(LL1, 'haar') 
    # Another application of 2D haar to LL2
    LL3,(LH3,HL3,HH3)= pywt.dwt2(LL2, 'haar')
    
    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2)+np.power(HL1, 2)+np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2)+np.power(HL2, 2)+np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2)+np.power(HL3, 2)+np.power(HH3, 2))
    
    M1, N1 = E1.shape

    # Sliding window size level 1
    sizeM1 = 8
    sizeN1 = 8
    
    # Sliding windows size level 2
    sizeM2 = int(sizeM1/2)
    sizeN2 = int(sizeN1/2)
    
    # Sliding windows size level 3
    sizeM3 = int(sizeM2/2)
    sizeN3 = int(sizeN2/2)
    
    # Number of edge maps, related to sliding windows size
    N_iter = int((M1/sizeM1)*(N1/sizeN1))
    
    Emax1 = np.zeros((N_iter))
    Emax2 = np.zeros((N_iter))
    Emax3 = np.zeros((N_iter))
    
    
    count = 0
    
    # Sliding windows index of level 1
    x1 = 0
    y1 = 0
    # Sliding windows index of level 2
    x2 = 0
    y2 = 0
    # Sliding windows index of level 3
    x3 = 0
    y3 = 0
    
    # Sliding windows limit on horizontal dimension
    Y_limit = N1-sizeN1
    
    while count < N_iter:
        # Get the maximum value of slicing windows over edge maps 
        # in each level
        Emax1[count] = np.max(E1[x1:x1+sizeM1,y1:y1+sizeN1])
        Emax2[count] = np.max(E2[x2:x2+sizeM2,y2:y2+sizeN2])
        Emax3[count] = np.max(E3[x3:x3+sizeM3,y3:y3+sizeN3])
        
        # if sliding windows ends horizontal direction
        # move along vertical direction and resets horizontal
        # direction
        if y1 == Y_limit:
            x1 = x1 + sizeM1
            y1 = 0
            
            x2 = x2 + sizeM2
            y2 = 0
            
            x3 = x3 + sizeM3
            y3 = 0
            
            count += 1
        
        # windows moves along horizontal dimension
        else:
                
            y1 = y1 + sizeN1
            y2 = y2 + sizeN2
            y3 = y3 + sizeN3
            count += 1
    
    # Step 3
    EdgePoint1 = Emax1 > threshold;
    EdgePoint2 = Emax2 > threshold;
    EdgePoint3 = Emax3 > threshold;
    
    # Rule 1 Edge Pojnts
    EdgePoint = EdgePoint1 + EdgePoint2 + EdgePoint3
    
    n_edges = EdgePoint.shape[0]
    
    # Rule 2 Dirak-Structure or Astep-Structure
    DAstructure = (Emax1[EdgePoint] > Emax2[EdgePoint]) * (Emax2[EdgePoint] > Emax3[EdgePoint]);
    
    # Rule 3 Roof-Structure or Gstep-Structure
    
    RGstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
            
                RGstructure[i] = 1
                
    # Rule 4 Roof-Structure
    
    RSstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
            
                RSstructure[i] = 1

    # Rule 5 Edge more likely to be in a blurred image 

    BlurC = np.zeros((n_edges));

    for i in range(n_edges):
    
        if RGstructure[i] == 1 or RSstructure[i] == 1:
        
            if Emax1[i] < threshold:
            
                BlurC[i] = 1                        
        
    # Step 6
    Per = np.sum(DAstructure)/np.sum(EdgePoint)
    
    # Step 7
    if (np.sum(RGstructure) + np.sum(RSstructure)) == 0:
        
        blur_extent = 100
    else:
        blur_extent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure))
    
    return Per, blur_extent

if __name__ == '__main__':
    # Code to create calibration files, should not execute if the two files exist in the data folder.
    # vid_path = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/gopro/calibration_M.mp4'
    # calib_file = '/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/gopro-M.npy'

    #vid_path = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/videos/gopro/calibration_W.mp4'
    #calib_file = '/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/gopro-W.npy'

    # Create calibration file
    # FrameProcessor.calibrate_camera(vid_path, calib_file, True)

    # TODO: navragen als de bolle lijnen volledig recht gemaakt kunnen worden door meer datapunten te geven
    # aan de calibratie.
    if len(sys.argv) != 3:
        print('Provide gopro file and calibration file')
        exit()

    gopro_video = sys.argv[1]
    calib_file = sys.argv[2]
    
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