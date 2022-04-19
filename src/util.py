import cv2
import random as rng
import numpy as np

def resize_with_aspectratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def random_color():
    return (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))

def order_points(pts):
    """
    source: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/ 
    """

	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "int")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect



def rectify_contour(src_points,img,display = False):
    (old_h,  old_w, _) = img.shape  

    min_x = min(src_points[0][0],src_points[3][0])
    max_x = max(src_points[1][0],src_points[2][0])

    min_y = min(src_points[0][1],src_points[1][1])
    max_y = max(src_points[2][1],src_points[3][1])

    src  = np.array(src_points,np.float32) # src_points are converted into a numpy array and floating points
    dst = np.array([[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]],np.float32) # dst array is setup with the previously defined points, this array is also converted into a numpy array and floats

    transform_mat = cv2.getPerspectiveTransform(src,dst) 
    affine_image = cv2.warpPerspective(img,M=transform_mat,dsize=(old_w,old_h))

    crop_img = affine_image[min_y:max_y,min_x:max_x] # crop image


    # Draw contours
    if display:
        # Show the tranformed image
        cv2.imshow('Rectified image',affine_image)
        cv2.imshow('Cropped image',crop_img)
        cv2.waitKey(0)
    
    return affine_image,crop_img