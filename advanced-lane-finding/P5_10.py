import numpy as np
import cv2
import glob
import time
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from skimage.feature import hog
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn.cross_validation import train_test_split # scikit-learn <= 0.17
#from sklearn.model_selection import train_test_split # scikit-learn >= 0.18

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.feature_selection import RFECV, RFE
from scipy.ndimage.measurements import label

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

import sys
print(sys.version)

def get_data():
    if (1): car_file, nocar_file = \
        glob.glob('./vehicles/*/*.png'), glob.glob('./non-vehicles/*/*.png')
    else: car_file, nocar_file = \
        glob.glob('./vehicles_smallset/*/*.jpeg'), glob.glob('./non-vehicles_smallset/*/*.jpeg')  
    
    cars, notcars = [], []
    for image in car_file: cars.append(image)
    for image in nocar_file: notcars.append(image)

    car, notcar = mpimg.imread(cars[8040]), mpimg.imread(notcars[3552]) # loads rgb colorspace
    
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar)
    plt.title('Example NotCar Image')
    plt.show()

    print('Number of Cars:',len(cars))
    print('Number of NotCars:',len(notcars),'\n')
    print('Shape:',car.shape)
    print('Data Type:',car.dtype)
    print('Average Value',np.mean(car)) # training with png, need to scale pipeline jpg from 0,255 to 0,1

    return cars, notcars, car, notcar

if (1): cars, notcars, car, notcar = get_data()

def draw_boxes(img, bboxes, color=(255,0,0), thick=6):
    draw_img = np.copy(img)
    for box in bboxes: cv2.rectangle(draw_img, box[0], box[1], color, thick)
    return draw_img
    
def color_hist(img, nbins=32):    
    # calculate color histogram with 32 bins
    channel1_hist = np.histogram(img[:,:,0], bins=nbins) #, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins) #, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins) #, range=bins_range)
    hist_feat32 = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
    # calculate color histogram with 16 bins
    ch1_16, ch2_16, ch3_16 = [], [], []
    for i in range(0,(len(channel1_hist[0])-1),2):
        ch1_16.append(np.sum(channel1_hist[0][i:(i+2)]))
        ch2_16.append(np.sum(channel2_hist[0][i:(i+2)]))
        ch3_16.append(np.sum(channel3_hist[0][i:(i+2)]))
    hist_feat16 = np.concatenate((ch1_16,ch2_16,ch3_16))

    # calculate color histogram with 8 bins
    ch1_8, ch2_8, ch3_8 = [], [], []
    for i in range(0,(len(channel1_hist[0])-1),4):
        ch1_8.append(np.sum(channel1_hist[0][i:(i+4)]))
        ch2_8.append(np.sum(channel2_hist[0][i:(i+4)]))
        ch3_8.append(np.sum(channel3_hist[0][i:(i+4)]))
    hist_feat8 = np.concatenate((ch1_8,ch2_8,ch3_8))
    
    # calculate color histogram with 4 bins
    ch1_4, ch2_4, ch3_4 = [], [], []
    for i in range(0,(len(channel1_hist[0])-1),8):
        ch1_4.append(np.sum(channel1_hist[0][i:(i+8)]))
        ch2_4.append(np.sum(channel2_hist[0][i:(i+8)]))
        ch3_4.append(np.sum(channel3_hist[0][i:(i+8)]))
    hist_feat4 = np.concatenate((ch1_4,ch2_4,ch3_4))
    
    # combine features into single feature vector
    hist_features = np.concatenate((hist_feat32,hist_feat16,hist_feat8,hist_feat4))
    
    # return color histogram features
    return hist_features
    
def bin_spatial(img, spatial=32):
    spatial_features = cv2.resize(img, (spatial,spatial)).ravel()
    return spatial_features

orient = 8 # 12
pix_per_cell = 16 # 16
cell_per_block = 2 # 1
transform_sqrt = True # True

def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=3, transform_sqrt=False, feature_vector=True):
    visualise = False
    hog_features = hog(img, orient, (pix_per_cell,pix_per_cell), (cell_per_block,cell_per_block),
                                  visualise, transform_sqrt, feature_vector)
    return hog_features

def extract_features(imgs):
    features = []
    for file in imgs:
        img= mpimg.imread(file) # loading png file, scaled 0 to 1
        image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
       
        hist_feat1 = color_hist(image, nbins=32) 
        hist_features = hist_feat1

        spat_feat1 = bin_spatial(image, spatial=4)
        spat_feat2 = bin_spatial(image, spatial=8)
        spat_feat3 = bin_spatial(image, spatial=16)
        spat_features = np.hstack((spat_feat1, spat_feat2, spat_feat3)) 
        #spat_features = spat_feat3

        ## orient, pix_per_cell, cell_per_block, transform_sqrt = 12, 16, 1, True # defined in above section
        #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #hog_feat1 = get_hog_features(gray, orient, pix_per_cell, cell_per_block, transform_sqrt)
        hog_feat2 = get_hog_features(image[:,:,0], orient, pix_per_cell, cell_per_block, transform_sqrt)
        hog_feat3 = get_hog_features(image[:,:,1], orient, pix_per_cell, cell_per_block, transform_sqrt)
        hog_feat4 = get_hog_features(image[:,:,2], orient, pix_per_cell, cell_per_block, transform_sqrt)
        hog_features = np.hstack((hog_feat2, hog_feat3, hog_feat4))        

        features.append(np.concatenate((hist_features, spat_features, hog_features)))
        #features.append(hog_features)
    return features

def feature_extraction(scaler = 'standard'):
    print ('Feature Extraction...')
    t=time.time()
    
    car_features = extract_features(cars)
    notcar_features = extract_features(notcars)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    if scaler == 'robust': X_scaler = RobustScaler().fit(X)
    else: X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    t2 = time.time()
    print (round(t2-t, 2), 'Seconds for Feature Extraction...')
    print ('Done!\n')    
    print ('X Shape:',X.shape)
    print ("scaled_X Shape:",scaled_X.shape)
    return X_scaler, X, y, scaled_X

if (1): X_scaler, X, y, scaled_X = feature_extraction(scaler = 'standard')

X = scaled_X

def feature_selection(X, y, step, n_select):  
    print ('Feature Selection...')
    t=time.time()
    clf = LinearSVC(C=0.001)
    cv = StratifiedShuffleSplit(y=y, n_iter=1, test_size=0.2, random_state=0)   
    rfecv = RFECV(clf,step,cv,verbose=2)
    rfecv.fit(X,y)
    #rfe = RFE(clf,n_select,step,verbose=1)
    #rfe.fit(X,y)
    
    plt.figure()
    plt.xlabel('Number of features tested x 100')
    plt.ylabel('CV Accuracy')
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.ylim([0.95,1.00])
    plt.title('Recursive Feature Elimination')
    plt.show()

    print ('\nMax CV Score: ',round(np.max(rfecv.grid_scores_), 3))
    print ('\nX Shape:',X.shape)
    X_new = rfecv.transform(X)
    print ('X_New Shape:',X_new.shape,'\n')
    #plt.savefig('P5-RFECV.png') ### save plots!!! ###
    #print (rfecv.grid_scores_)
    t2 = time.time()
    print (round(t2-t, 2), 'Seconds for Feature Selection...')
    print ('Done!')
    return rfecv, X_new

if (1): rfecv, X_new = feature_selection(X, y, step=100, n_select=800)

sss = StratifiedShuffleSplit(y=y, n_iter=3, test_size=0.2, random_state=0)
for train_index, test_index in sss:
    X_train, X_test = X_new[train_index], X_new[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
print ('Training set:',X_train.shape)
print ('Testing set:',X_test.shape,'\n')

t=time.time()
svc = LinearSVC(C=0.001) # creating new classifier with optimized settings
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

print('\nTrain Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# training svc with full dataset for sliding window search
svc = svc.fit(X_new, y) # ==> not used in P5_7.ipynb

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb': return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    
def find_cars(img, ystart, ystop, scale, xstart, dec_func_thresh=1.00):
    # classifier trained with png [0:1] images, so we must normalize jpg [0:255] values
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    # only search where cars are likely to be found
    img_tosearch = img[ystart:ystop,:,:]
    # converting to best colorspace
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    
    # resizing sample image region
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, \
                                     (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    # separating channels for hog features
    ch1, ch2, ch3 = ctrans_tosearch[:,:,0], ctrans_tosearch[:,:,1], ctrans_tosearch[:,:,2]

    # define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1,orient,pix_per_cell,cell_per_block,transform_sqrt,feature_vector=False)
    hog2 = get_hog_features(ch2,orient,pix_per_cell,cell_per_block,transform_sqrt,feature_vector=False)
    hog3 = get_hog_features(ch3,orient,pix_per_cell,cell_per_block,transform_sqrt,feature_vector=False)
    
    box_list = []
    count = 0
    for xb in range(xstart, nxsteps+1):
        for yb in range(nysteps):
            count += 1
            ypos = yb*cells_per_step #start position of x
            xpos = xb*cells_per_step #start positon of y
            
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft,ytop = xpos*pix_per_cell, ypos*pix_per_cell
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            hist_features = color_hist(subimg, nbins=32)
            
            spat_feat1 = bin_spatial(subimg, spatial=4)
            spat_feat2 = bin_spatial(subimg, spatial=8)
            spat_feat3 = bin_spatial(subimg, spatial=16)
            spat_features = np.hstack((spat_feat1, spat_feat2, spat_feat3)) 

            combined_features = np.hstack((hist_features, spat_features, hog_features))
                      
            test_features = X_scaler.transform(combined_features.reshape(1, -1))             
            test_features = rfecv.transform(test_features) 
            test_prediction = svc.predict(test_features)
            test_dec_func = svc.decision_function(test_features)
            
            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),
                          (xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)            
            
            if test_prediction == 1 and test_dec_func >= dec_func_thresh:
                #print (xb,yb,scale,np.round(test_dec_func,3))
                box_list.append(((xbox_left, ytop_draw+ystart),\
                                 (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    for box in box_list: cv2.rectangle(draw_img, box[0], box[1], (255,0,0), 6)
    #print ('number of red / blue boxes:',len(box_list), count)
    return draw_img, box_list

dec_func_thresh = 1.00
xstart, ystart, ystop, scale = 30, 393, 585, 0.75 # 393, 656, 1.0

#img = mpimg.imread('test_image.jpg')
img = mpimg.imread('test_images/test23.jpg')
out_img, box_list = find_cars(img, ystart, ystop, scale, xstart, dec_func_thresh)

plt.imshow(out_img)
plt.show()

def add_heat(img, bbox_list):
    heatmap = np.zeros_like(img[:,:,0])
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1): 
        
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Define a bounding box based on min/max x and y
        min_x = np.min(nonzerox)
        min_y = np.min(nonzeroy)
        max_x = np.max(nonzerox)
        max_y = np.max(nonzeroy)
        if (max_x - min_x) > 30 or (max_y - max_y)> 30:
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 6)
    # Return the image
    return img

def draw_boxes(img, bboxes, color=(255,0,0), thick=6):
    draw_img = np.copy(img)
    for box in bboxes: cv2.rectangle(draw_img, box[0], box[1], color, thick)
    return draw_img

def initial_pipeline(img, ystart = 360, ystop = 660):        
    bboxes =[]

    _, box_list1 = find_cars(img, ystart = 393, ystop = 720, scale=2.5, xstart = 9)     
    _, box_list2 = find_cars(img, ystart = 369, ystop = 720, scale=2.5, xstart = 9) 
    _, box_list3 = find_cars(img, ystart = 393, ystop = 720, scale=2.0, xstart = 11) 
    _, box_list4 = find_cars(img, ystart = 373, ystop = 720, scale=2.0, xstart = 11) 
    _, box_list5 = find_cars(img, ystart = 393, ystop = 720, scale=1.3, xstart = 18) 
    _, box_list6 = find_cars(img, ystart = 381, ystop = 720, scale=1.3, xstart = 18) 
    #_, box_list7 = find_cars(img, ystart = 393, ystop = 585, scale=0.75, xstart = 30) 

    bboxes.extend(box_list1)
    bboxes.extend(box_list2)    
    bboxes.extend(box_list3)    
    bboxes.extend(box_list4)     
    bboxes.extend(box_list5)   
    bboxes.extend(box_list6)
    #bboxes.extend(box_list7)
       
    out_img = draw_boxes(img, bboxes, color=(0, 0, 255), thick=6)
    
    heat = add_heat(img, bboxes)    
    heat = apply_threshold(heat,2.1) ### 2.1
    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    #heatmap = (heatmap / np.max(heatmap)) * 255 # needed when used for output video
    out_img = draw_labeled_bboxes(out_img, labels)
    
    return out_img, heatmap, draw_img

heat_parms = np.zeros((720,1280))
alpha = 0.08

def final_pipeline_merge(img, result):
    global heat_parms

    out_img, heatmap, _ = initial_pipeline(img)
    heat_parms = alpha*heat_parms + (1-alpha)*heatmap
    labels = label(heatmap)
    
    draw_img = draw_labeled_bboxes(np.copy(result), labels)    
    
    return draw_img, out_img, heatmap, heat_parms













