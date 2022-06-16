import cv2
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize, binary_closing
from skimage import morphology
from scipy import ndimage
from skimage import io, filters


def Get_skeleton_image_and_remove_ruler(img_file):
        global skel_gray,I_binary,skel_gray_copy,skel_gray_copy_show,I_binary_copy,img,opening,length,height,width,I_draw,ruler_image,ruler_background,gray_image
        '''
        img = cv2.imread(img_file)
        (length,height,width)=np.shape(img)
        I = cv2.imread(img_file, 0)
        T,I = cv2.threshold(I,150,255,cv2.THRESH_BINARY)
        kernel=np.ones((3,3),np.uint8)
        opening=cv2.morphologyEx(I,cv2.MORPH_OPEN,kernel,iterations=2)
        sure_bg = cv2.dilate(opening,kernel,iterations=7)
        I = cv2.subtract(sure_bg,opening)
        '''
        img = cv2.imread(img_file)
        (length,height,width)=np.shape(img)
        I = cv2.imread(img_file, 0)
        gray_image=I.copy()
        T,I = cv2.threshold(I,150,255,cv2.THRESH_BINARY)
        kernel=np.ones((3,3),np.uint8)
        output = cv2.connectedComponentsWithStats(I,4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        x_max_area=0
        y_max_area=0
        max_area=0
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if(area>max_area):
                max_area=area
                j=i
                x_max_area=x
                y_max_area=y
                w_max_area=w
                h_max_area=h
        componentMask = (labels == j).astype("uint8") * 255
        ruler_image=componentMask.copy()
        sure_bg = cv2.dilate(componentMask,kernel,iterations=2)
        contours,hierarchy=cv2.findContours(sure_bg, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        max_area_error_point=0
        for i in range(1,len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area_error_point):
                max_area_error_point=area
        radius_max_area_point=(max_area_error_point/3.1412)**0.5
        cv2.drawContours(sure_bg, contours, -1,(255,255,255),int(radius_max_area_point*2)+1)
        ruler_background=sure_bg.copy()

        I_draw = cv2.subtract(I,sure_bg)
        I_draw=cv2.morphologyEx(I_draw,cv2.MORPH_OPEN,kernel,iterations=2)
        sure_bg = cv2.dilate(I_draw,kernel,iterations=7)
        I = cv2.subtract(sure_bg,I_draw)
        I_binary= cv2.merge((I,I,I))
        I_binary_copy=I_binary.copy()
        skel =skeletonize(I_binary)
        skel_gray = cv2.cvtColor(skel, cv2.COLOR_BGR2GRAY)
        skel_gray_copy = cv2.cvtColor(skel_gray, cv2.COLOR_GRAY2BGR)
        skel_gray_copy_show=skel_gray_copy.copy()

def Ruler_process():
    global average_distance_number
    kernel=np.ones((3,3),np.uint8)
    ruler_foreground=255-ruler_background
    ruler_image_remove_grain=cv2.subtract(gray_image,ruler_foreground)
    T,ruler_image_remove_grain_binary = cv2.threshold(ruler_image_remove_grain,150,255,cv2.THRESH_BINARY)
    ruler_image_remove_grain_binary=255-ruler_image_remove_grain_binary
    '''
    plt.imshow(ruler_image_remove_grain_binary)
    plt.show()
    '''
    eroded = cv2.erode(ruler_image_remove_grain_binary, kernel,3)
    sure_bg = cv2.dilate(eroded,kernel,iterations=2)
    output = cv2.connectedComponentsWithStats(sure_bg,4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    average_area=0
    center_number=[]
    number=[]
    for i in range(4, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        average_area=(average_area*(i-2)+area)/(i-1)
        
    for i in range(4, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if((area>0.5*average_area)and(area<2.5*average_area)):
            (cX, cY) = centroids[i]
            center_number.append((cX, cY))
            output = ruler_image_remove_grain.copy()
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
            componentMask = (labels == i).astype("uint8") * 255
            '''
            output=cv2.resize(output,(600,800))
            componentMask=cv2.resize(componentMask,(600,800))
            cv2.imshow("Output", output)
            cv2.imshow("Connected Component", componentMask)
            cv2.waitKey(0)
            '''
    distance_min=1000000
    average_distance_number=0
    for i in range(len(center_number)):
        for j in range(len(center_number)):
            (x1,y1)=center_number[i]
            (x2,y2)=center_number[j]
            distance= ((x1-x2)**2+(y1-y2)**2)**0.5
            if((distance>0)and(distance<distance_min)):
                distance_min=distance
    for i in range(len(center_number)):
        count=0
        for j in range(len(center_number)):
            (x1,y1)=center_number[i]
            (x2,y2)=center_number[j]
            distance= ((x1-x2)**2+(y1-y2)**2)**0.5
            if((distance>0)and(distance<1.5*distance_min)):
                count=count+1
                break
        if(count==0):
            number.append((x1,y1))
    distance_min=1000000
    for i in range(len(number)):
        for j in range(len(number)):
            (x1,y1)=number[i]
            (x2,y2)=number[j]
            distance= ((x1-x2)**2+(y1-y2)**2)**0.5
            if((distance>0)and(distance<distance_min)):
                distance_min=distance
    for i in range(len(number)):
        count=0
        for j in range(len(number)):
            (x1,y1)=number[i]
            (x2,y2)=number[j]
            distance= ((x1-x2)**2+(y1-y2)**2)**0.5
            if((distance>0)and(distance<1.5*distance_min)):
                count=count+1
                average_distance_number=(average_distance_number*(count-1)+distance)/(count)
                break
    Invalid_number=0.05
    average_distance_number=average_distance_number*(1+Invalid_number)
    print("one cm =",average_distance_number,"pixel")

def Kernel_to_find_endpoint():
        global kernel0,kernel1,kernel2,kernel3,kernel4,kernel5,kernel6,kernel7
        kernel0 = np.array((
        [0,1,0],
        [-1,1,-1],
        [-1,-1,-1],), dtype="int")
        kernel1 = np.array((
        [1,-1,-1],
        [-1,1,-1],
        [-1,-1,-1],), dtype="int")
        kernel2 = np.array((
        [0,-1,-1],
        [1,1,-1],
        [0,-1,-1],), dtype="int")
        kernel3 = np.array((
        [-1,-1,-1],
        [-1,1,-1],
        [1,-1,-1],), dtype="int")
        kernel4 = np.array((
        [-1,-1,-1],
        [-1,1,-1],
        [0,1,0],), dtype="int")
        kernel5 = np.array((
        [-1,-1,-1],
        [-1,1,-1],
        [-1,-1,1],), dtype="int")
        kernel6 = np.array((
        [-1,-1,0],
        [-1,1,1],
        [-1,-1,0],), dtype="int")
        kernel7 = np.array((
        [-1,-1,1],
        [-1,1,-1],
        [-1,-1,-1],), dtype="int")

def Kernel_to_find_branch_point():
        global kernel_0,kernel_1,kernel_2,kernel_3,kernel_4,kernel_5,kernel_6,kernel_7,kernel_8,kernel_9,kernel_10,kernel_11
        kernel_0 = np.array((
        [1,0,1],
        [0,1,0],
        [0,1,0],), dtype="int")
        kernel_1 = np.array((
        [0,1,0],
        [0,1,1],
        [1,0,0],), dtype="int")
        kernel_2 = np.array((
        [0,0,1],
        [1,1,0],
        [0,0,1],), dtype="int")
        kernel_3 = np.array((
        [1,0,0],
        [0,1,1],
        [0,1,0],), dtype="int")
        kernel_4 = np.array((
        [0,1,0],
        [0,1,0],
        [1,0,1],), dtype="int")
        kernel_5 = np.array((
        [0,0,1],
        [1,1,0],
        [0,1,0],), dtype="int")
        kernel_6 = np.array((
        [1,0,0],
        [0,1,1],
        [1,0,0],), dtype="int")
        kernel_7 = np.array((
        [0,1,0],
        [1,1,0],
        [0,0,1],), dtype="int")
        kernel_8 = np.array((
        [1,0,0],
        [0,1,0],
        [1,0,1],), dtype="int")
        kernel_9 = np.array((
        [1,0,1],
        [0,1,0],
        [1,0,0],), dtype="int")
        kernel_10 = np.array((
        [1,0,1],
        [0,1,0],
        [0,0,1],), dtype="int")
        kernel_11 = np.array((
        [0,0,1],
        [0,1,0],
        [1,0,1],), dtype="int")

def Find_branch_point(skel_gray):
        global skel_coords_branch,skel_gray_copy_show
        skel_coords_branch = []    
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_0)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_1)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_2)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_3)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_4)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_5)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_6)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_7)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_8)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_9)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_10)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel_11)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords_branch.append((r,c))
        for (r,c) in skel_coords_branch[:]:
                #cv2.circle(skel_gray_copy_show, (c,r), 5, (255, 160, 0))
                cv2.circle(skel_gray_copy, (c,r), 1, (255, 160, 0))
                #skel_gray_copy[r,c]=(255, 160, 0)

def Find_end_point_and_connect_to_branch_point(skel_gray,think):
        global skel_coords,skel_coords_branch_connect
        skel_coords_branch_connect = []
        skel_coords = []    
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel0)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                count=0
                done=True
                r_start,c_start=r,c
                while(done):
                        for i in range (-1,2):
                                r_check=r_start-1
                                c_check=c_start+i
                                if ((skel_gray_copy[r_check,c_check]==[255, 160, 0]).all()):
                                        r_end=r_check
                                        c_end=c_check
                                        done=False
                                        break
                                elif ((skel_gray_copy[r_check,c_check]==[150, 150, 150]).all()):
                                        count=0
                                        r_start=r_check
                                        c_start=c_check
                                        i_check=i
                                        done=True
                                        break
                                else:
                                        count+=1
                                if(count==3):
                                        r_end=r_start
                                        c_end=c_start
                                        done=False
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)
                                
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel1)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                j_check=0
                count=0
                done=True
                r_start,c_start=r,c
                while(done):
                        out=False
                        for i in range (-1,1):
                                for j in range (-1,1):
                                        r_check=r_start+i
                                        c_check=c_start+j
                                        if ((skel_gray_copy[r_check,c_check]==[255, 160, 0]).all()):
                                                r_end=r_check
                                                c_end=c_check
                                                done=False
                                                out=True
                                                break
                                        elif ((skel_gray_copy[r_check,c_check]==[150, 150, 150]).all()):
                                                count=0
                                                r_start=r_check
                                                c_start=c_check
                                                i_check=i
                                                j_check=j
                                                done=True
                                                out=True
                                                break
                                        else:
                                                count+=1
                                        if(count==3):
                                                r_end=r_start
                                                c_end=c_start
                                                done=False
                                                out=True
                                                break
                                if(out):
                                        break                                
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)
                
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel2)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                count=0
                done=True
                r_start,c_start=r,c
                while(done):
                        for i in range (-1,2):
                                r_check=r_start+i
                                c_check=c_start-1
                                if ((skel_gray_copy[r_check,c_check]==[255, 160, 0]).all()):
                                        r_end=r_check
                                        c_end=c_check
                                        done=False
                                        break
                                elif ((skel_gray_copy[r_check,c_check]==[150, 150, 150]).all()):
                                        count=0
                                        r_start=r_check
                                        c_start=c_check
                                        i_check=i
                                        done=True
                                        break
                                else:
                                        count+=1
                                if(count==3):
                                        r_end=r_start
                                        c_end=c_start
                                        done=False
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel3)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                j_check=0
                count=0
                done=True
                r_start,c_start=r,c
                while(done):
                        out=False
                        for i in range (-1,1):
                                for j in range (-1,1):
                                        r_check=r_start-i
                                        c_check=c_start+j
                                        if ((skel_gray_copy[r_check,c_check]==[255, 160, 0]).all()):
                                                r_end=r_check
                                                c_end=c_check
                                                out=True
                                                done=False
                                                break
                                        elif ((skel_gray_copy[r_check,c_check]==[150, 150, 150]).all()):
                                                done=True
                                                count=0
                                                r_start=r_check
                                                c_start=c_check
                                                i_check=i
                                                j_check=j
                                                out=True
                                                break
                                        else:
                                                count+=1
                                        if(count==3):
                                                r_end=r_start
                                                c_end=c_start
                                                done=False
                                                out=True
                                                break
                                if(out):
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel4)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                count=0
                done=True
                r_start,c_start=r,c
                while(done):
                        for i in range (-1,2):
                                r_check=r_start+1
                                c_check=c_start+i
                                if ((skel_gray_copy[r_check,c_check]==[255, 160, 0]).all()):
                                        r_end=r_check
                                        c_end=c_check
                                        done=False
                                        break
                                elif ((skel_gray_copy[r_check,c_check]==[150, 150, 150]).all()):
                                        count=0
                                        r_start=r_check
                                        c_start=c_check
                                        i_check=i
                                        done=True
                                        break
                                else:
                                        count+=1
                                if(count==3):
                                        r_end=r_start
                                        c_end=c_start
                                        done=False
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel5)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                j_check=0
                count=0
                done=True
                r_start,c_start=r,c
                while(done):
                        out=False
                        for i in range (-1,1):
                                for j in range (-1,1):
                                        r_check=r_start-i
                                        c_check=c_start-j
                                        if ((skel_gray_copy[r_check,c_check]==[255, 160, 0]).all()):
                                                r_end=r_check
                                                c_end=c_check
                                                done=False
                                                out=True
                                                break
                                        elif ((skel_gray_copy[r_check,c_check]==[150, 150, 150]).all()):
                                                count=0
                                                r_start=r_check
                                                c_start=c_check
                                                i_check=i
                                                j_check=j
                                                out=True
                                                done=True
                                                break
                                        else:
                                                count+=1
                                        if(count==3):
                                                r_end=r_start
                                                c_end=c_start
                                                out=True
                                                done=False
                                                break
                                if(out):
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel6)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                count=0
                done=True
                r_start,c_start=r,c
                while(done):
                        for i in range (-1,2):
                                r_check=r_start+i
                                c_check=c_start+1
                                if ((skel_gray_copy[r_check,c_check]==[255, 160, 0]).all()):
                                        r_end=r_check
                                        c_end=c_check
                                        done=False
                                        break
                                elif ((skel_gray_copy[r_check,c_check]==[150, 150, 150]).all()):
                                        count=0
                                        r_start=r_check
                                        c_start=c_check                                       
                                        i_check=i
                                        done=True
                                        break
                                else:
                                        count+=1
                                if(count==3):
                                        r_end=r_start
                                        c_end=c_start
                                        done=False
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel7)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                j_check=0
                count=0
                done=True
                r_start,c_start=r,c
                while(done):
                        out=False
                        for i in range (-1,1):
                                for j in range (-1,1):
                                        r_check=r_start+i
                                        c_check=c_start-j
                                        if ((skel_gray_copy[r_check,c_check]==[255, 160, 0]).all()):
                                                r_end=r_check
                                                c_end=c_check
                                                done=False
                                                out=True
                                                break
                                        elif ((skel_gray_copy[r_check,c_check]==[150, 150, 150]).all()):
                                                count=0
                                                r_start=r_check
                                                c_start=c_check
                                                i_check=i
                                                j_check=j
                                                out=True
                                                done=True
                                                break
                                        else:
                                                count+=1
                                        if(count==3):
                                                r_end=r_start
                                                c_end=c_start
                                                out=True
                                                done=False
                                                break
                                if(out):
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)
        for (r,c) in skel_coords[:]:
                #cv2.circle(skel_gray_copy, (c,r), 5, (255, 255, 255))
                skel_gray_copy[r,c]=(255, 255, 255)
                #cv2.circle(skel_gray_copy_show, (c,r), 5, (255, 255, 255))

def Pre_connected_component_labeling_and_analysis(image,min_remove_pixel,max_remove_pixel):
    global pre_average_length_rice,pre_number_grain,pre_average_area_rice
    output = cv2.connectedComponentsWithStats(image,4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    pre_number_grain=0
    pre_average_area_rice_test=0
    count=0
    pre_average_length_rice=0
    pre_average_area_rice=0
    for i in range(1, int(0.2*numLabels)+1):
        area = stats[i, cv2.CC_STAT_AREA]
        if((area>min_remove_pixel)and(area<max_remove_pixel)):
            pre_average_area_rice_test=(pre_average_area_rice_test*(i-1)+area)/(i)

    for i in range(1, int(0.2*numLabels)+1):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if((area>min_remove_pixel)and(area<max_remove_pixel)and(area<1.35*pre_average_area_rice_test)):
            count=count+1
            length_rice=(w**2+h**2)**0.5
            pre_average_length_rice=(pre_average_length_rice*(count-1)+length_rice)/(count)
            pre_average_area_rice=(pre_average_area_rice*(count-1)+area)/(count)
            pre_number_grain=pre_number_grain+1
    print("pre_average_area_rice:",pre_average_area_rice)
    print("pre_average_length_rice:",pre_average_length_rice)

def Draw_line_through_end_point_and_branch_point(imga,color_line,color_point,range_line):
    for i in range(len(skel_coords)):
        (x2,y2)=skel_coords[i]
        (x1,y1)=skel_coords_branch_connect[i]
        dx = x2 - x1
        dy = y2 - y1
        x = x2
        y = y2
        pixel=range_line
        imga[x,y] = color_line
        if((x2-x1)>=(y2-y1)and(x2>x1)and(y2>y1)):
            D = 2*dy - dx
            DE = 2*dy
            DNE = 2*(dy-dx) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    y=y+1
                else:
                    D=D+DE
                x=x+1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==(150,150,150)).all()):
                    imga[x,y]=color_point
                    break
                if((imga[x,y]==color_line).all()):
                    imga[x,y]=color_point
                    break
                else:
                    imga[x,y] = color_line
                    pixel=pixel-1
            
        if(((x2-x1)<(y2-y1))and(x2>=x1)and(y2>y1)):
            D=2*dx - dy
            DE = 2*dx
            DNE = 2*(dx-dy)
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    x=x+1
                else:
                    D=D+DE
                y=y+1
                if((x==length)or(y==height)):
                    break                
                if((imga[x,y]==(150,150,150)).all()):
                    imga[x,y]=color_point
                    break
                if((imga[x,y]==color_line).all()):
                    imga[x,y]=color_point
                    break                
                else:
                    imga[x,y] = color_line
                    pixel=pixel-1
            
        if(((x1-x2)<(y2-y1))and(x1>x2)and(y2>y1)):
            D=2*(-dx) - dy
            DE = 2*(-dx)
            DNE = 2*(-dx-dy)
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    x=x-1
                else:
                    D=D+DE
                y=y+1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==(150,150,150)).all()):
                    imga[x,y]=color_point
                    break
                if((imga[x,y]==color_line).all()):
                    imga[x,y]=color_point
                    break                
                else:
                    imga[x,y] = color_line
                    pixel=pixel-1
            
        if((x1-x2)>=(y2-y1)and(x1>x2)and(y2>=y1)):
            D = 2*dy - (-dx)
            DE = 2*dy
            DNE = 2*(dy-(-dx)) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    y=y+1
                else:
                    D=D+DE
                x=x-1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==(150,150,150)).all()):
                    imga[x,y]=color_point
                    break
                if((imga[x,y]==color_line).all()):
                    imga[x,y]=color_point
                    break                
                else:
                    imga[x,y] = color_line
                    pixel=pixel-1
            
        if((x1-x2)>=(y1-y2)and(x1>x2)and(y1>y2)):
            D = 2*(-dy) - (-dx)
            DE = 2*(-dy)
            DNE = 2*((-dy)-(-dx)) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    y=y-1
                else:
                    D=D+DE
                x=x-1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==(150,150,150)).all()):
                    imga[x,y]=color_point
                    break
                if((imga[x,y]==color_line).all()):
                    imga[x,y]=color_point
                    break                
                else:
                    imga[x,y] = color_line
                    pixel=pixel-1
            
        if((x1-x2)<(y1-y2)and(x1>=x2)and(y1>y2)):
            D = 2*(-dx) - (-dy)
            DE = 2*(-dx)
            DNE = 2*((-dx)-(-dy)) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    x=x-1
                else:
                    D=D+DE
                y=y-1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==(150,150,150)).all()):
                    imga[x,y]=color_point
                    break
                if((imga[x,y]==color_line).all()):
                    imga[x,y]=color_point
                    break                
                else:
                    imga[x,y] = color_line
                    pixel=pixel-1
            
        if((x2-x1)<(y1-y2)and(x2>x1)and(y1>y2)):
            D = 2*dx - (-dy)
            DE = 2*dx
            DNE = 2*(dx-(-dy)) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    x=x+1
                else:
                    D=D+DE
                y=y-1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==(150,150,150)).all()):
                    imga[x,y]=color_point
                    break
                if((imga[x,y]==color_line).all()):
                    imga[x,y]=color_point
                    break                
                else:
                    imga[x,y] = color_line
                    pixel=pixel-1
            
        if((x2-x1)>=(y1-y2)and(x2>x1)and(y1>=y2)):
            D = 2*(-dy) - dx
            DE = 2*(-dy)
            DNE = 2*((-dy)-dx) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    y=y-1
                else:
                    D=D+DE
                x=x+1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==(150,150,150)).all()):
                    imga[x,y]=color_point
                    break
                if((imga[x,y]==color_line).all()):
                    imga[x,y]=color_point
                    break                
                else:
                    imga[x,y] = color_line
                    pixel=pixel-1

def Draw_line_between_end_point_and_connect_point(imga,color_line,color_point,range_line):
    global endpoint,connectpoint,endpoint_check,connectpoint_check
    endpoint=[]
    connectpoint=[]
    endpoint_check=[]
    connectpoint_check=[]
    x_check=0
    y_check=0
    for i in range(len(skel_coords)):
        (x2,y2)=skel_coords[i]
        (x1,y1)=skel_coords_branch_connect[i]
        dx = x2 - x1
        dy = y2 - y1
        x = x2
        y = y2
        pixel=range_line
        imga[x,y] = color_line
        if((x2-x1)>=(y2-y1)and(x2>x1)and(y2>y1)):
            D = 2*dy - dx
            DE = 2*dy
            DNE = 2*(dy-dx) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    y=y+1
                else:
                    D=D+DE
                x=x+1
                if((x==length)or(y==height)):
                    break                
                if((imga[x,y]==color_point).all()):
                    endpoint.append((x2,y2))
                    connectpoint.append((x,y))
                    break
                else:
                    imga[x,y] = color_line
                    if (pixel==int(0.5*range_line)):
                        x_check=x
                        y_check=y
                    pixel=pixel-1
            while(pixel==0):
                endpoint_check.append((x2,y2))
                connectpoint_check.append((x_check,y_check))
                break

        if(((x2-x1)<(y2-y1))and(x2>=x1)and(y2>y1)):
            D=2*dx - dy
            DE = 2*dx
            DNE = 2*(dx-dy)
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    x=x+1
                else:
                    D=D+DE
                y=y+1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==color_point).all()):
                    endpoint.append((x2,y2))
                    connectpoint.append((x,y))
                    break           
                else:
                    imga[x,y] = color_line
                    if (pixel==int(0.5*range_line)):
                        x_check=x
                        y_check=y
                    pixel=pixel-1
            while(pixel==0):
                endpoint_check.append((x2,y2))
                connectpoint_check.append((x_check,y_check))
                break
            
        if(((x1-x2)<(y2-y1))and(x1>x2)and(y2>y1)):
            D=2*(-dx) - dy
            DE = 2*(-dx)
            DNE = 2*(-dx-dy)
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    x=x-1
                else:
                    D=D+DE
                y=y+1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==color_point).all()):
                    endpoint.append((x2,y2))
                    connectpoint.append((x,y))
                    break   
                else:
                    imga[x,y] = color_line
                    if (pixel==int(0.5*range_line)):
                        x_check=x
                        y_check=y
                    pixel=pixel-1
            while(pixel==0):
                endpoint_check.append((x2,y2))
                connectpoint_check.append((x_check,y_check))
                break
            
        if((x1-x2)>=(y2-y1)and(x1>x2)and(y2>=y1)):
            D = 2*dy - (-dx)
            DE = 2*dy
            DNE = 2*(dy-(-dx)) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    y=y+1
                else:
                    D=D+DE
                x=x-1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==color_point).all()):
                    endpoint.append((x2,y2))
                    connectpoint.append((x,y))
                    break       
                else:
                    imga[x,y] = color_line
                    if (pixel==int(0.5*range_line)):
                        x_check=x
                        y_check=y
                    pixel=pixel-1
            while(pixel==0):
                endpoint_check.append((x2,y2))
                connectpoint_check.append((x_check,y_check))
                break
            
        if((x1-x2)>=(y1-y2)and(x1>x2)and(y1>y2)):
            D = 2*(-dy) - (-dx)
            DE = 2*(-dy)
            DNE = 2*((-dy)-(-dx)) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    y=y-1
                else:
                    D=D+DE
                x=x-1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==color_point).all()):
                    endpoint.append((x2,y2))
                    connectpoint.append((x,y))
                    break             
                else:
                    imga[x,y] = color_line
                    if (pixel==int(0.5*range_line)):
                        x_check=x
                        y_check=y
                    pixel=pixel-1
            while(pixel==0):
                endpoint_check.append((x2,y2))
                connectpoint_check.append((x_check,y_check))
                break
            
        if((x1-x2)<(y1-y2)and(x1>=x2)and(y1>y2)):
            D = 2*(-dx) - (-dy)
            DE = 2*(-dx)
            DNE = 2*((-dx)-(-dy)) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    x=x-1
                else:
                    D=D+DE
                y=y-1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==color_point).all()):
                    endpoint.append((x2,y2))
                    connectpoint.append((x,y))
                    break      
                else:
                    imga[x,y] = color_line
                    if (pixel==int(0.5*range_line)):
                        x_check=x
                        y_check=y
                    pixel=pixel-1
            while(pixel==0):
                endpoint_check.append((x2,y2))
                connectpoint_check.append((x_check,y_check))
                break
            
        if((x2-x1)<(y1-y2)and(x2>x1)and(y1>y2)):
            D = 2*dx - (-dy)
            DE = 2*dx
            DNE = 2*(dx-(-dy)) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    x=x+1
                else:
                    D=D+DE
                y=y-1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==color_point).all()):
                    endpoint.append((x2,y2))
                    connectpoint.append((x,y))
                    break            
                else:
                    imga[x,y] = color_line
                    if (pixel==int(0.5*range_line)):
                        x_check=x
                        y_check=y
                    pixel=pixel-1
            while(pixel==0):
                endpoint_check.append((x2,y2))
                connectpoint_check.append((x_check,y_check))
                break
            
        if((x2-x1)>=(y1-y2)and(x2>x1)and(y1>=y2)):
            D = 2*(-dy) - dx
            DE = 2*(-dy)
            DNE = 2*((-dy)-dx) 
            while(pixel!=0):
                if(D>0):
                    D=D+DNE
                    y=y-1
                else:
                    D=D+DE
                x=x+1
                if((x==length)or(y==height)):
                    break
                if((imga[x,y]==color_point).all()):
                    endpoint.append((x2,y2))
                    connectpoint.append((x,y))
                    break           
                else:
                    imga[x,y] = color_line
                    if (pixel==int(0.5*range_line)):
                        x_check=x
                        y_check=y
                    pixel=pixel-1
            while(pixel==0):
                endpoint_check.append((x2,y2))
                connectpoint_check.append((x_check,y_check))
                break

def Connect_error_point_to_nearest_point(image,range_check):
    pixel=range_check
    radius_circle=int(range_check)
    ex_end=0
    ey_end=0
    x_1_end=0
    y_1_end=0
    for i in range(len(connectpoint_check)):
        count=0
        for j in range(len(connectpoint_check)):
            (x1,y1)=connectpoint_check[i]
            (x2,y2)=connectpoint_check[j]
            distance= ((x1-x2)**2+(y1-y2)**2)**0.5
            if((distance>0)and(distance<pixel)):
                x=int((x1+x2)*0.5)
                y=int((y1+y2)*0.5)
                ex,ey=endpoint_check[i]
                endpoint.append((ex,ey))
                connectpoint.append((x,y))
                count=count+1
        if(count==0):
            (x1,y1)=connectpoint_check[i]
            ex,ey=endpoint_check[i]
            distance_min_quared=radius_circle**2
            for x in range(0, +radius_circle+1):
                for y in range(0, +radius_circle+1):
                    x_1=x1+x
                    y_1=y1+y
                    if((x_1>=length)or(y_1>=height)):
                        break
                    distancesquared=((x_1-x1)**2)+((y_1-y1)**2)         
                    if(distancesquared<=(radius_circle**2)):
                        if((image[x_1,y_1]==(150,150,150)).all()):
                            if(distancesquared<distance_min_quared):
                                ex_end=ex
                                ey_end=ey
                                x_1_end=x_1
                                y_1_end=y_1
                                
                    x_1=x1-x
                    y_1=y1+y
                    if((x_1>=length)or(y_1>=height)):
                        break
                    distancesquared=((x_1-x1)**2)+((y_1-y1)**2)         
                    if(distancesquared<=(radius_circle**2)):
                        if((image[x_1,y_1]==(150,150,150)).all()):
                            if(distancesquared<distance_min_quared):
                                ex_end=ex
                                ey_end=ey
                                x_1_end=x_1
                                y_1_end=y_1
                            
                    x_1=x1+x
                    y_1=y1-y
                    if((x_1>=length)or(y_1>=height)):
                        break
                    distancesquared=((x_1-x1)**2)+((y_1-y1)**2)         
                    if(distancesquared<=(radius_circle**2)):
                        if((image[x_1,y_1]==(150,150,150)).all()):
                            if(distancesquared<distance_min_quared):
                                ex_end=ex
                                ey_end=ey
                                x_1_end=x_1
                                y_1_end=y_1
                            
                    x_1=x1-x
                    y_1=y1-y
                    if((x_1>=length)or(y_1>=height)):
                        break
                    distancesquared=((x_1-x1)**2)+((y_1-y1)**2)         
                    if(distancesquared<=(radius_circle**2)):
                        if((image[x_1,y_1]==(150,150,150)).all()):
                            if(distancesquared<distance_min_quared):
                                ex_end=ex
                                ey_end=ey
                                x_1_end=x_1
                                y_1_end=y_1

            endpoint.append((ex_end,ey_end))
            connectpoint.append((x_1_end,y_1_end))      
                            
def Draw_line_in_binary_image(image,think):
    for i in range(len(endpoint)):
        (x1,y1)=endpoint[i]
        (x2,y2)=connectpoint[i]
        cv2.line(image,(y1,x1),(y2,x2),(0,0,0),think)

def Connected_component_labeling_and_analysis(image,min_remove_pixel,max_remove_pixel,resize,one_mm):
    
    output = cv2.connectedComponentsWithStats(image,4, cv2.CV_32S)
    global number_grain,average_area_rice,average_length_rice
    (numLabels, labels, stats, centroids) = output
    number_grain=0
    average_area_rice=0
    average_length_rice=0
    count=0
    one_mm_pixel=one_mm
    count_long_grain=0
    count_short_grain=0
    broken_rice_a=0
    broken_rice_b=0
    broken_rice_c=0
    broken_rice_d=0
    broken_rice_e=0
    broken_rice_f=0
    broken_rice_small_a=0
    broken_rice_small_b=0
    broken_rice_small_c=0
    broken_rice_small_d=0
    broken_rice_small_e=0
    broken_rice_small_f=0
    whole_rice_a=0
    whole_rice_b=0
    whole_rice_c=0
    whole_rice_d=0
    whole_rice_e=0
    whole_rice_f=0
    Found=True
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if((area>min_remove_pixel)and(area<max_remove_pixel)):
            
            (cX, cY) = centroids[i]
            output = img.copy()
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
            componentMask = (labels == i).astype("uint8") * 255
            
            output=cv2.resize(output,resize)
            componentMask=cv2.resize(componentMask,resize)
            cv2.imshow("Output", output)
            cv2.imshow("Connected Component", componentMask)
            cv2.waitKey(0)
            
            if((area>=1.35*pre_average_area_rice)and((area<2.35*pre_average_area_rice))):
                number_grain=number_grain+2
                '''
                print(+2)
                print(area)
                '''
            if((area>=2.35*pre_average_area_rice)and((area<3.35*pre_average_area_rice))):
                number_grain=number_grain+3
                '''
                print(+3)
                print(area)
                '''
            if(area<1.35*pre_average_area_rice):
                count=count+1
                number_grain=number_grain+1
                length_rice=(w**2+h**2)**0.5
                average_length_rice=(average_length_rice*(count-1)+length_rice)/(count)
                average_area_rice=(average_area_rice*(count-1)+area)/(count)

                if(length_rice>7*one_mm_pixel):
                    count_long_grain=count_long_grain+1
                if(length_rice<6*one_mm_pixel):
                    count_short_grain=count_short_grain+1
                    
                if((length_rice>0.5*average_length_rice)and(length_rice<0.8*average_length_rice)):
                    broken_rice_a=broken_rice_a+1
                if(length_rice<=0.5*average_length_rice):
                    broken_rice_small_a=broken_rice_small_a+1
                    
                if((length_rice>0.35*average_length_rice)and(length_rice<0.75*average_length_rice)):
                    broken_rice_b=broken_rice_b+1
                if(length_rice<=0.35*average_length_rice):
                    broken_rice_small_b=broken_rice_small_b+1
                    
                if((length_rice>0.35*average_length_rice)and(length_rice<0.7*average_length_rice)):
                    broken_rice_c=broken_rice_c+1
                if(length_rice<=0.35*average_length_rice):
                    broken_rice_small_c=broken_rice_small_c+1
                    
                if((length_rice>0.35*average_length_rice)and(length_rice<0.65*average_length_rice)):
                    broken_rice_d=broken_rice_d+1
                if(length_rice<=0.35*average_length_rice):
                    broken_rice_small_d=broken_rice_small_d+1
                    
                if((length_rice>0.25*average_length_rice)and(length_rice<0.6*average_length_rice)):
                    broken_rice_e=broken_rice_e+1
                if(length_rice<=0.25*average_length_rice):
                    broken_rice_small_e=broken_rice_small_e+1
                    
                if((length_rice>0.25*average_length_rice)and(length_rice<0.5*average_length_rice)):
                    broken_rice_f=broken_rice_f+1
                if(length_rice<=0.25*average_length_rice):
                    broken_rice_small_f=broken_rice_small_f+1
                    
            whole_rice_a=count-broken_rice_a-broken_rice_small_a
            whole_rice_b=count-broken_rice_b-broken_rice_small_b
            whole_rice_c=count-broken_rice_c-broken_rice_small_c
            whole_rice_d=count-broken_rice_d-broken_rice_small_d
            whole_rice_e=count-broken_rice_e-broken_rice_small_e
            whole_rice_f=count-broken_rice_f-broken_rice_small_f
    
    while(True):
        if(((count_short_grain/count)>0.75)):
            if(((whole_rice_b/count)>=0.6)and(((broken_rice_b/count)<=0.07)and((broken_rice_small_b/count)<=0.002))):
                print("Hat gao ngan 5%")
                break
            if(((whole_rice_c/count)>=0.55)and(((broken_rice_c/count)<=0.12)and((broken_rice_small_c/count)<=0.003))):
                print("Hat gao ngan 10%")
                break
            else:
                print("error Hat ngan 75%")
                break
            
        if((count_short_grain/count)>0.7):
            if(((whole_rice_d/count)>=0.5)and(((broken_rice_d/count)<=0.17)and((broken_rice_small_d/count)<=0.005))):
                print("Hat gao ngan 15%")
                break
            if(((whole_rice_e/count)>=0.45)and(((broken_rice_e/count)<=0.22)and((broken_rice_small_e/count)<=0.01))):
                print("Hat gao ngan 20%")
                break
            if(((whole_rice_f/count)>=0.40)and(((broken_rice_f/count)<=0.27)and((broken_rice_small_f/count)<=0.02))):
                print("Hat gao ngan 25%")
                break
            else:
                print("error Hat ngan 70%")
                break

        if(((count_short_grain/count)<=0.10)and((count_long_grain/count)>=0.1)and()):
            if(((whole_rice_a/count)>=0.60)and(((broken_rice_a/count)<0.04)and((broken_rice_small_a/count)<=0.001))):
                print("Hat gao dai 100% loai A")
                break
            if(((whole_rice_a/count)>=0.60)and(((broken_rice_a/count)<0.045)and((broken_rice_small_a/count)<=0.001))):
               print("Hat gao dai 100% loai B")
               break
            else:
                print("error Hat ngan 10%")
                break
            
        if(((count_short_grain/count)<=0.15)and((count_long_grain/count)>=0.05)):
            if(((whole_rice_b/count)>=0.60)and(((broken_rice_b/count)<=0.07)and((broken_rice_small_b/count)<=0.002))):
                print("Hat gao dai 5%")
                break
            if(((whole_rice_c/count)>=0.55)and(((broken_rice_c/count)<=0.12)and((broken_rice_small_c/count)<=0.003))):
                print("Hat gao dai 10%")
                break
            else:
                print("error Hat ngan 15%")
                break

        if((count_short_grain/count)<0.3):
            if(((whole_rice_d/count)>=0.50)and(((broken_rice_d/count)<=0.17)and((broken_rice_small_d/count)<=0.005))):
                print("Hat gao dai 15%")
                break
            else:
                print("error Hat ngan 30%")
                break

        if((count_short_grain/count)<0.5):
            if(((whole_rice_e/count)>=0.45)and(((broken_rice_e/count)<=0.22)and((broken_rice_small_e/count)<=0.01))):
                print("Hat gao dai 20%")
                break
            if(((whole_rice_f/count)>=0.40)and(((broken_rice_f/count)<=0.27)and((broken_rice_small_f/count)<=0.02))):
                print("Hat gao dai 25%")
                break
            else:
                print("error Hat ngan 50%")
                break
        else:
            print("not found")
            break

    '''
    print("number_grain:",number_grain)
    print("average_area_rice:",average_area_rice)
    print("average_length_rice:",average_length_rice)
    print("count_long_grain:",count_long_grain)
    print("count_short_grain:",count_short_grain)
    print("broken_rice_a:",broken_rice_a)
    print("broken_rice_small_a:",broken_rice_small_a)
    print("whole_rice_a:",whole_rice_a)
    print("broken_rice_b:",broken_rice_b)
    print("broken_rice_small_b:",broken_rice_small_b)
    print("whole_rice_b:",whole_rice_b)
    print("broken_rice_c:",broken_rice_c)
    print("broken_rice_small_c:",broken_rice_small_c)
    print("whole_rice_c:",whole_rice_c)
    print("broken_rice_d:",broken_rice_d)
    print("broken_rice_small_d:",broken_rice_small_d)
    print("whole_rice_d:",whole_rice_d)
    print("broken_rice_e:",broken_rice_e)
    print("broken_rice_small_e:",broken_rice_small_e)
    print("whole_rice_e:",whole_rice_e)
    print("broken_rice_f:",broken_rice_f)
    print("broken_rice_small_f:",broken_rice_small_f)
    print("whole_rice_f:",whole_rice_f)
    '''
    print("number_grain:",number_grain)
    print("average_area_rice:",average_area_rice)
    print("average_length_rice:",average_length_rice)
    print("count_long_grain:",count_long_grain)
    print("count_short_grain:",count_short_grain)

def Show_image(I_binary,skel_gray,skel_gray_copy_show,skel_gray_copy,I_binary_copy,img):
        fig = plt.figure(figsize=(10, 7))
        rows = 3
        columns = 2
        
        fig.add_subplot(rows, columns, 1)
        plt.imshow(I_binary)
        
        fig.add_subplot(rows, columns, 2)
        plt.imshow(skel_gray)

        fig.add_subplot(rows, columns, 3)
        plt.imshow(skel_gray_copy_show)
                
        fig.add_subplot(rows, columns, 4)
        plt.imshow(skel_gray_copy)
        
        fig.add_subplot(rows, columns, 5)
        plt.imshow(I_binary_copy)

        fig.add_subplot(rows, columns, 6)
        plt.imshow(img)
        
        plt.show()

Get_skeleton_image_and_remove_ruler("img/sample1.jpg")
Ruler_process()

Kernel_to_find_endpoint()
Kernel_to_find_branch_point()
Find_branch_point(skel_gray)
Find_end_point_and_connect_to_branch_point(skel_gray,2)
Pre_connected_component_labeling_and_analysis(I_draw,0.02*length*height*0.02,0.1*0.1*length*height)#0.02;0.01
Draw_line_through_end_point_and_branch_point(skel_gray_copy_show,(255,255,255),(255, 160, 0),int(pre_average_length_rice*0.4))
Draw_line_between_end_point_and_connect_point(skel_gray_copy_show,(255, 0, 0),(255, 160, 0),int(pre_average_length_rice*0.4))
Connect_error_point_to_nearest_point(skel_gray_copy_show,int(pre_average_length_rice*0.2))
Draw_line_in_binary_image(I_draw,3)
Connected_component_labeling_and_analysis(I_draw,0.02*length*height*0.02,0.1*0.1*length*height,(500,800),average_distance_number/10)
Show_image(I_binary,skel_gray,skel_gray_copy_show,skel_gray_copy,I_binary_copy,I_draw)

#color_line_([150 150 150])