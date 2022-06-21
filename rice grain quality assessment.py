import cv2
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize, binary_closing
from skimage import morphology
from scipy import ndimage
from skimage import io, filters


def Get_skeleton_image_and_remove_ruler(img_file):
        global skel_gray,I_binary,skel_gray_copy,skel_gray_copy_show,I_binary_copy,img,opening,length,height,width,I_draw,ruler_image,ruler_background,gray_image,I
        kernel=np.ones((3,3),np.uint8)
        kernel_7=np.ones((7,7),np.uint8)
        img = cv2.imread(img_file)
        (length,height,width)=np.shape(img)
        I_gray = cv2.imread(img_file, 0)
        gray_image=I_gray.copy()
        T,I_gray = cv2.threshold(I_gray,150,255,cv2.THRESH_BINARY)
        output = cv2.connectedComponentsWithStats(I_gray,4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        
        #dòng 22-40: Thuật toán tìm vùng diện tích lớn nhất trong bức ảnh, sử dụng để tìm kiếm phần diện tích thước.
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
        #Dòng 43-53: Lấp đầy các vùng tối trong phần diện tích thước.
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
        
        #Dòng 56: Loại bỏ toàn bộ phần diên tích thước ra khỏi bức ảnh, chỉ giữ lại phần hạt
        I_draw = cv2.subtract(I_gray,sure_bg)
        #Dòng 60-65: Tìm khung xương của bức ảnh
        I_draw=cv2.morphologyEx(I_draw,cv2.MORPH_OPEN,kernel_7,iterations=1) # thay đổi các thông số trên để có thể có được kết quả tốt nhất
        sure_bg = cv2.dilate(I_draw,kernel,iterations=3) # thay đổi các thông số trên để có thể có được kết quả tốt nhất
        I = cv2.subtract(sure_bg,I_draw)
        I_binary= cv2.merge((I,I,I))
        I_binary_copy=I_binary.copy()
        skel =skeletonize(I_binary)
        skel_gray = cv2.cvtColor(skel, cv2.COLOR_BGR2GRAY)
        skel_gray_copy = cv2.cvtColor(skel_gray, cv2.COLOR_GRAY2BGR)
        skel_gray_copy_show=skel_gray_copy.copy()

def Ruler_process(rate_min_remove,rate_max_remove):
    global average_distance_number,l,h
    
    #Dòng 74-78: Loại bỏ phần hạt ra khỏi bức ảnh chỉ giữ lại thước để xử lý
    kernel=np.ones((3,3),np.uint8)
    ruler_foreground=255-ruler_background
    ruler_image_remove_grain=cv2.subtract(gray_image,ruler_foreground)
    T,ruler_image_remove_grain_binary = cv2.threshold(ruler_image_remove_grain,150,255,cv2.THRESH_BINARY)
    ruler_image_remove_grain_binary=255-ruler_image_remove_grain_binary
    '''
    plt.imshow(ruler_image_remove_grain_binary)
    plt.show()
    '''
    #Dòng 83-85: Loại bỏ phần những phần thừa chỉ giữ lại số trên thước
    eroded = cv2.erode(ruler_image_remove_grain_binary, kernel,3)
    sure_bg = cv2.dilate(eroded,kernel,iterations=2)
    (l,h)=np.shape(sure_bg)
    #Tìm diện tích trung bình của 1 chữ số
    output = cv2.connectedComponentsWithStats(sure_bg,4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    average_area=0
    center_number=[]
    number=[]
    for i in range(3, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if((area>rate_min_remove*rate_min_remove*l*h)and(area<rate_max_remove*rate_max_remove*l*h)): #Loại bỏ những phần làm nhiễu bức ảnh
                average_area=(average_area*(i-3)+area)/(i-2)
    count=0
    diagonal_length_of_number_average=0
    
    # Lọc ra các chữ số và tìm đường chéo trung bình của chúng
    for i in range(3, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        diagonal_length=(w**2+h**2)**0.5
        if((area>0.4*average_area)and(area<2.5*average_area)):
            count=count+1
            (cX, cY) = centroids[i]
            center_number.append((cX, cY))
            diagonal_length_of_number_average=(diagonal_length_of_number_average*(count-1)+diagonal_length)/(count)
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
    #Những chữ số nào mà khoảng cách giữa 2 chữ số < đường chéo trung bình thì hiểu rằng đó là 1 số và tìm trung điểm của 2 số VD:10,11,12,13 ....
    distance_min_average=0
    average_distance_number=0
    count=0
    for i in range(len(center_number)):
        for j in range(len(center_number)):
            (x1,y1)=center_number[i]
            (x2,y2)=center_number[j]
            distance= ((x1-x2)**2+(y1-y2)**2)**0.5
            if((distance>0)and(distance<diagonal_length_of_number_average)):
                count=count+1
                distance_min_average=(distance_min_average*(count-1)+distance)/(count)
    for i in range(len(center_number)):
        count=0
        for j in range(len(center_number)):
            (x1,y1)=center_number[i]
            (x2,y2)=center_number[j]
            distance= ((x1-x2)**2+(y1-y2)**2)**0.5
            if((distance>0)and(distance<1.5*distance_min_average)):
                count=count+1
                number.append(((x1+x2)/2,(y1+y2)/2))
                break
        if(count==0):
            number.append((x1,y1))
    distance_min=1000000
    
    # Tìm kiếm khoảng cách nhỏ nhất giữa 2 số liền nhau VD: 1-2,3-4,5-6 ....
    for i in range(len(number)):
        for j in range(len(number)):
            (x1,y1)=number[i]
            (x2,y2)=number[j]
            distance= ((x1-x2)**2+(y1-y2)**2)**0.5
            if((distance>0)and(distance<distance_min)):
                distance_min=distance
                
    # Tìm kiểm khoảng cách trung bình giữa 2 số liền nhau hay chính là 1cm theo pixel
    for i in range(len(number)):
        count=0
        for j in range(len(number)):
            (x1,y1)=number[i]
            (x2,y2)=number[j]
            distance= ((x1-x2)**2+(y1-y2)**2)**0.5
            if((distance>0)and(distance<1.5*distance_min)):
                count=count+1
                average_distance_number=(average_distance_number*(count-1)+distance)/(count)
    Invalid_number=0.00# Sai số của thước, tùy mẫu thước có thể thay đổi thông số trên
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

def Find_end_point_and_connect_to_branch_point(skel_gray,max_searching_pixel,think):
        global skel_coords,skel_coords_branch_connect
        skel_coords_branch_connect = []
        skel_coords = []    
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel0)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                count=0
                pixel=int(max_searching_pixel)
                done=True
                r_start,c_start=r,c
                while(done):
                        for i in range (-1,2): #Tìm kiếm theo hướng pixel phát triển
                                r_check=r_start-1
                                c_check=c_start+i
                                if ((skel_gray_copy[r_check,c_check]==[255, 160, 0]).all()): #Nếu tìm kiếm được điểm nối ngã 3 thì dừng lại, lưu lại 2 điểm kết thúc và điểm nối
                                        r_end=r_check
                                        c_end=c_check
                                        done=False
                                        break
                                elif ((skel_gray_copy[r_check,c_check]==[150, 150, 150]).all()): #Nếu tìm kiếm được pixel khung xương thì lưu lại vị trí và bắt đầu tìm kiếm tiếp tục từ điểm pixel đó
                                        pixel=pixel-1
                                        count=0
                                        r_start=r_check
                                        c_start=c_check
                                        i_check=i
                                        done=True
                                        break
                                else:
                                        count+=1
                                if((count==3)or (pixel==0)): # Nếu quét hết các pixel mà ko tìm thấy điểm nối hay khung xương hoặc đi quá max pixel searching thì dừng lại, lấy điểm đang bắt đầu là điểm kết thức, lưu lại điểm kết thúc này và endpoint
                                        r_end=r_start
                                        c_end=c_start
                                        done=False
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think) #Nối 2 điểm vừa tìm được để kiểm tra hướng phát triển của đường thẳng
        
        #Tương tự trên chỉ thay đổi hướng phát triển                        
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel1)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                j_check=0
                count=0
                pixel=int(max_searching_pixel)
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
                                                pixel=pixel-1
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
                                        if((count==3)or(pixel==0)):
                                                r_end=r_start
                                                c_end=c_start
                                                done=False
                                                out=True
                                                break
                                if(out):
                                        break                                
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)

        #Tương tự trên chỉ thay đổi hướng phát triển                         
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel2)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                count=0
                pixel=int(max_searching_pixel)
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
                                        pixel=pixel-1
                                        count=0
                                        r_start=r_check
                                        c_start=c_check
                                        i_check=i
                                        done=True
                                        break
                                else:
                                        count+=1
                                if((count==3)or(pixel==0)):
                                        r_end=r_start
                                        c_end=c_start
                                        done=False
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)
                
        #Tương tự trên chỉ thay đổi hướng phát triển         
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel3)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                j_check=0
                count=0
                pixel=int(max_searching_pixel)
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
                                                pixel=pixel-1
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
                                        if((count==3)or(pixel==0)):
                                                r_end=r_start
                                                c_end=c_start
                                                done=False
                                                out=True
                                                break
                                if(out):
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)
                
        #Tương tự trên chỉ thay đổi hướng phát triển         
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel4)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                count=0
                pixel=int(max_searching_pixel)
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
                                        pixel=pixel-1
                                        count=0
                                        r_start=r_check
                                        c_start=c_check
                                        i_check=i
                                        done=True
                                        break
                                else:
                                        count+=1
                                if((count==3)or(pixel==0)):
                                        r_end=r_start
                                        c_end=c_start
                                        done=False
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)

        #Tương tự trên chỉ thay đổi hướng phát triển
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel5)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                j_check=0
                count=0
                pixel=int(max_searching_pixel)
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
                                                pixel=pixel-1
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
                                        if((count==3)or(pixel==0)):
                                                r_end=r_start
                                                c_end=c_start
                                                out=True
                                                done=False
                                                break
                                if(out):
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)

        #Tương tự trên chỉ thay đổi hướng phát triển         
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel6)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                count=0
                pixel=int(max_searching_pixel)
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
                                        pixel=pixel-1
                                        count=0
                                        r_start=r_check
                                        c_start=c_check                                       
                                        i_check=i
                                        done=True
                                        break
                                else:
                                        count+=1
                                if((count==3)or(pixel==0)):
                                        r_end=r_start
                                        c_end=c_start
                                        done=False
                                        break
                skel_coords_branch_connect.append((r_end,c_end))
                cv2.line(skel_gray_copy,(c,r),(c_end,r_end),(240,248,255),think)

        #Tương tự trên chỉ thay đổi hướng phát triển         
        output_image = cv2.morphologyEx(skel_gray, cv2.MORPH_HITMISS, kernel7)
        (rows,cols) = np.nonzero(output_image)
        for (r,c) in zip(rows,cols):
                skel_coords.append((r,c))
                i_check=0
                j_check=0
                count=0
                pixel=int(max_searching_pixel)
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
                                                pixel=pixel-1
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
                                        if((count==3)or(pixel==0)):
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

def Pre_connected_component_labeling_and_analysis(image,rate_min_remove,rate_max_remove):
    global pre_average_length_rice,pre_number_grain,pre_average_area_rice
    
    #Tìm kiếm khoảng 20% số hạt gạo phía trên cùng để lấy thông số hạt gạo như diện tích và chiều dài
    output = cv2.connectedComponentsWithStats(image,4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    pre_number_grain=0
    pre_average_area_rice_test=0
    count=0
    pre_average_length_rice=0
    pre_average_area_rice=0
    for i in range(1, int(0.2*numLabels)+1):
        area = stats[i, cv2.CC_STAT_AREA]
        if((area>rate_min_remove*length*height*rate_min_remove)and(area<rate_max_remove*length*height*rate_max_remove)):
            count=count+1
            pre_average_area_rice_test=(pre_average_area_rice_test*(count-1)+area)/(count)
            
    count=0
    for i in range(1, int(0.2*numLabels)+1):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if((area>rate_min_remove*length*height*rate_min_remove)and(area<rate_max_remove*length*height*rate_max_remove)and(area<1.35*pre_average_area_rice_test)):
            count=count+1
            length_rice=(w**2+h**2)**0.5 # Tính chiều dài hạt gạo bằng chiều dài đường chéo, có thể tìm thuật toán tối ưu hơn
            pre_average_length_rice=(pre_average_length_rice*(count-1)+length_rice)/(count)
            pre_average_area_rice=(pre_average_area_rice*(count-1)+area)/(count)
            pre_number_grain=pre_number_grain+1
    print("pre_average_area_rice:",pre_average_area_rice)
    print("pre_average_length_rice:",pre_average_length_rice)

def Draw_line_through_end_point_and_branch_point(imga,color_line,color_point,range_line):
    
    # Từ hướng phát triển của 2 điểm kết thúc và điểm đã tìm kiếm bên trên, ta tiếp tục mở rộng hướng đường thẳng để tìm các đường thẳng cắt điểm giao nhau của hạt
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
                if((x==length)or(y==height)): #Nếu đi ra quá chiều dài và chiều rộng bức ảnh thì dừng lại
                    break
                if((imga[x,y]==(150,150,150)).all()): #Nếu tìm thấy điểm khung xương thì đánh dấu lại 
                    imga[x,y]=color_point
                    break
                if((imga[x,y]==color_line).all()): #Nếu tìm thấy đường thẳng khác thì đánh dấu lại
                    imga[x,y]=color_point
                    break
                else:
                    imga[x,y] = color_line #nếu không thấy thì tiếp tục mở rộng cho đến khi pixel=0
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
                if((imga[x,y]==color_point).all()): #Nếu tìm thấy điểm đánh dấu bên trên thì lưu lại 2 điểm kết thức và điểm dánh dấu
                    endpoint.append((x2,y2))
                    connectpoint.append((x,y))
                    break
                else:
                    imga[x,y] = color_line
                    if (pixel==int(0.5*range_line)): #Lưu lại điểm này sau khi đã tìm kiếm được 1 nửa số lần pixel
                        x_check=x
                        y_check=y
                    pixel=pixel-1
            while(pixel==0): #Nếu không tìm thấy điểm đánh dấu thì lưu lại điểm kết thúc và điểm đi được 1 nủa số lần pixel ngay trên
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
    #Tìm cách kết nối các đường thẳng không tìm thấy điểm để cắt
    pixel=range_check
    radius_circle=int(range_check)
    ex_end=0
    ey_end=0
    x_1_end=0
    y_1_end=0
    for i in range(len(connectpoint_check)):
        count=0
        for j in range(len(connectpoint_check)): #Cố gắng kết nối các điểm không thấy đường cắt
            (x1,y1)=connectpoint_check[i]
            (x2,y2)=connectpoint_check[j]
            distance= ((x1-x2)**2+(y1-y2)**2)**0.5
            if((distance>0)and(distance<pixel)): #Trong bán kính pixel nếu có 2 điểm kết nối check thì tìm trung điểm của 2 điểm, lưu điểm này lại với 2 điểm endpoint
                x=int((x1+x2)*0.5)
                y=int((y1+y2)*0.5)
                ex,ey=endpoint_check[i]
                endpoint.append((ex,ey))
                connectpoint.append((x,y))
                count=count+1
        if(count==0):# Nếu như không tìm thấy điểm kết nối check nào trong bán kính thì tìm kiếm điểm khung xương trong bán kính mà gần nhất với điểm check
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
    #Vẽ các đường cắt đã tìm được vào hình
    for i in range(len(endpoint)):
        (x1,y1)=endpoint[i]
        (x2,y2)=connectpoint[i]
        cv2.line(image,(y1,x1),(y2,x2),(0,0,0),think)

def Connected_component_labeling_and_analysis(image,rate_min_remove,rate_max_remove,resize,one_mm):
    #Lọc ra từng hạt và phân tích sau khi đã cắt các hạt chạm nhau
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
        if((area>rate_min_remove*length*height*rate_min_remove)and(area<rate_max_remove*length*height*rate_max_remove)):
            
            (cX, cY) = centroids[i]
            output = img.copy()
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
            componentMask = (labels == i).astype("uint8") * 255
            '''
            output=cv2.resize(output,resize)
            componentMask=cv2.resize(componentMask,resize)
            cv2.imshow("Output", output)
            cv2.imshow("Connected Component", componentMask)
            cv2.waitKey(0)
            '''
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
        cv2.imwrite("img/I_binary.jpg",I_binary)
        
        fig.add_subplot(rows, columns, 2)
        plt.imshow(skel_gray)
        cv2.imwrite("img/skel_gray.jpg",skel_gray)

        fig.add_subplot(rows, columns, 3)
        plt.imshow(skel_gray_copy_show)
                
        fig.add_subplot(rows, columns, 4)
        plt.imshow(skel_gray_copy)
        
        fig.add_subplot(rows, columns, 5)
        plt.imshow(I_binary_copy)

        fig.add_subplot(rows, columns, 6)
        plt.imshow(img)
        
        plt.show()

Get_skeleton_image_and_remove_ruler("img/sample.jpg")
Kernel_to_find_endpoint()
Kernel_to_find_branch_point()
Ruler_process(0.005,0.1)
Pre_connected_component_labeling_and_analysis(I_draw,0.001,0.1)
Find_branch_point(skel_gray)
Find_end_point_and_connect_to_branch_point(skel_gray,pre_average_length_rice*0.2,2)
Draw_line_through_end_point_and_branch_point(skel_gray_copy_show,(255,255,255),(255, 160, 0),int(pre_average_length_rice*0.4))
Draw_line_between_end_point_and_connect_point(skel_gray_copy_show,(255, 0, 0),(255, 160, 0),int(pre_average_length_rice*0.4))
Connect_error_point_to_nearest_point(skel_gray_copy_show,int(pre_average_length_rice*0.2))
Draw_line_in_binary_image(I_draw,2)
Connected_component_labeling_and_analysis(I_draw,0.001,0.1,(500,800),average_distance_number/10)
Show_image(I_binary,skel_gray,skel_gray_copy_show,skel_gray_copy,I_binary_copy,I_draw)
plt.imshow(skel_gray_copy_show)
plt.show()
#color_line_([150 150 150])