import cv2
import numpy as np 
import argparse

def find_crop_boundaries(gradient_sum):
    len_thresh = 10 
    smooth =1 
    

    # 找到非空白区域的起始和结束索引
    start_indices = np.where(gradient_sum > gradient_thresh)[0]
    if len(start_indices) == 0:
        return None 

    blank_block = []
    for i in range(len(start_indices)-1):
        if start_indices[i+1] - start_indices[i]>len_thresh:
            blank_block.append([start_indices[i]+smooth,start_indices[i+1]-smooth])
    
    ## add start 
    if start_indices[0]>len_thresh:
        if start_indices[0]-smooth>0:
            blank_block = [[0,start_indices[0]-smooth]]+blank_block 
        else:
            blank_block = [[0,start_indices[0]]]+blank_block 
    
    ## add end 
    if len(gradient_sum) - start_indices[-1]>len_thresh:
        if start_indices[-1]+smooth<len(gradient_sum):
            blank_block.append([start_indices[-1]+smooth,len(gradient_sum)])
        else:
            blank_block.append([start_indices[-1],len(gradient_sum)])
    
    if len(blank_block)==0:
        return None 

    return blank_block

def crop_image(image, direction='x'):
    global h,w 
    h, w  = image.shape[:2]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    global gradient_thresh
    gradient_thresh = 6000

    ## 计算图像的水平梯度
    gradient_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    gradient_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    gradient_x = np.abs(gradient_x)
    gradient_y = np.abs(gradient_y)
    gradient = np.maximum(gradient_x, gradient_y)
    gradient[gradient<50] = 0 
    gradient = cv2.resize(gradient, (2048,2048))

    if direction=='x':
        # 计算每一行的梯度总和
        gradient_sum = np.num(gradient,axis=1)

        # 找到上下裁剪边界
        blank_blocks = find_crop_boundaries(gradient_sum)

        # 如果找不到，则返回原始图像
        if blank_blocks is None:
            return image 
        
        # 裁剪图像
        cut_pixes = 0 
        cropped_image = image.copy()
        for blank_block in blank_blocks:
            cropped_image = np.concatenate((cropped_image[:blank_block[0]-cut_pixes,:],cropped_image[blank_block[1]-cut_pixes:,:]),0)
            cut_pixes+=blank_block[1]-blank_block[0]
        return cropped_image
    
    elif direction=='y':
        gradient_sum = np.sum(gradient, axis=0)

        blank_blocks = find_crop_boundaries(gradient_sum)

        if blank_blocks is None:
            return image 

        cut_pixes=0
        cropped_image = image.copy()
        for blank_block in blank_blocks:
            cropped_image = np.concatenate((cropped_image[:,:blank_block[0]-cut_pixes],cropped_image[:,blank_block[1]-cut_pixes:]),0)
            cut_pixes+=blank_block[1]-blank_block[0]
    
    elif direction=='xy':
        regions = np.zeros(gradient.shape)
        regions = cv2.cvtColor(regions.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
        gradient_sum = np.sum(gradient, axis=1)
        blank_blocks = find_crop_boundaries(gradient_sum)

        scale = h/2048
        if blank_blocks is not None:
            cut_pixes=0
            cropped_image = image.copy()
            for blank_block in blank_blocks:
                cropped_image = np.concatenate((cropped_image[:int((blank_block[0]-cut_pixes)*scale),:], cropped_image[int((blank_block[1]-cut_pixes)*scale):,:]),0)
                cut_pixes+=blank_block[1]-blank_block[0]
                regions[blank_block[0]:blank_block[1],:,:] = (240,175,0)
        else:
            cropped_image = image 

        gradient_sum = np.sum(gradient, axis=0)
        blank_blocks = find_crop_boundaries(gradient_sum)
        if blank_blocks is None:
            return cropped_image, gradient, regions

        cut_pixes = 0
        cropped_image = cropped_image.copy()
        scale = w/2048
        for blank_block in blank_blocks:
            cropped_image = np.concatenate((cropped_image[:,:int((blank_block[0]-cut_pixes)*scale)],cropped_image[:,int((blank_block[1]-cut_pixes)*scale):]),1)
            cut_pixes+=blank_block[1]-blank_block[0]
            regions[:,blank_block[0]:blank_block[1],:]=(240,176,0)

        return cropped_image, gradient, regions 







if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--im_path',type=str,default='demo/demo.jpg')
    parser.add_argument('--resize',action='store_true')
    parser.add_argument('--visualize',action='store_true')
    args = parser.parse_args()


    im = cv2.imread(args.im_path)
    h,w = im.shape[:2]
    cropped_image, gradient, regions = crop_image(im,'xy')


    if args.resize:
        new_h, new_w = cropped_image.shape[:2]
        if h>w and new_h>new_w or h<w and new_h<new_w:
            cropped_image = cv2.resize(cropped_image,(w,h))
        else:
            cropped_image = cv2.resize(cropped_image,(h,w))


    im_format = '.'+args.im_path.split('.')[-1]
    cv2.imwrite(args.im_path.replace(im_format,'_aps'+im_format),cropped_image)
    if args.visualize:
        gradient = cv2.resize(gradient,(w,h))
        regions = cv2.resize(regions,(w,h))
        regions = cv2.addWeighted(im,0.5,regions,0.5,gamma=0)
        cv2.imwrite(args.im_path.replace(im_format,'_visualize1'+im_format),gradient)
        cv2.imwrite(args.im_path.replace(im_format,'_visualize2'+im_format),regions)




