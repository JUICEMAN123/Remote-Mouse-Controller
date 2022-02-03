import cv2 as cv
import numpy as np
import pyautogui as pyag


def main():
    id_color = [252, 252, 252]
    deviation = 5
    
    url = 'https://192.168.11.50:8080/' + 'video'
    cap = cv.VideoCapture(url)
    
    x = 0
    y = 0
    w = 0
    h = 0
    
    while(True):
        ret, frame = cap.read()
        frame = frame[0:len(frame)][300:1000]
        frame = rescale_frame(frame, 0.5)
        frame_filtered = filter_using_color(frame, id_color, deviation)
        frame_blur = cv.GaussianBlur(frame_filtered, (9,9), cv.BORDER_DEFAULT)
        frame_canny = cv.Canny(frame_blur, 30, 200, 1)
        cnts = cv.findContours(frame_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        # Iterate thorugh contours and draw rectangles around contours
        for c in cnts:
            x,y,w,h = cv.boundingRect(c)
   
        cv.rectangle(frame, (x, y), (x + w, y + h), (255,0,255), 1)
        cv.circle(frame, (x + w//2, y + h//2), 5, (255, 0, 255), cv.FILLED)
            
        cv.imshow('frame', frame)
        cv.imshow('framef', frame_filtered)
        cv.imshow('framec', frame_canny)
        
        print(w, h)
        # 130,80
        # 80, 55
        # 55, 40
        interpolated_X = (w-130) * -24
        interpolated_Y = (x-(w*-1.25+162.5))/(len(frame)-(w*-2.5+325)) * 1080
        coords = (interpolated_X, interpolated_Y)
        print(coords)
        
        pyag.moveTo(coords, _pause = False)
        
        
        key = cv.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("c"):
            id_color[0] = int(input('R: '))
            id_color[1] = int(input('G: '))
            id_color[2] = int(input('B: '))
        if key == ord("i"):
            id_color[0]-=1
            print(id_color)
        if key == ord("o"):
            id_color[1]-=1
            print(id_color)
        if key == ord("p"):
            id_color[2]-=1
            print(id_color)
        if key == ord("j"):
            id_color[0]+=1
            print(id_color)
        if key == ord("k"):
            id_color[1]+=1
            print(id_color)
        if key == ord("l"):
            id_color[2]+=1
            print(id_color)
    
    cap.release()
    cv.destroyAllWindows()

def filter_using_color(frame, color, deviation):
    lower, upper = gen_bounds(color, deviation)
    lower = np.array(lower, dtype = 'uint8')
    upper = np.array(upper, dtype = 'uint8')
    mask = cv.inRange(frame, lower, upper)
    output = cv.bitwise_and(frame, frame, mask = mask)
    return output

def rescale_frame(frame, scale = 1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    newDimensions = (width, height)
    return cv.resize(frame, newDimensions, interpolation=cv.INTER_AREA)

def gen_bounds(color, deviation):
    lower = []
    upper = []
    for val in color:
        lower.append(constrain_value(val - deviation, 0, 255))
        upper.append(constrain_value(val + deviation, 0, 255))
    return (lower, upper)

def constrain_value(val, bound1, bound2):
    if bound2 < bound1:
        tmp = bound2
        bound2 = bound1
        bound1 = tmp
    if val < bound1:
        return bound1
    elif val > bound2:
        return bound2
    else:
        return val

if __name__ == '__main__':
    main()