import cv2 as cv
import mediapipe as mp
import time
import pyautogui as pyag
import numpy as np
from collections import deque

def moveMouse(x, y):
    pyag.moveTo(x, y, duration = 0.05, tween=(pyag.easeInQuad), _pause=False)
    
def dragMouse(x, y):
    pyag.dragTo(x, y, duration = 0.05, tween=(pyag.easeInQuad), _pause=False)

def isBetween(num, bound1, bound2):
    if bound2 < bound1:
        tmp = bound2
        bound2 = bound1
        bound1 = tmp
    return num > bound1 and num < bound2

def dynamics(coords, previousDistances, x, y, clickTracker):
    previousDistances.appendleft(distance(coords[4], coords[8]))
    previousDistances.pop()
    average = sum(previousDistances)/len(previousDistances)
    clickTracker = clickDynamic(average < 0.05, clickTracker)
    moveMouse(x, y)
    if (average < 0.04): # dragging
        print('dragging', average)
    else: # moving
        print('moving', average)
    return clickTracker
    
def clickDynamic(isDown, clickTracker):
    if isDown:
        if not clickTracker:
            pyag.mouseDown(_pause = False)
            clickTracker = True
    else:
        pyag.mouseUp(_pause = False)
        clickTracker = False
    return clickTracker

def isFist(coords):
    return standardDeviation(coords) < 0.055

def isExit(coords):
    ys = [coord[1] for coord in coords]
    std = np.std(ys)
    return std < 0.02

def isWindowSwitch(coords):
    xs = [coord[0] for coord in coords]
    std = np.std(xs)
    return std < 0.0175
    
    
def averageCoord(coords):
    coordVals = [0] * len(coords[0])
    for coord in coords:
        for i, val in enumerate(coord):
            coordVals[i] += val
    for i in range(len(coordVals)):
        coordVals[i] /= len(coords)
    return tuple(coordVals)

def standardDeviation(coords):
    avgCoord = averageCoord(coords)
    standardDeviation = 0
    for coord in coords:
        standardDeviation += (coord[0] - avgCoord[0]) ** 2
        standardDeviation += (coord[1] - avgCoord[1]) ** 2
        standardDeviation += (coord[2] - avgCoord[2]) ** 2
    standardDeviation /= len(coords)
    standardDeviation = standardDeviation ** 0.5
    return standardDeviation

def distance(point1, point2):
    if len(point1) == len(point2):
        distance = 0
        for i in range(len(point1)):
            distance += (point1[i] - point2[i]) ** 2
        return distance ** 0.5
    else:
        return -1

def drawLineBetween(img, wFrame, hFrame, coords, p1, p2):
    img = cv.circle(img, (int(coords[p1][0] * wFrame), int(coords[p1][1] * hFrame)), 7, (100, 100, 255), cv.FILLED)
    img = cv.circle(img, (int(coords[p2][0] * wFrame), int(coords[p2][1] * hFrame)), 7, (100, 100, 255), cv.FILLED)
    img = cv.circle(img, (int((coords[p1][0] + coords[p2][0]) / 2 * wFrame), 
                          int((coords[p1][1] + coords[p2][1]) / 2 * hFrame)), 7, (100, 100, 255), cv.FILLED)
    img = cv.line(img, 
                  (int(coords[p1][0] * wFrame), int(coords[p1][1] * hFrame)), 
                  (int(coords[p2][0] * wFrame), int(coords[p2][1] * hFrame)), 
                  (100,100,255),
                  thickness = 5)
    return img

def HandController():
    capture = cv.VideoCapture(0)    
    
    hands = mp.solutions.hands.Hands(max_num_hands = 1, min_detection_confidence=0.7, min_tracking_confidence=0.7) # can change num of hands, and detection sensistivity later
    
    moveSmoothening = 5
    FR = 100
    
    clickTracker = False
    
    dragging_confidence = 5
    
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    
    pTime = 0
    cTime = 0
    
    windowSwitchDelay = 0.75
    windowPtime = time.time()
    isSwitching = False
    
    # xSens = 1.3
    # ySens = 1.3
    # zSens = 3
    
    hScreen, wScreen = 1080, 1920
    hFrame, wFrame = 0, 0
    
    scrollSpeed = 50
    
    pyag.FAILSAFE = False # RISKY BUT WORKS FOR NOW
    
    active = True
    
    previousDistances = deque([1] * dragging_confidence)
    
    while active:
        success, img = capture.read()
        hFrame, wFrame, _ = img.shape
        
        img = cv.flip(img, 1)
        
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
    
        controlLocation = (0, 0, 0)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)
                coords = []
                for id, lm in enumerate(handLms.landmark):
                    coords.append((lm.x, lm.y, lm.z))
                controlLocation = coords[9]
            
            img = cv.circle(img, (int(controlLocation[0] * wFrame), int(controlLocation[1] * hFrame)), 15, (255, 0, 255), cv.FILLED)
            img = cv.rectangle(img, (FR, FR), (wFrame-FR, hFrame-FR), (255,0,255))
            img = drawLineBetween(img, wFrame, hFrame, coords, 4, 8)

            # move
            x1, y1, z1 = controlLocation
            x_conv = np.interp(x1, (FR/wFrame, 1 - FR/wFrame), (0, wScreen))
            y_conv = np.interp(y1, (FR/hFrame, 1 - FR/hFrame), (0, hScreen))
            clocX = plocX + (x_conv - plocX) / moveSmoothening
            clocY = plocY + (y_conv - plocY) / moveSmoothening
            
            clickTracker = dynamics(coords, previousDistances, clocX, clocY, clickTracker)
            
            plocX = clocX
            plocY = clocY
            
            # scroll
            if isFist(coords):
                scrollAmount = int(np.interp(y1, (FR/hFrame, 1 - FR/hFrame), (scrollSpeed, -scrollSpeed)))
                pyag.scroll(scrollAmount, _pause = False)
            
            if isWindowSwitch(coords):
                if(isSwitching):
                    if(time.time() - windowPtime > windowSwitchDelay):
                        windowPtime = time.time()
                        pyag.press('right', _pause = False)
                else:
                    isSwitching = True
                    pyag.keyDown('alt', _pause=False)
                    pyag.keyDown('tab', _pause=False)
            else:
                isSwitching = False
                pyag.keyUp('tab', _pause=False)
                pyag.keyUp('alt', _pause=False)
                    
                    
            
            # exit
            if isExit(coords):
                active = False
                
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        cv.putText(img, str(int(fps)), (30, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv.imshow('Webcam', img)
        
        if cv.waitKey(10) and 0xFF == ord('d'):  
            active = False
        
    capture.release()
    cv.destroyAllWindows()

HandController()