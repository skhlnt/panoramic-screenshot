import os
import shutil
import cv2
from PIL import Image
import numpy as np
import time
import pyautogui
import tkinter as tk
from pynput import keyboard
import sys

# a component to stitch all the pictures
class picStitcher:
    def __init__(self, pathToPics, nfeatures = 500, draw = False):
        assert os.path.exists(pathToPics), "Path does not exist"
        self.pathToPics = pathToPics
        self.pathToRemoval = pathToPics+'tempRemoval/'
        if not os.path.exists(self.pathToRemoval):
            os.mkdir(self.pathToRemoval)
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures = nfeatures)
        self.bf = cv2.BFMatcher()
        self.draw = draw


    def match(self, i, j):
        print("start matching %d and %d" % (i, j))
        # preprocess
        pic1path, pic2path = self.pathToPics+str(i)+'.png', self.pathToPics+str(j)+'.png'
        assert os.path.exists(pic1path) and os.path.exists(pic2path), "Path does not exist for a pic in " + str(i) + 'or' + str(j)
        pic1raw, pic2raw = cv2.imread(pic1path), cv2.imread(pic2path)
        pic1, pic2 = cv2.cvtColor(pic1raw, cv2.COLOR_BGR2GRAY), cv2.cvtColor(pic2raw, cv2.COLOR_BGR2GRAY)
        # extract features
        kps1, des1 = self.sift.detectAndCompute(pic1, None)
        kps2, des2 = self.sift.detectAndCompute(pic2, None)
        matches = self.bf.knnMatch(des1, des2, k=2)
        good, tempgoods = [], []
        distances = {}
        for m, n in matches:
            if m.distance == 0:
                pos0 = kps1[m.queryIdx].pt
                pos1 = kps2[m.trainIdx].pt
                dist = (int(pos0[0]-pos1[0]), int(pos0[1]-pos1[1]))
                if dist in distances:
                    distances[dist] += 1
                else:
                    distances[dist] = 0
                tempgoods.append(m)
        sortedDistance = sorted(distances.items(), key=lambda dict: dict[1], reverse=True)
        # print(sortedDistance)
        try:
            distanceMode = sortedDistance[0][0]
            if sortedDistance[0][1] < 4: # not significantly matched
                print("%d and %d are not really matched" % (i, j))
                return -1
        except:
            print("%d and %d are not really matched" % (i, j))
            return -1

        # use the match that has minimal overlap
        # if(distanceMode[0] > 0):
        #     preferred_x0 = 0
        # else: 
        #     preferred_x0 = pic1.shape[0]
        # if(distanceMode[1] > 0):
        #     preferred_y0 = 0
        # else:
        #     preferred_y0 = pic1.shape[1]
        # preferred_x1 = preferred_x0
        # preferred_y1 = preferred_y0
        for match in tempgoods:
            pos0 = kps1[match.queryIdx].pt
            pos1 = kps2[match.trainIdx].pt
            if (int(pos0[0]-pos1[0]), int(pos0[1]-pos1[1])) == distanceMode:
                # if(distanceMode[0] > 0):
                #     if(pos0[0] > preferred_x0):
                #         preferred_x0 = pos0[0]
                #         preferred_x1 = pos1[0]
                # else:
                #     if(pos0[0] < preferred_x0):
                #         preferred_x0 = pos0[0]
                #         preferred_x1 = pos1[0]
                # if(distanceMode[1] > 0):
                #     if(pos0[1] > preferred_y0):
                #         preferred_y0 = pos0[1]
                #         preferred_y1 = pos1[1]
                # else:
                #     if(pos0[1] < preferred_y0):
                #         preferred_y0 = pos0[1]
                #         preferred_y1 = pos1[1]
                good.append(match)

        # print(len(good), distances, preferred_x0, preferred_x1, preferred_y0, preferred_y1)
        if self.draw:
            self.showSift(pic1, pic2, kps1, kps2, good)
        
        # stitch together and save
        if distanceMode == (0, 0):
            print("Duplicate!")
            shutil.move(pic1path, self.pathToRemoval + pic1path)
            return 1
        # the distanceMode is the difference between the 2 top left corners of the 2 pics
        # for different circumstances, find the top left corner coordinates of the 2 pics in final pic
        # the size of which is determined by the extremum of the 8 sides' position of the 2 pics
        if distanceMode[0] > 0:
            pic1TopLeftx = 0
            pic2TopLeftx = distanceMode[0]
            newPic_x = max(pic1raw.shape[1], pic2raw.shape[1] + distanceMode[0])
        else:
            pic1TopLeftx = -distanceMode[0]
            pic2TopLeftx = 0
            newPic_x = max(pic2raw.shape[1], pic1raw.shape[1] - distanceMode[0])
        if distanceMode[1] > 0:
            pic1TopLefty = 0
            pic2TopLefty = distanceMode[1]
            newPic_y = max(pic1raw.shape[0], pic2raw.shape[0] + distanceMode[1])
        else:
            pic1TopLefty = -distanceMode[1]
            pic2TopLefty = 0
            newPic_y = max(pic2raw.shape[0], pic1raw.shape[0] - distanceMode[1])
        newPic = np.zeros([newPic_y, newPic_x, 3])
        # print(distanceMode)
        newPic[pic1TopLefty:pic1TopLefty+pic1raw.shape[0], pic1TopLeftx:pic1TopLeftx+pic1raw.shape[1], :] = pic1raw
        newPic[pic2TopLefty:pic2TopLefty+pic2raw.shape[0], pic2TopLeftx:pic2TopLeftx+pic2raw.shape[1], :] = pic2raw
        shutil.move(pic1path, self.pathToRemoval + str(i) + '.png')
        # os.remove(pic1path)
        shutil.move(pic2path, self.pathToRemoval + str(j) + '.png')
        cv2.imwrite(pic1path, newPic)
        return distanceMode

    def showSift(self, pic1, pic2, kps1, kps2, good):
        pic3 = cv2.drawMatches(pic1, kps1, pic2, kps2, good, None, flags = 2)
        img_sift1 = np.zeros(pic1.shape, np.uint8)
        img_sift2 = np.zeros(pic2.shape, np.uint8)
        cv2.drawKeypoints(pic1, kps1, img_sift1)
        cv2.drawKeypoints(pic2, kps2, img_sift2)
        cv2.imwrite(pathToPics+"showingSiftResults.png", pic3)
        print("write a frame")

    def integrate(self):
        # get all the pics names in the current path and pick those with integer names
        validSuffices = [".png", ".PNG"]
        filesInPath = os.listdir(pathToPics)
        pics = sorted([int(f[:-4]) for f in filesInPath if f[-4:] in validSuffices and f[:-4].isnumeric()])
        print(pics)
        if(len(pics) < 2):
            print("No enough pics to put together")
            return 0
        pics_temp = pics
        while len(pics) > 1:
            self.match(pics[0], pics[1])
            filesInPath = os.listdir(pathToPics)
            pics = sorted([int(f[:-4]) for f in filesInPath if f[-4:] in validSuffices and f[:-4].isnumeric()])
            print(pics_temp, pics)
            if(pics_temp == pics):
                print("cannot find matching here")
                break
            pics_temp = pics
        return 0


def openScreenshoot():
    window = tk.Tk()
    window.title("Screenshooter")
    window.geometry("300x50")
    window.wm_attributes('-topmost', 1)

    def startArea():
        global flag
        flag = 1
        window.destroy()

    b = tk.Button(window, text="Start", command=startArea)
    b.pack()
    window.mainloop()
    return flag

def setArea():
    root = tk.Tk()

    def sys_out(event):
        root.destroy()
    def button_1(event):
        global x, y, xstart, ystart
        x, y = event.x, event.y
        xstart, ystart = event.x, event.y
        # print("event.x, event.y = ", event.x, event.y)
        cv.configure(height=1)
        cv.configure(width=1)
        cv.place(x=event.x, y=event.y)
    def b1_Motion(event):
        global x, y
        x, y = event.x, event.y
        # print("event.x, event.y = ", event.x, event.y)
        cv.configure(height=event.y - ystart)
        cv.configure(width=event.x - xstart)
    def buttonRelease_1(event):
        global x, y
        x, y = event.x, event.y
        # print("xs, ys = ", xstart, ystart)
        # print("event.x, event.y = ", event.x, event.y)
        cv.configure(height=event.y - ystart)
        cv.configure(width=event.x - xstart)
        # cv.place_forget()
        # img = pyautogui.screenshot(region=[xstart, ystart, x - xstart, y - ystart])  # x,y,w,h
        # img.save('screenshot.png')

    root.bind("<Button-3>", sys_out)
    root.bind("<Return>", sys_out) # enter to set area
    root.bind("<Button-1>", button_1)
    root.bind("<B1-Motion>", b1_Motion)
    root.bind("<ButtonRelease-1>", buttonRelease_1)

    # this window stays at the top of all windows
    root.wm_attributes('-topmost', 1)
    root.overrideredirect(True)  # hide title
    root.attributes("-alpha", 0.2)  
    root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    root.configure(bg="blue")
    # canvas to select area
    cv = tk.Canvas(root)
    root.mainloop()
    
    return xstart, ystart, x, y

def mainProcess(xstart, ystart, xend, yend):
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.overrideredirect(True)
    window.attributes("-alpha", 0.2)
    window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), ystart)) 
    window.configure(bg="blue")
    def sys_out(event):
        window.destroy()
    def screenShot(event):
        global counter
        if not os.path.exists(pathToPics):
            os.mkdir(pathToPics)
        img = pyautogui.screenshot(region=[xstart, ystart, xend - xstart, yend - ystart])
        img.save(pathToPics+str(counter)+".png")
        counter += 1
        print("Catched "+str(counter)+'.png')
    window.bind("<Button-3>", sys_out)
    window.bind("<Return>", sys_out)
    window.bind("<Button-1>", screenShot)
    window.bind("<space>", screenShot)

    def openNew(x1, y1, x2, y2):
        window2 = tk.Toplevel()
        window2.wm_attributes('-topmost', 1)
        window2.overrideredirect(True)
        window2.attributes("-alpha", 0.2)
        window2.geometry("{0}x{1}+{2}+{3}".format(x2-x1, y2-y1, x1, y1))
        window2.configure(bg="blue")
        window2.bind("<Button-3>", sys_out)
        window2.bind("<Return>", sys_out)
        window2.bind("<Button-1>", screenShot)
        window2.bind("<space>", screenShot)
    window.after(1, openNew(0, ystart, xstart, yend))
    window.after(1, openNew(0, yend, window.winfo_screenwidth(), window.winfo_screenheight()))
    window.after(1, openNew(xend, ystart, window.winfo_screenwidth(), yend))
    window.mainloop()
    return 0

def finalCheck():
    window = tk.Tk()
    window.geometry("500x100+0+0")
    window.title("Default is SAVE")
    def save():
        global flag
        flag = 2
        window.destroy()
        print("images saved at "+pathToPics)
    def cancel():
        if os.path.exists(pathToPics):
            shutil.rmtree(pathToPics)
        window.destroy()
        print("already removed "+pathToPics)
    bSave = tk.Button(window, text="Save and make panorama", command=save)
    bCancel = tk.Button(window, text="Cancel and delete screenshots", command=cancel)
    bSave.pack()
    bCancel.pack()
    window.mainloop()
    return 0
        



pathToPics = "./tempPics/"

if __name__ == "__main__":
    global flag
    global counter
    flag = 0
    counter = 0
    openScreenshoot()
    if flag == 1:
        xstart, ystart, xend, yend = setArea()
        print(xstart, ystart, xend, yend)
        mainProcess(xstart, ystart, xend, yend)
        finalCheck()
        if flag == 2: # choose to save and make full pic
            newPicStitcher = picStitcher(pathToPics, draw=False)
            newPicStitcher.integrate()
