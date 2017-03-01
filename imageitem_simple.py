# -*- coding: utf-8 -*-
"""
Demonstrates very basic use of ImageItem to display image data inside a ViewBox.
"""
import PyQt5
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import cubehelix
import jet
import viridis

app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('pyqtgraph example: ImageItem')
view = win.addViewBox()

## lock the aspect ratio so pixels are always square
view.setAspectLocked(True)

## Create image item
img = pg.ImageItem(border='w')
view.addItem(img)

#img.setLookupTable(lut_cubehelix)

## Set initial view bounds
im_width=660
im_height=425
view.setRange(QtCore.QRectF(0, 0, im_width, im_height))

## Create random image
#data = np.random.normal(size=(15, 600, 600), loc=1024, scale=64).astype(np.float)
#data = np.random.normal(size=(15, 425, 330), loc=0.35, scale=0.55).astype(np.float)


data=np.zeros((im_width,im_height))

for i in range(im_height):
    for j in range(im_width):
        data[j,i]=np.sin(i*(2*np.pi/(im_height-1)))*np.cos(j*((2*np.pi/(im_width-1))))*0.55+0.35

i = 0

updateTime = ptime.time()
fps = 0

def updateData():
    global img, data, i, updateTime, fps

    ## Display the data
    #img.setImage(data[i],autoLevels=False, levels=[-0.2, 0.9], lut=cubehelix.cubehelix())
    #img.setImage(np.roll(data,i,axis=0),autoLevels=False, levels=[-0.2, 0.9], lut=cubehelix.cubehelix())
    #img.setImage(np.roll(data,i,axis=0),autoLevels=False, levels=[-0.2, 0.9], lut=jet.jet())
    img.setImage(np.roll(data,i,axis=0),autoLevels=False, levels=[-0.2, 0.9], lut=viridis.viridis())

    i = (i+1) % im_width

    QtCore.QTimer.singleShot(1, updateData)
    now = ptime.time()
    fps2 = 1.0 / (now-updateTime)
    updateTime = now
    fps = fps * 0.9 + fps2 * 0.1
    
    print(int(fps))
    

updateData()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()