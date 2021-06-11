#!/usr/bin/env python
"""

  Initially identical with oldEdgar/edgar7a.py.  This version abandons
  inpaint library, due to ugly artifacts in plots.  

  Now handles complex interferogram input values.  This entails:
    When "sim" data is recognized as complex, set dataType = 'C' (vs 'F')
    Early image plots are created from phase of complex vals for 'C'
    Edge plots should be unchanged.
    Depth-slip values and plots arise from modified profile methods for 'C'

  DESCRIPTION:
  edgar.py takes a set of simplex input lines fron InSAR to edge detection data
  using the cv (computer vision) library "Canny"
  Trial version replacing linear interp in holeInterp with repeated gaussian
  smoothing  (each iter smoothes entire image, then restores good pixels to
    makeGoogEarthFile(pngName, limit, post)
  previous values).

  USAGE:

     edgar.py [-a <aperture> -m <threshold in mm> 
           -r <threshRatio> -s <smoothingSigmaPixels> 
           <input> <index> <summary>  

aperture must be 3, 5, or 7 (default is 5)
threshold sets smallest apparent fault slip that will be detected
   (larger values may miss interesting slip, smaller values add clutter)
   (default is 2 mm)
threshRatio sets ratio of low to high thresholds (default 0.75)
smoothingSigmaPixels sets width (in UAVSAR original pixels) of smoothing sigma
   (default is 1.0, minimal smoothing)

Introducing new methods to use unw data image to trim pixels on 
valid-data peninsulas (penTrim) and to perform bilinear 
interpolation in holes (holeInterp).

  FILES:
     dataFile format of simplex input data, eg.

type x         y (km)    LOS_displ unc    elev_ang   azimuth (deg)
7 -14.181635 -6.442968 -8.313177 1.000000 67.704704 -5.319699

     index lines has the location of the corresponding simplex line
     in indices on a rectangular grid.
     index lines must have same number of lines as simplex lines:

idx  idy
293 61

     summary has keyword-values pairs indicating original averaging cell
     (NlonPix 3 means sarSample.pyc -m 3 -n 3 was used prior)
     range idx, idy (indices of pixels in the original full UAVSAR strip,
     giving bounds of the rectangular region superscribing the cutting
     polygon (indicated in sarSample.pyc command line as -p <poly>.kml)
     and now path to data (Simplex input - observation format)
     and index file (produced by sarSample.pyc or sar2simplex.py
     used with -e option).

     Note first five lines of summary are produced by sarSample.pyc with
     the -e option, and put in file isum.txt

NlonPix 3
MinIx 15
MaxIx 295
MinIy 61
MaxIy 95
LonMin -116.086008
LatMin 32.931801
LonDelta 0.000056
LatDelta -0.000056
Nlon 20521
Nlat 6306

Currently produces image file edgemap.png and intermediate images 
(logmag.png shows the raw gradient data used by Canny, for example).
"""

import sys
import getopt
from math import * 
import numpy as np
import scipy.ndimage as nd
import cv2
import copy
import geofunc
import time
import zipfile

emptyFlag = 1.e20

# pCal defined to be (from repeated edgar0 runs) (-m value)/(ideal jump)
# where -m value is value that barely detects the jump in simulated (disloc-
# generated) NS.txt  input file
#pCal = 29.53
# Attempting correction for actual NS 4.3 slip
pCal = 27.4697

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

class DataType():
   def __init__(self,dType):
      if dType == 'C':
          self.dataType = 'C'
          self.emptyFlag = 1e20+0j
          self.emptyTest = 1e20
      elif dType == 'F':
          self.dataType = 'F' 
          self.emptyFlag = 1e20
          self.emptyTest = 1e20
      
def image2outs( image,rootName,limit,doScale,iPct=0.,doTrans=False,transPct=2):
   """
   Issues: not sure works on both np array and cv2 array
   Usage:
      image,iScale,iShift,iPct = image2outs(image,rootName,limit,doScale=,iPct,\
           doTrans=,transPct=)
   Args:
      image: numpy array of values (supporting uint8,float,complex?)
         if type is complex, do not scale; products from raw phase
         if type is otherwise, scale if doScale == True
         type can be complex, float, or rgba vector
      rootName: initial part of filenames, unique to each call
      so files not overwritten; eg., 'trim'
      doScale: boolean, True means do internal rescaling to 0:255
      iPct: optionl; if supplied, scale to this lower, upper percentile (reduces
         scaing problems from outliers)
      doTrans: optional, create transparent background
      transPct: optional, pixels below this percent of max become transparent
   Returns:
      image: original image, or (if doScale) scaled image
      iScale: 1, or (if doScale) scale factor
      iShift: 0, or (if doScale) shift used to rescale to 0:255
   """
   if image.size < 1: return image,1,0
   if len(image.shape) == 3: # this is rgba vector type
      print(("Running image2outs rgba vector with ",rootName))
      makeGoogEarthFile(image, limit, rootName+'V')
      return image,1.,0.
   if isinstance(image[0,0],complex):
      # complex: do not scale, but create outputs from direct (scaled) phase
      print(("Running image2outs complex with ",rootName))
      imagePhase = np.arctan2(image.imag,image.real)
      i2txt(image,rootName+'C')
      i2txt(imagePhase,rootName+'P')
      makeGoogEarthFile(imagePhase, limit, rootName+'P')
      return image,1.,0.
   if isinstance(image[0,0], float) or isinstance(image[0,0], np.uint8) \
       or isinstance(image[0,0], np.int64):
      if not doScale: 
         print(("Running image2outs float noScale with ",rootName))
         # Create unscaled outputs, return array
         i2txt(image,rootName+'NS')
            
         makeGoogEarthFile(image, limit, rootName+'NS')
         return image,1., 0.
      else:
         print(("Running image2outs float doScale with ",rootName))
         # find cMin, cMax for both cases; clip in iPct case
         if iPct  == 0.:
            cMin,cMax = image.min(),image.max()
         else:
            imageAlt = copy.deepcopy(image).reshape(-1)
            imageAlt.sort()
            cMin = imageAlt[int(len(imageAlt)*iPct/100.)]
            cMax = imageAlt[int(len(imageAlt)*(100-iPct)/100.)]
            image=np.clip(image,cMin,cMax)

         # Apply the scaling
         cScale = 255./(cMax-cMin)
         cShift = -cScale*cMin
         image = image*cScale
         image = image + cShift
         # Create scaled outputs
         i2txt(image,rootName+'DS')
         image=cv2.convertScaleAbs(image.astype(int))
         if doTrans:
            transLevel = image.max()*(transPct/100.)
            print(('DBG transLevel,image.max(): ',transLevel,image.max()))
            alpha = image > transLevel # a bool array 
            alpha = np.uint8(255*alpha)
            rgba = (image,image,image,alpha)
            imTrans = cv2.merge(rgba)
            makeGoogEarthFile(imTrans, limit, rootName+'DS')
         else:
            makeGoogEarthFile(image, limit, rootName+'DS')
         return image,cScale,cShift
     
def cCanny(image, threshold,phase2mm,scaling,limit,dT):
   """
   Similar to cv2.Canny, except only args are
   image: the BW image in numpy array format (check)
   threshold: the "high" threshold, which edges must exceed
   phase2mm: converts phase to mm
   scaling: any additional scaling for calibration, normalization

   Later might include 
   "low" threshold, 
   aperture (3 here), 
   L2gradient flag (as in Canny)
   """
   val2mm =  10./0.527 # based on 0.527 produced by edgarCal for 10mm step

   #Create positive (abs) valued scharr, and combine with proper signs
   # through  conjugation in LRphase, UDphase.
   ScharrL = np.array([[3., 0., 0.],[10., 0., 0.], [3., 0., 0.]])
   ScharrR = np.array([[0., 0., 3.],[0., 0., 10.], [0., 0., 3.]])
   ScharrD = np.array([[ 3., 10., 3.],[0., 0., 0.], [0., 0., 0.]])
   ScharrU = np.array([[0., 0., 0.], [0., 0., 0.],[3.,10., 3]])
   SLGr = nd.convolve(image.real,ScharrL)
   SLGi = nd.convolve(image.imag,ScharrL)
   SRGr = nd.convolve(image.real,ScharrR)
   SRGi = nd.convolve(image.imag,ScharrR)
   SUGr = nd.convolve(image.real,ScharrU)
   SUGi = nd.convolve(image.imag,ScharrU)
   SDGr = nd.convolve(image.real,ScharrD)
   SDGi = nd.convolve(image.imag,ScharrD)
   SLG = SLGr + SLGi*(0+1j)
   SRG = SRGr + SRGi*(0+1j)
   SUG = SUGr + SUGi*(0+1j)
   SDG = SDGr + SDGi*(0+1j)
   LRphase = np.angle(SLG*SRG.conj())*val2mm
   UDphase = np.angle(SDG*SUG.conj())*val2mm
   LRphaseNorm = (LRphase*LRphase.conjugate()).real
   UDphaseNorm = (UDphase*UDphase.conjugate()).real
   imageJun,scJun,scSJun = image2outs(LRphase,'LRphase',limit,False)
   imageJun,scJun,scSJun = image2outs(LRphaseNorm,'LRphaseNorm',limit,True)
   imageJun,scJun,scSJun = image2outs(UDphase,'UDphase',limit,False)
   imageJun,scJun,scSJun = image2outs(UDphaseNorm,'UDphaseNorm',limit,True)
   Gmag = np.hypot(LRphase, UDphase)
   GmagNorm = (Gmag*Gmag.conjugate()).real
   Gdir = np.arctan2(UDphase, LRphase)*180./np.pi # Dir ccw from x
   Gdir[np.where(Gdir < 0.)] += 360.
   Gdir[np.where(Gdir >360.)] -= 360.
   imageJun,scJun,scSJun = image2outs(Gmag,'Gmag',limit,True,iPct = 1,doTrans=True,transPct = 40)
   imageJun,scJun,scSJun = image2outs(GmagNorm,'GmagNorm',limit,True,doTrans=True)

   # Round to four directions to make along-gradient maxima simple
   Gdir[np.where(np.logical_and(0. <= Gdir,Gdir < 22.5))] = 0.
   Gdir[np.where(np.logical_and(157.5 <= Gdir,Gdir < 202.5))] = 0.
   Gdir[np.where(np.logical_and(337.5 <= Gdir,Gdir < 360.))] = 0.

   Gdir[np.where(np.logical_and(22.5 <= Gdir,Gdir < 67.5))] = 45.
   Gdir[np.where(np.logical_and(202.5 <= Gdir,Gdir < 247.5))] = 45.

   Gdir[np.where(np.logical_and(67.5 <= Gdir,Gdir < 112.5))] = 90.
   Gdir[np.where(np.logical_and(247.5 <= Gdir,Gdir < 292.5))] = 90.

   Gdir[np.where(np.logical_and(112.5 <= Gdir,Gdir < 157.5))] = 135.
   Gdir[np.where(np.logical_and(292.5 <= Gdir,Gdir < 337.5))] = 135.

   # Mark all directional non-maxima with zeros in mag:
   # Consider only N-2 center so operaitons stay in bounds.
   # Create convenient shifted array names (should take no space)
   h,w = image.shape
   GmCtr = Gmag[1:-1,1:-1]
   GdCtr = Gdir[1:-1,1:-1]
   GmL = Gmag[1:-1,2:]
   GmR = Gmag[1:-1,:-2]
   GmU = Gmag[2:,1:-1]
   GmD = Gmag[:-2,1:-1]
   GmSE =Gmag[:-2, 2:]
   GmNW = Gmag[2:, :-2]
   GmNE = Gmag[2:, 2:] 
   GmSW = Gmag[:-2, :-2]
   
   GmCtr[np.where(np.logical_and(GdCtr == 0., 
       np.logical_or(GmCtr < GmL, GmCtr < GmR)))] = 0. 
   GmCtr[np.where(np.logical_and(GdCtr == 45., 
       np.logical_or(GmCtr < GmSE, GmCtr < GmNW)))] = 0. 
   GmCtr[np.where(np.logical_and(GdCtr == 90., 
       np.logical_or(GmCtr < GmU, GmCtr < GmD)))] = 0. 
   GmCtr[np.where(np.logical_and(GdCtr == 135., 
       np.logical_or(GmCtr < GmNE, GmCtr < GmSW)))] = 0. 
   
   # for remaining magnitude records, prune all below threshold
   # (requires calibrating threshold correctly)
   Gmag[0,:] = 0
   Gmag[-1,:] = 0
   Gmag[:,0] = 0
   Gmag[:,-1] = 0
   Gmag[np.where(Gmag < threshold)] = 0
   Gmag[np.where(Gmag >= threshold)] = 1

   return Gmag.astype(int)

class DepthSlip():
   """ 
       JT_loopParmas contains the params for a JT_loop
       widMin,
       widMax,
       minTop,
       maxTop,
       TPercentile,
       minSlip,
       maxSlip,
    """
   def __init__(self, widMin,widMax,minTop,maxTop,TPercentile,minSlip,maxSlip,
      colorMin,colorMax,labelN):
      self.widMin = widMin
      self.widMax = widMax
      self.minTop = minTop
      self.maxTop = maxTop
      self.TPercentile = TPercentile
      self.minSlip = minSlip
      self.maxSlip = maxSlip
      self.colorMin = colorMin
      self.colorMax = colorMax
      self.labelN= labelN

   def depthSlipPlots(self,jtList,slipList,slateShape,limit,elevAngle,dT):
      """
      depthSlipPlots estimates the depth to top and slip estimates and produces
      related map products.
      
      SlipList: a list of slip values.
      
      If self.minSlip, maxSlip == 0,0 overwrite with 0, NN percentiles
      
      Draw variable-thickness "lines" from points,
      thickness varies with topDepth, hot color with slip amount
      new: skip dispaly of point edgePix if inValid

      """
      blankSlate = np.zeros(slateShape, np.uint8)
      if len(slipList) != 0:
         minTop,maxTop = self.minTop,self.maxTop

      if self.minSlip == 0. and self.maxSlip == 0.:
         # The slip histogram tends to long thin tails, and
         # there are tiny values that don't appear meaningful.
         # So use index 0 to  NNth percentile
         idxNN = int(self.TPercentile*0.01*len(jtList) )
         minSlip = jtList[0].slip
         maxSlip = jtList[idxNN].slip
      else:
         minSlip = self.minSlip
         maxSlip = self.maxSlip
        
      if(maxSlip == minSlip):
         raise ValueError(\
            'Slip Range len %d min %f max %f pixel %d %d,w %f s %f pct %d'\
            % (len(slipList),minSlip,maxSlip,px.eRow,px.eCol,
            px.slip,px.topDepth,self.TPercentile))
      labelN = self.labelN
      jtThisRun = []
      jtf = open("jumpTable"+labelN+".txt","w")
      for px in jtList:
         sL = px.slip
         tD = px.topDepth
         # Clip to whatever range has been set
         if tD < minTop: tD = minTop
         if tD > maxTop: tD = maxTop
         #if sL < minSlip: sL = minSlip
         if sL > maxSlip: sL = maxSlip
         if sL < minSlip: 
            continue


         fracSlip = (sL - minSlip)/(maxSlip-minSlip)
         lineColor = fracSlip*self.colorMax + (1.-fracSlip)*self.colorMin
         lineColor = int(lineColor) + 1
         fracWidth = (tD - minTop)/(maxTop-minTop)
         lineWidth = fracWidth * self.widMax + (1.-fracWidth)*self.widMin
         lineWidth = int(lineWidth) + 1
         # circle arg "-1" indicates filled circles
         cirCenter = (px.eCol,px.eRow)

         cv2.circle(blankSlate,cirCenter,lineWidth,lineColor,-1)

         jtThisRun.append(px)
         # and write fracture pixel params out to file:
         # recalling row index movies in y direction, col in x.
         if px.slip > 0.1: # avoid zero-slip cases
            jtf.write('%f %f %f %f %f %f %f %f %f %f %f\n'%(px.x,px.y,
                   px.gradDir, px.slip,px.slipCaliper,
                   px.topDepth,elevAngle[px.eRow,px.eCol],px.eCol,px.eRow,
                   px.lonPoint,px.latPoint))
        
      blankSlate,iScale,sh = image2outs(blankSlate,'BlankSlate'+labelN,limit,False)

      # Color faults according to slip (already BW by lineColor, above).
      imColor = cv2.applyColorMap(blankSlate,cv2.COLORMAP_HOT)
      rootName = 'BScolor'+labelN
      image2outs(imColor,rootName,limit,False)

      # make background transparent:
      alpha = blankSlate < 0.1 # really 0 on a 255 scale
      alpha = np.uint8(alpha)*255 # background is now 0 (black)
      atmp = np.logical_not(alpha)
      abar = np.uint8(atmp)
      abar = abar*255 # background is here 255 (white, or "a" transparent); 
      rgba = (imColor[:,:,0],imColor[:,:,1],imColor[:,:,2],abar)
      imTrans = cv2.merge(rgba)
      rootName = 'TSfracs' + labelN
      image2outs(imTrans,rootName,limit,False)

      jtf.close()
   
   def sleevedSlipPlot(self,image,jtList,slipList,slateShape,limit,dT):
      """
      image: non-dilated white-on-black slip image, used for shape
      sleevedSlipPlot plots slip estimates by heat color on uniforn 3pt line
      encasedin 5pt black sleeve.
         
      jtList: basic pixel data: eRow,eCol,gradDir,slip,. . .
      slipList: aligned (with jtList) lists of just the slips
      slateShape: the image dimensions; not used here, use image as template
      limit: geometric information allowing registration of kml
      """
      labelN = self.labelN
      if self.minSlip == 0. and self.maxSlip == 0.:
         # The slip histogram tends to long thin tails, and
         # there are tiny values that don't appear meaningful.
         # So use index 0 to  NNth percentile
         idxNN = int(self.TPercentile*0.01*len(jtList) )
         minSlip = jtList[0].slip
         maxSlip = jtList[idxNN].slip
      else:
         minSlip = self.minSlip
         maxSlip = self.maxSlip

      # Create sleeveImage that included black dots where slip >= minSlip
      sleeveImage = np.ones(image.shape,np.uint8)*255 # filled with 255:white 
      for px in jtList:
         if px.slip < minSlip:
            continue
         sL = px.slip
         fracSlip = (sL - minSlip)/(maxSlip-minSlip)
         lineColor = 0
         #lineWidth = 7 #check if this is a reasonable thickness by doing
         lineWidth = 1 #check if this is a reasonable thickness by doing
         cirCenter = (px.eCol,px.eRow)
         # circle arg "-1" indicates filled circles
         ###cv2.circle(sleeveImage,cirCenter,lineWidth,lineColor,-1)

      # alphaOpac: where 0 make image transparent
      alphaOpac = np.uint8(sleeveImage < 128)*255 # T:background; now 255
      #So we have nkern-wide black sleeves, and an opacity mask.  
      # Need colored data.
      for px in jtList:
         sL = px.slip
         #if sL < minSlip: sL = minSlip
         if sL > maxSlip: sL = maxSlip
         if sL < minSlip: 
            continue

         fracSlip = (sL - minSlip)/(maxSlip-minSlip)
         lineColor = fracSlip*self.colorMax + (1.-fracSlip)*self.colorMin
         ###lineColor = int(lineColor) + 1
         lineColor = int(lineColor)-1
         #lineWidth = 4
         lineWidth = 1
         # circle arg "-1" indicates filled circles
         cirCenter = (px.eCol,px.eRow)
         cv2.circle(sleeveImage,cirCenter,lineWidth,lineColor,-1)
      
      ### Moved and modified for no-black case.  Remove for normal operation
      # alphaOpac: where 0 make image transparent
      alphaOpac = np.uint8(sleeveImage < 255)*255 # T:background; now 255
      ###
      # Color those pixels (the last set of dots) according to
      # their grayscale values; use these values to  make HOT.
      imColor = cv2.applyColorMap(sleeveImage,cv2.COLORMAP_HOT)

      # make background transparent:
      rgba = (imColor[:,:,0],imColor[:,:,1],imColor[:,:,2],alphaOpac)
      imTrans = cv2.merge(rgba)
      rootName = 'SlSlip' + labelN
      imageIn,scPatch,scShift = image2outs(imTrans,rootName,limit,True)

   def tdGrayPlot(self,image,jtList,slipList,slateShape,
           limit,dT):
      """
      tdGrayPlot produces gray dots proportional to fault depth
      image: non-dilated white-on-black slip image
      jtList: basic pixel data: eRow,eCol,gradDir,slip,. . .
      slipList: aligned (with jtList) lists of just the slips
      slateShape: the image dimensions
      limit: geometric information allowing registration of kml
      from above:
          widMin,widMax,minTop,maxTop,TPercentile,minSlip,maxSlip,
      """
      labelN = self.labelN
      if self.minSlip == 0. and self.maxSlip == 0.:
         # The slip histogram tends to long thin tails, and
         # there are tiny values that don't appear meaningful.
         # So use index 0 to  NNth percentile
         idxNN = int(self.TPercentile*0.01*len(jtList) )
         minSlip = jtList[0].slip
         maxSlip = jtList[idxNN].slip
      else:
         minSlip = self.minSlip
         maxSlip = self.maxSlip

      blankSlate = np.zeros(slateShape, np.uint8)
      if len(slipList) != 0:
         minTop,maxTop = self.minTop,self.maxTop
      jtThisRun = []
      for px in jtList:
         sL = px.slip
         #if sL < minSlip: sL = minSlip
         if sL > maxSlip: sL = maxSlip
         if sL < minSlip:
            continue
         tD = px.topDepth
         # Clip to whatever range has been set
         if tD < minTop: tD = minTop
         if tD > maxTop: tD = maxTop
         fracWidth = (tD - minTop)/(maxTop-minTop)
         lineWidth = fracWidth * self.widMax + (1.-fracWidth)*self.widMin
         lineWidth = int(lineWidth) 
         # circle arg "-1" indicates filled circles
         cirCenter = (px.eCol,px.eRow)
         lineColor = 127 # gray
         cv2.circle(blankSlate,cirCenter,lineWidth,lineColor,-1)
      alpha = blankSlate < 0.1 # really 0 on a 255 scale
      alpha = np.uint8(alpha)*255 # background is now 0 (black)
      atmp = np.logical_not(alpha)
      abar = np.uint8(atmp)
      abar = abar*255 # background is here 255 (white, or "a" transparent);
      gbar = abar*127 # does this make lines gray??
      rgba = (gbar,gbar,gbar,abar)
      imTrans = cv2.merge(rgba)
      rootName = 'TDgray' + labelN
      image2outs(imTrans,rootName,limit,False)

class pxl:
   """
   pxl is a row, column pair
   """
   def __init__(self, row,col):
      self.row = int(row)
      self.col = int(col)
    
class deadPixEnv:
   """
   pixEnv contains the object pxl for the dead pixel encountered, and
   indices of the bracketing valid information
   for that pixel, typically in the deadPixel list.

   Intention: initialize as item in list with known pxl, 
   with no additional information.
   As row or column information is encountered, access by list index
   """
   def __init__(self, pxl,lower=None,upper=None,left=None,right=None):
      self.pxl = pxl
      self.lower = lower
      self.upper = upper
      self.left = left
      self.right = right

# the following collect all the dead pixel cells (from a prior collected list)
# for a requested column or row.  
# Might be quicker to initialized a list for every col and row
# and add the row and column indices as we first discover dead pixels
def colOfPixList(col, pixList):
   """ 
   colOfPixList finds all the pixList (dead pixels) in requested col
   and returns the list of row indices.
   """
   rowsOfColumn = []
   for px in pixList:
      if px.col == col:
         rowsOfColumn.append(px.row)
   return rowsOfColumn
   
def rowOfPixList(row,pixList):
   """
   rowOfPixList finds all the pixList (dead pixels) in requsted row
   and returns list of the column indices
   """
   columnsOfRow = []
   for px in pixList:
      if px.row == row:
         columnsOfRow.append(px.col)
   return columnsOfRow

def nextContiguousSet(indexList):
   """
   nextContiguousSet finds contiguous entries in indexList starting
   with index "start."  Might be more convenient to have calling function
   modify the list each call, removing the contiguous set as it is processed
   so that next call finds the next one without "start" item passed.

   Calling example: from column indices of dead pixels, print contiguous sets

   #t=rowOfPixList(2, deadPxList)
   #t.sort()
   #st = 0
   #while 1:
   #   a,t =nextContiguousSet( t)
   #   if a == []: break
   #   st = max(a) + 1
   #   print a
   #
   """
   start = 0
   contigList = []
   firstInList = None
   remaining = [] # if contigList gets to end of row/col, none remain
   for idex,item in enumerate(indexList):
      if firstInList is None:
         firstInList = item
         contigList.append(item)
         
      else:
         if item == contigList[-1] + 1:
            contigList.append(item)
         else:
            remaining = indexList[idex:]
            break
   return contigList,remaining

def penTrim(image, nrow,ncol,dT):
    """ 
    Modify image to have additional zeros where more than half adjacent
    pixels are zero.
    """
    adjcy = 3 # 3x3 set of pixels considered
    buf = adjcy//2
    rad = adjcy//2
    modpix = [] # collect a list, so we don't have feedback
    # for each pixel away from outer boundary:
    for col in range(buf, ncol-buf):
       for row in range(buf, nrow-buf):
          countz = 0;
          # count zeros in ajacent pixels; skip self
          for drow in range(adjcy):
            for dcol in range(adjcy):
              if drow  == rad and dcol == rad: continue
              if image[row - rad + drow, col - rad + dcol] == dT.emptyFlag:
                countz += 1

          if countz > (adjcy*adjcy)/2.:
              modpix.append((row, col))
    for px in modpix:
       image[px] =  dT.emptyFlag
    return image
          
def holeSmooth(image, nrow,ncol,dT,limit):
    """ 
    Modify image to eliminate zeros with smoothed values.
    Use 100,50,25,12,6,3 Gaussian kernels.  That way jump at (original)
    dead pixel boundary should be small, and 98 pixel transition zone
    should be smooth transition out to a zero-value far-field
    Leave non-dead pixels alone: after each global smooth copy back
    new values for only dead pixels.
    """
    # identify dead pixels: they are preset with emptyFlag
    hSstartTime = time.time()
    deadPixList = []
    print('Processing image of size (rows, columns):',nrow,ncol)
    for col in range(ncol):
       for row in range(nrow):
          if image[row, col] == dT.emptyFlag:
             deadPixList.append(pxl(row, col))

    ################
    # biline interpolation for dead pixels
    imagePatch = holeInterp(image, nrow,ncol,dT)

    hSnowTime = time.time(); print(('\tpost holeInterp time: ',hSnowTime-hSstartTime))
    imageIn,scPatch,scShift = image2outs(imagePatch,"patch0",limit,True)

    ################
    # biline interpolation repeat: fills in corners
    image = holeInterp(imagePatch, nrow, ncol,dT)

    hSnowTime = time.time(); print(('\tpost 2holeInterp time: ',hSnowTime-hSstartTime))
    imageIn,scPatch,scShift = image2outs(image,"imageInterp",limit,True)

    #for px in deadPixList:
    #   image[px.row,px.col] = 0.0 # or mean, or something innocuous
    # apply blur to dead pixels only
    wlist = [100,50,50,25,25,25,12,12,12,6,6,6,6,3,3,3]
    for width in wlist:
        print('Smoothing dead pixels with width:',width)

        ################
        # apply gaussian smoothing, retaining new values at dead pixels
        # note imageSmooth from image, then image from imageSmooth
        if dT.dataType == 'F':
           imageSmooth = nd.gaussian_filter(image, width)
        if dT.dataType == 'C':
           imageSmooth = np.ones(imageIn.shape)*(1.+1.j)
           imageSmooth.real = nd.gaussian_filter(imageIn.real, width)
           imageSmooth.imag = nd.gaussian_filter(imageIn.imag, width)

        # Apply results of smoothing to dead pixels of "image"
        for px in deadPixList:
            image[px.row,px.col] = imageSmooth[px.row,px.col]
        del imageSmooth
        
        hSnowTime = time.time(); print(('\tHoleSmooth time: ',hSnowTime-hSstartTime))
    return image
    
     
def holeInterp(image, nrow,ncol,dT):
    """
    Modify image to eliminate zeros with interpolated values.
    Interpolation in each cell is mean of vertical and horizontal
    values at boundary of hole.
    """
    hIstartTime = time.time()
    deadPixList = []
    deadPixBookends = {} # dict keyed by row*ncol+coo containing bookends
    print('Processing image of size (rows, columns):',nrow,ncol)
    print('interpolating within each column')
    imTmpByCol = np.zeros(image.shape) + dT.emptyFlag
    imTmpByRow = np.zeros(image.shape) + dT.emptyFlag
    #Fresh Version combines Try2:
    for iCol, col in enumerate(image.T):
       diffArr = col[1:]-col[:-1] # records col(n+1)-col(n)
       if dT.dataType == "C": 
          diffArr = diffArr.real

       #nStart: positions where void sections start; nEnd: end positions
       nStart = np.where(diffArr >dT.emptyTest/2.)[0]+1
       nEnd = np.where(diffArr < -dT.emptyTest/2.)[0]
       # modify for empty region touching walls:  save in left.., right..
       # when so, snip nEnd, nStart so they describe interior voids only
       # First handle cases where a) all void, b) no void; both are no-op
       # so just skip.
       if len(nStart) != 0 or len(nEnd) != 0:
          # there are both empty and valid data on this line.
          # Treat gaps, first the ones at the boundaries:
          if col[0] == dT.emptyFlag:
             # treat initial void; trim nEnd list
             imTmpByCol[0:nEnd[0]+1,iCol] = image[nEnd[0]+1,iCol]
             nEnd = np.delete(nEnd,0)
          if col[-1] == dT.emptyFlag:
             # treat final void; trim nStart list
             imTmpByCol[nStart[-1]:, iCol] = image[nStart[-1] - 1, iCol]
             nStart = np.delete(nStart,-1)
             # loop over nStart nEnd pair gaps (if any)
          for bE in zip(nStart,nEnd):
             spanIncl = bE[1] - bE[0] + 3
             # +3 to include end and the bounding points
             v0, v1 = image[bE[0]-1,iCol], image[bE[1]+1,iCol]
             # fill each cell currently undefined by line thru bounding points
             w0 = np.arange(1.,spanIncl-1) / (spanIncl - 1) #float ramp, fills emptyFlag span
             w1 = 1. - w0
             bitOfItmp = w0*v0 + w1*v1
             imTmpByCol[bE[0]:bE[1]+1,iCol] = bitOfItmp
    for iRow, row in enumerate(image):
       diffArr = row[1:]-row[:-1] # records row(n+1)-row(n)
       if dT.dataType == "C": 
          diffArr = diffArr.real

       nStart = np.where(diffArr > dT.emptyTest/2.)[0]+1
       nEnd = np.where(diffArr < -dT.emptyTest/2.)[0]
       if len(nStart) != 0 or len(nEnd) != 0:
          if row[0] == dT.emptyFlag:
              
             imTmpByRow[iRow,0:nEnd[0]+1] = image[iRow,nEnd[0]+1]
             nEnd = np.delete(nEnd,0)
          
          if row[-1] == dT.emptyFlag:
              
             imTmpByRow[iRow,nStart[-1]:] = image[iRow,nStart[-1] - 1]
             nStart = np.delete(nStart,-1)
          
          for bE in zip(nStart,nEnd):
             spanIncl = bE[1] - bE[0] + 3
             v0, v1 = image[iRow,bE[0]-1], image[iRow,bE[1]+1]
             w0 = np.arange(1.,spanIncl-1) / (spanIncl-1) #float ramp, fills emptyFlag span
             w1 = 1. - w0
             bitOfItmp = w0*v0 + w1*v1
             
             imTmpByRow[iRow,bE[0]:bE[1]+1] = bitOfItmp

    # now fill empty parts of image with mean of the linear interp vals
    operatingIndices = np.where(np.logical_and( imTmpByCol < dT.emptyTest/2., 
        imTmpByRow == dT.emptyTest ))
    image[operatingIndices] = imTmpByCol[operatingIndices]
                        
    operatingIndices = np.where(np.logical_and( imTmpByRow < dT.emptyTest/2., 
        imTmpByCol == dT.emptyTest ))
    image[operatingIndices] = imTmpByRow[operatingIndices]
                        
    operatingIndices = np.where(np.logical_and( imTmpByRow < dT.emptyTest/2., 
        imTmpByCol< dT.emptyTest/2. ))
    image[operatingIndices] = 0.5*(imTmpByCol[operatingIndices]+imTmpByRow[operatingIndices])

    hInowTime = time.time(); print(('\t\tholeInterp Done time: ',hInowTime-hIstartTime))
    return image


def finalImage(nkern,edges,limit,rootName,color,dT):
   '''
   nkern: boxy width of lines representing edges
   edges: Canny-produced image with edges marked 1 (white)
   limit: dict of geometric scaling factors
   rootName: tag, first part of produced file names
   color: "black" or "white" indicating line color
   '''
   print('Dilating with kernel',nkern,'x',nkern)
   scEdges = np.uint8(edges)
   scEdges = scEdges*255 # edges now marked 255
   kernel= np.ones((nkern,nkern),np.uint8)
   scEdges = cv2.dilate(scEdges,kernel,iterations = 1) # grows white

   # Canny produces white stripes on black.
   # Transparent alpha is same: black is 0, so as alpha, is fully transparent
   # Also need reverse (black stripes on white) for final image, rgb.
   # Divide values into True or False (x>128 is True)
   alpha = scEdges>128  # calls edges "True"
   atmp = np.logical_not(alpha) # calls edges "False" (and abar does, too)
   abar = np.uint8(atmp)
   abar = abar*255 # abar has edges set to 0, background to 255
   alpha = np.uint8(alpha)*255 # edges now 255, background 0 (transparent).
   #so abar has black stripes on white background.

   if color is 'black':
      rgba = (abar,abar,abar,alpha) # edges 0 0 0 255
   else:
      rgba = (alpha,alpha,alpha,alpha) # edges 255 255 255 255

   imTrans = cv2.merge(rgba)

   image2outs(imTrans,rootName,limit,False)


def makeGoogEarthFile(image, limit, rootName):
   """
   image: the array of valued to be made png, kmz
   limit: needed geometry numbers to compute coordinates for layer
   post: optional file postfix (to distguish thickness outputs)
   ###XXX the naming and passing of files is poor: just pass a root name
   to here, and make both the png and kml files; or make png above, but
   with a named varable "fileRoot" that is also passed.
   """
   nlonpix = limit['NlonPix']
   # insisting subsampling is on square arrays
   # the input points lie on a 2D indexed grid of points,
   # with spacing in lon, lat of LonDelta and LatDelta,
   # occupying indices from MinIx, MinIy to MaxIx, MaxIy
   # which correpond to a box in lon, lat of 
   # (lon spacing > 0):
   # LonMin + LonDelta*MinIx to LonMin + LonDelta *MaxIx
   # (or if spacing < 0):
   # LonMin + LonDelta*MaxIx to LonMin + LonDelta *MinIx
   # (lat spacing > 0):
   # LatMin + LatDelta*MinIx to LatMin + LatDelta *MaxIx
   # (or if spacing < 0):
   # LatMin + LatDelta*MaxIx to LatMin + LatDelta *MinIx
   # (in which case LatMin is not really a minimum, just the start).  
   
   lonDelta,lonMin,nLon = limit['LonDelta'],limit['LonMin'],limit['Nlon']
   latDelta,latMin,nLat = limit['LatDelta'],limit['LatMin'],limit['Nlat']
   minIx,minIy,maxIx,maxIy = limit['MinIx'],limit['MinIy'],\
           limit['MaxIx'],limit['MaxIy']

   if lonDelta > 0:
       westLon = lonMin + minIx*lonDelta
       eastLon = lonMin + maxIx*lonDelta
   else:
       westLon = lonMin + maxIx*lonDelta
       eastLon = lonMin + minIx*lonDelta
   if latDelta > 0:
       southLat = latMin + minIy*latDelta
       northLat = latMin + maxIy*latDelta
   else:
       southLat = latMin + maxIy*latDelta
       northLat = latMin + minIy*latDelta
       
   # Define lines for google earth overlay kml file
   kml1 = \
"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
<GroundOverlay>
	<name>%s</name>
	<Icon>
		<href>%s</href>
		<viewBoundScale>0.75</viewBoundScale>
	</Icon>
        <altitudeMode>absolute</altitudeMode>
	<LatLonBox>
		<north>%f</north>
		<south>%f</south>
		<east>%f</east>
		<west>%f</west>
	</LatLonBox>
</GroundOverlay>
</kml>
"""
   pngName = rootName + ".png"
   kmlName = rootName + '.kml'
   kmzName = rootName + '.kmz'
   # overwrite any existing files via open 'w' and open-close for kmz

   if len(image.shape) == 2:
      i2png(image,rootName)

   if len(image.shape) == 3: # this is rgba image with color and transparency
      cv2.imwrite(pngName, image)

   kml_interp = kml1 % (rootName,pngName,northLat, southLat,eastLon,westLon)
   fp = open(kmlName, "w")
   fp.write(kml_interp)
   fp.close()

   # create, populate kmz file
   # make sure we start with empty kmz file
   zf = zipfile.ZipFile(kmzName,'w')
   zf.close()

   zf = zipfile.ZipFile(kmzName,'a')
   zf.write(kmlName)
   zf.write(pngName)
   zf.close()

def get_keys(ism):
   """
   read keys from ism file
   """
   int_keys = ["MinIx", "MaxIx","MinIy","MaxIy","NlonPix","NlatPix",
               "Nlon","Nlat"]
   flt_keys = ["LonMin", "LatMin","LonDelta","LatDelta","GridX","GridY",
                "Phase2mm","RefLon","RefLat"]
   good_keys = int_keys + flt_keys
   limit = {}
   for line in ism:
      if line.find("#") == 0: # skip comments
         continue
      else:
         lkey, lval = line.split()
         if lkey in good_keys:
            if lkey in int_keys:
               limit[lkey] = int(lval)
            elif lkey in flt_keys:
               limit[lkey] = float(lval)
            else:
               limit[lkey] = lval
         else:
            print("Bad key!", lkey)
   return limit

def mscale(m):
   """
   m is nd matrix  of values
   return  same (but scaled to 0-255)
   """
   cmin, cmax = m.min(),m.max()
   cscale = 255.0/(cmax-cmin)
   cshift = -cscale*cmin
   m=m*cscale
   m=m+cshift
   return m, cscale

def iPctScale(cp, pct):
   """
   cp is cv array
   create scale from pct, 100-pct values of array,
   saturate values beyond these.
   return scaled cv array
   """
   cpTmp = copy.deepcopy(cp).reshape(-1)
   cpTmp.sort()
   cmin = cpTmp[int(len(cpTmp)*pct/100.)]
   cmax = cpTmp[int(len(cpTmp)*(100-pct)/100.)]
   cp=np.clip(cp,cmin,cmax)
   #cmin, cmax,minij,maxij=cv2.minMaxLoc(cp)
   if cmax - cmin == 0.:
      return cp,1.
   cscale = 255.0/(cmax-cmin)
   cshift = -cscale*cmin
   cu=cv2.convertScaleAbs(cp.astype(int), alpha=cscale,beta=cshift)
   return cu, cscale

def iscale(cp):
   """
   cp is cv array
   return scaled cv array
   """
   cmin = np.min(cp)
   cmax = np.max(cp)
   #cmin, cmax,minij,maxij=cv2.minMaxLoc(cp)
   if cmax - cmin == 0.:
      return np.uint8(cp),1.
   cscale = 255.0/(cmax-cmin)
   cshift = -cscale*cmin
   cu = cp*cscale+cshift
   #cu=cv2.convertScaleAbs(cp.astype(int), alpha=cscale,beta=cshift)
   return cu, cscale


def i2txt(arr, arname):
   """
   write numpy array arr to txt file.
   """
   qp = open(arname + '.txt','w')
   nrow,ncol = arr.shape
   for row in range(nrow):
       for col in range(ncol):
           qp.write('{} {} {}\n'.format(row,col,arr[row,col]))
   qp.close()

def i2png(arr, arname):
   """
   write numpy array arr to Portable Network Graphics file arname.png
   NOW using opencv SaveImage with scaling (but scaling looks funky)
   Get, pass the value strech scale from iscale so we can use for threshold
   """
   cu, cscale=iscale(arr)
   cv2.imwrite(arname+'.png', cu)
   return cscale

def gradient_plots(image, ec,dT,limit):
    '''
    gradient_plots products inspection-only plots of gradients, mag, logmag.
    'C' needs: support split operators for dgau2Dx, y, and convolutions
    Pretty sure want gradient of phase from split operator recombination,
    not gradient of complex (as perhaps now, see end mag)
    args:
      image - the array in view
      ec - edge control object (structure)
      dT - data type ("C", "F")
      limit - geo information to create kml
    '''
    pw = list(range(1, 31))
    ssq = ec.sigma*ec.sigma
    wlist = np.nonzero([exp(-k*k/(2*ssq))>ec.gauss_dieoff for k in pw])
    if len(wlist[0]) > 0:
       width = wlist[0][-1]
    else:
       width = 1

    x = list(range(-width, width+1))
    y = list(range(-width, width+1))
    
    dgau2Dx = np.array([[-xi * exp(-(xi*xi+yi*yi)/(2*ssq))/(pi*ssq) for xi in x] for yi in y])
    i2png(dgau2Dx, "dgau2Dx")
    dgau2Dy = np.array([[-yi * exp(-(xi*xi+yi*yi)/(2*ssq))/(pi*ssq) for xi in x] for yi in y])
    i2png(dgau2Dy, "dgau2Dy")
    ax = nd.convolve(image, dgau2Dx,mode='nearest')
    ay = nd.convolve(image, dgau2Dy,mode='nearest')
    i2png(ax, "ax")
    i2png(ay, "ay")

    if dT.dataType == 'C':
        mag2 = ax*ax.conjugate() + ay*ay.conjugate()
    else:
        mag2 = ax*ax + ay*ay
        
    del ax
    del ay
    mag = np.sqrt(mag2) + 0.01
    logmag = np.log(mag)
    image2outs(mag,"mag",limit,False)
    image2outs(logmag,"logmag",limit,False)
    #i2png(mag, "mag")
    #i2png(logmag, "logmag")

class DirectionJump:
   """
   DirectionJump holds a set of principal angles
   that all directional interpolations use for approximations,
   and methods to simply define interpolation grids for a direction.

   Invocation:
      dJ = DirectionJump(4)
      dKernel, dWeight,dL = dJ.directionKernel(46)
      ..
      dKernel, dWeight,dL = dJ.directionKernel(130)
      
   """
   def __init__(self, principalDirectionsPerQuad,):

      # dimensionless? constants from synthetic fault slip tests for inverting W,S
      self.A = -0.726
      self.B=  0.180
      self.C =0.851
      self.C90 = 0.908
      self.C70 = 0.894
      self.C55 = 0.875 
      self.C49 = 0.854
      self.E = -0.0500
      self.E90 = -0.0411
      self.E70 = -0.0455
      self.E55 = -0.0512
      self.E49 = -0.0534

      # directional scale factors from pythagoras
      self.Root10v3 = np.sqrt(10.)/3.
      self.Root2v2 = np.sqrt(2.)/2.

      if principalDirectionsPerQuad == 4:
         self.principalDirectionsPerQuad = principalDirectionsPerQuad
      else: 
         print("principalDirectionsPerQuad must be 4.")  
         print(("Requested: ", principalDirectionsPerQuad))
      baseDirection = [0,geofunc.atan2o(1.,3.),geofunc.atan2o(1.,1.),
                     geofunc.atan2o(3., 1.) ]
      self.direction = []
      for stepDirection in baseDirection:
         self.direction.append(stepDirection)
         self.direction.append(stepDirection + 90)
         self.direction.append(stepDirection + 180)
         self.direction.append(stepDirection + 270)
      loopDirection = [self.direction[-1]-360] + \
                       self.direction + [self.direction[0]+360]
      loopDirection.sort()
      self.bounds = []
      for indx in range(len(self.direction)):
         # Determine sector bounds that define region 
         #   closest to this direction
         lowBound = (loopDirection[indx] + loopDirection[indx+1])/2.
         highBound = (loopDirection[indx+1] + loopDirection[indx+2])/2.
         self.bounds.append((lowBound,highBound))
         
   def sanityTest(self,nCell,vSigned, signInt):
       # Slope must match sign of signInt
       localSlope = vSigned[nCell] - vSigned[nCell-1]
       meanSlope = vSigned[nCell] - vSigned[0]
       if localSlope*signInt > 0 and meanSlope*signInt > 0:
          isSane = True
       else:
          isSane = False
       return isSane

   def ratioTest(self,nCell,vSigned):
       """
       ratio Test determines mean slope (based on endpoints) vs. local slope
          nCell: index of cell between adject points
          vSigned: list of values of presumed sigmoid
       """
       numer = vSigned[nCell] - vSigned[nCell-1]
       denom0 = vSigned[nCell] - vSigned[0]
       if denom0 == 0:
          return 0.
       denom =  (1./float(nCell))*(denom0)
       return numer / denom

   def directionKernel(self,qDirection):
      """
      find the best matching principal direction 
         for arg qDirection in degrees

      qDirection: defined somewhat like strike (from +y, clockwise)
         but entirely in index coordinates (not NEWS, no regard for nonsquare).
      Returns:
         kernel: list of point and point-pairs in dRow, dCol
         weight: these weights are used to sum values at point-pairs
         dLength: length traversed at each increment of nCell; 
              1 on axes, or
              1/3 of a 1x3 cycle (diagonal sqrt(10)), or
              1/2 of a 1x1 cycle (diagonal sqrt(2))
         
      """
      if qDirection < self.bounds[0][0]: qDirection += 360.
      if qDirection > self.bounds[-1][1]: qDirection -= 360.
      self.qDirection = qDirection
      for indx in range(len(self.direction)):
         if self.bounds[indx][0] <= qDirection <= self.bounds[indx][1]:
              direction0 = self.direction[indx]
              primeIndex = indx
              break
      # these directions (kernels) start at x and proceed CCW.
      # Must match gradDir, which is based on atan2(gx,gy) (strike-like)
      if primeIndex == 0:
         kernel = [(1,0)]
         weight = [1.]
         dLength = 1.
      elif primeIndex == 1:
         kernel = [[(1, 0),(1, 1)],[(2,0),(2,1)],(3,1)]
         weight = [[2./3., 1./3.], [1./3., 2./3.], 1.]
         dLength = self.Root10v3
      elif primeIndex == 2:
         kernel = [[(1,0),(0,1)],(1,1)]
         weight = [[0.5, 0.5], 1.0]
         dLength = self.Root2v2
      elif primeIndex == 3:
         kernel = [[(0,1),(1,1)],[(0,2),(1,2)],(1,3)]
         weight = [[2./3., 1./3.], [1./3., 2./3.], 1.]
         dLength = self.Root10v3
         
      elif primeIndex == 4:
         kernel = [(0,1)]
         weight = [1.]
         dLength = 1.
      elif primeIndex == 5:
         kernel = [[(0,1),(-1,1)],[(0,2),(-1,2)],(-1,3)]
         weight = [[2./3., 1./3.], [1./3., 2./3.], 1.]
         dLength = self.Root10v3
      elif primeIndex == 6:
         kernel = [[(-1,0),(0,1)],(-1,1)]
         weight = [[0.5, 0.5], 1.0]
         dLength = self.Root2v2
      elif primeIndex == 7:
         kernel = [[(-1, 0),(-1, 1)],[(-2,0),(-2,1)],(-3,1)]
         weight = [[2./3., 1./3.], [1./3., 2./3.], 1.]
         dLength = self.Root10v3

      elif primeIndex == 8:
         kernel = [(-1,0)]
         weight = [1.]
         dLength = 1.
      elif primeIndex == 9:
         kernel = [[(-1, 0),(-1, -1)],[(-2,0),(-2,-1)],(-3,-1)]
         weight = [[2./3., 1./3.], [1./3., 2./3.], 1.]
         dLength = self.Root10v3
      elif primeIndex == 10:
         kernel = [[(-1,0),(0,-1)],(-1,-1)]
         weight = [[0.5, 0.5], 1.0]
         dLength = self.Root2v2
      elif primeIndex == 11:
         kernel = [[(0,-1),(-1,-1)],[(0,-2),(-1,-2)],(-1,-3)]
         weight = [[2./3., 1./3.], [1./3., 2./3.], 1.]
         dLength = self.Root10v3
         
      elif primeIndex == 12:
         kernel = [(0,-1)]
         weight = [1.]
         dLength = 1.
      elif primeIndex == 13:
         kernel = [[(0,-1),(1,-1)],[(0,-2),(1,-2)],(1,-3)]
         weight = [[2./3., 1./3.], [1./3., 2./3.], 1.]
         dLength = self.Root10v3
      elif primeIndex == 14:
         kernel = [[(1,0),(0,-1)],(1,-1)]
         weight = [[0.5, 0.5], 1.0]
         dLength = self.Root2v2
      elif primeIndex == 15:
         kernel = [[(1, 0),(1, -1)],[(2,0),(2,-1)],(3,-1)]
         weight = [[2./3., 1./3.], [1./3., 2./3.], 1.]
         dLength = self.Root10v3
      else:
         print("Warning: no matching direction found by directionKernel")
         print(( "primeIndex,indx,direction0",primeIndex,indx,direction0))
      self.kernel = kernel
      self.weight = weight
      self.dLength = dLength
      return kernel,weight,dLength

   def evalSignedBranch(self,edgePix,imageIn,signStr):
      """
      evalSignedBranch: collect single-side values near detected edge
      Arguments:
        edgePix : row, col of detection cell
        imageIn: the image array
        signStr: "Positive" or "Negative"
      Method:
        as we traverse cells, nCell: 1 2 3 4 5 6 7
        if cycN is 3, cycle count is 1 1 1 2 2 2 3
        and kIndex is supposed       0 1 2 0 1 2
      """
      if signStr == 'Positive':
          signInt = 1
      elif signStr == 'Negative':
          signInt = -1
      else :
         print(("Error: evalSignedBranch called with invalid signStr:%s")%(signStr,))
         exit()

      ratTestList = [0.]
      iRowList,jColList = [],[]
      maxRow,maxCol = imageIn.shape
      cycDrow,cycDcol = self.kernel[-1]
      cycN = len(self.weight)
      nCell = 0 # the "while" is an infinite "for" over nCell, beginning at 1
      vList = [imageIn[edgePix]]
      iRowList,jColList = [edgePix[0]],[edgePix[1]]
      while(1):
         cycleCount = nCell//cycN
         kIndex = nCell % cycN
         nCell += 1
         vIntp = 0.
         iRintp = 0.
         jCintp = 0.
         # For reference, typical kernel = [[(1,0),(0,-1)],(1,-1)]
         #   that is, [[list of tuples], tuple]
         if type(self.kernel[kIndex]) == type(()):
            # This kernel point is in the kernel direction: use directly
            dRow0,dCol0 = self.kernel[kIndex]
            # kernel given for one cycle only.  Extend it by unwinding
            # using cycle count
            dRow = dRow0 + cycleCount * cycDrow
            dCol = dCol0 + cycleCount * cycDcol
            # in positive branch, advance dRow; else retreat dRow (dCol)
            iRow = edgePix[0] + signInt*dRow
            jCol = edgePix[1] + signInt*dCol
            iRintp = iRow
            jCintp = jCol
            # Don't allow index to run off array:
            #    better to lose data near image edges
            if not(0 <= iRow < maxRow and 0 <= jCol < maxCol) :
               cellCount = nCell - 1
               return vList[:-1],iRowList[:-1],jColList[:-1],ratTestList,cellCount
            vIntp = imageIn[iRow,jCol]

         else:
            # interpolate between off-direction kernel points
            for kernelValue, interpolationWeight\
                   in zip(self.kernel[kIndex],self.weight[kIndex]):
               dRow0, dCol0 = kernelValue
               dRow = dRow0 + cycleCount * cycDrow
               dCol = dCol0 + cycleCount * cycDcol
               iRow = edgePix[0] + signInt*dRow
               jCol = edgePix[1] + signInt*dCol
               if not(0 <= iRow < maxRow and 0 <= jCol < maxCol) :
                  cellCount = nCell - 1
                  return vList[:-1],iRowList[:-1],jColList[:-1],ratTestList,cellCount
               vIntp += imageIn[iRow,jCol]*interpolationWeight
               iRintp += float(iRow)*interpolationWeight
               jCintp += float(jCol)*interpolationWeight

         vList.append(vIntp)
         iRowList.append(iRintp)
         jColList.append(jCintp)

         isSane = self.sanityTest(nCell,vList,signInt)

         ratTestVal = self.ratioTest(nCell,vList)
         ratTestList.append(ratTestVal)

         # check for sanity, and acceptable range for continuing.
         # cannot consider case where ratTestVal > 1, as that must
         # be past an inflection point. (but first point will test at == 1.)
         if isSane and 0.2 <= ratTestVal <= 1.:
            continue
         else:
            # If rtest < 0.2 the slope flattens out, done
            # Note cellCount will be used for mean slope
            # Also wind up here if not isSane, so we terminate cleanly
            return vList,iRowList,jColList,ratTestList,nCell
      # end while(1) loop


   def valueCaliper(self,edgePix,imageIn,twoBranchCellSpan,fracSpan):
      """
      valueCaliper uses imageIn values at presumed stable distance from rupture.
      Returns difference of these values, representing deep slip (plate motion)
      After two calls to evalSignedBranch we have the bidirectional cell
      span defined by ratioTest. Cell count corresponding to 5*topDepth,
      for topDepth = (cellSize)*twoBranchCellSpan/5.55 (from W'' vs alpha plot),
      is +/- 0.9 * twoBranchCellSpan  
      This could be refined by inflating by another projection-like term, assuming
      we really are following an atan function (project to asymptote).
      Args:
         edgePix - the row, col of the rupture pixel in view
         imageIn - the uncalibrated values making up the image (0-255 I think)
         twoBranchCellSpan - equivalent to W'', 
            the cell count in both directions
            (end-to-end, sums both branches).
         fracSpan - originally 0.9 to match arctan leveling in .125 case;
            some results negative, so making variable to improve those.
      Return:
         valueCaliper - corresponds to the slip estimate (divide by cumScale to get mm).
           May be "0." to indicate error, suggesting skipping this pixel
      """
      maxRow,maxCol = imageIn.shape
      nCell = int(fracSpan*twoBranchCellSpan + 1.)
      cycDrow,cycDcol = self.kernel[-1]
      cycN = len(self.weight)
      cycleCount = nCell//cycN
      kIndex = nCell % cycN
      # positive branch:
      signInt = 1
      if type(self.kernel[kIndex]) == type(()):
         dRow0,dCol0 = self.kernel[kIndex]
         dRow = dRow0 + cycleCount * cycDrow
         dCol = dCol0 + cycleCount * cycDcol
         iRow = edgePix[0] + signInt*dRow
         jCol = edgePix[1] + signInt*dCol
         if not(0 <= iRow < maxRow and 0 <= jCol < maxCol) :
            return 0.0
         vPos = imageIn[iRow,jCol]
      else:
         # interpolate between off-direction kernel points
         vPos = 0.
         for kernelValue, interpolationWeight\
                   in zip(self.kernel[kIndex],self.weight[kIndex]):
            dRow0, dCol0 = kernelValue
            dRow = dRow0 + cycleCount * cycDrow
            dCol = dCol0 + cycleCount * cycDcol
            iRow = edgePix[0] + signInt*dRow
            jCol = edgePix[1] + signInt*dCol
            if not(0 <= iRow < maxRow and 0 <= jCol < maxCol) :
               return 0.0
            vPos += imageIn[iRow,jCol]*interpolationWeight
      signInt = -1
      if type(self.kernel[kIndex]) == type(()):
         dRow0,dCol0 = self.kernel[kIndex]
         dRow = dRow0 + cycleCount * cycDrow
         dCol = dCol0 + cycleCount * cycDcol
         iRow = edgePix[0] + signInt*dRow
         jCol = edgePix[1] + signInt*dCol
         if not(0 <= iRow < maxRow and 0 <= jCol < maxCol) :
            return 0.0
         vNeg = imageIn[iRow,jCol]
      else:
         # interpolate between off-direction kernel points
         vNeg = 0.
         for kernelValue, interpolationWeight\
                   in zip(self.kernel[kIndex],self.weight[kIndex]):
            dRow0, dCol0 = kernelValue
            dRow = dRow0 + cycleCount * cycDrow
            dCol = dCol0 + cycleCount * cycDcol
            iRow = edgePix[0] + signInt*dRow
            jCol = edgePix[1] + signInt*dCol
            if not(0 <= iRow < maxRow and 0 <= jCol < maxCol) :
               return 0.0
            vNeg += imageIn[iRow,jCol]*interpolationWeight

      return vPos-vNeg


   def slipWidthFit(self,xS,yS,cumScale,edgePix,imageIn,valf):
      """
      Produce estimate of slip, width (defined as elastic fault top depth)
        from single pixel imageIn(edgePix) and vicinity, using self.kernel and 
        self.weight to find count of points where transition has 
        serious slope.
      Account for cumScale for slip, grid spacing.
      Args:
        xS: spacing in x, GridX
        yS: spacing in y, GridY
        cumScale: scale factor from prior scaling
        edgePix: Row, col of this edge pixel
        imageIn: array of pixel values
      """
      # ################bail for use during development
      cycN = len(self.weight)
      iNRowList, iPRowList, jNColList,jPColList = [],[],[],[]
      maxRow,maxCol = imageIn.shape
      vPosList = [imageIn[edgePix]]
      vNegList = [imageIn[edgePix]]
   

      # These returned lists,  and count, include the considered point at end, which
      # failed ratioTest.  So we must drop it, but value here can be
      # reported as debug quantity.
      pVals,pRow,pCol,pRatT,cellPosCount = self.evalSignedBranch(edgePix,imageIn,"Positive")
      nVals,nRow,nCol,nRatT,cellNegCount = self.evalSignedBranch(edgePix,imageIn,"Negative")
      # compose full profile:
      vals = list(reversed(nVals))+pVals[1:]
      rowIndex = list(reversed(nRow))+pRow[1:]
      colIndex = list(reversed(nCol))+pCol[1:]
      ratio = list(reversed(nRatT))+ pRatT[1:]

      # twoBranchCellSpan is cells to left, +cells to right
      # W''
      twoBranchCellSpan = cellPosCount + cellNegCount -2
      # D''
      # skip first and last value, as these were rejected by ratioTest
      # and appear here only for debugging.
      valSpread = vals[-2]-vals[1] # vNeg is constructed of opposite sign
      valCaliper = self.valueCaliper(edgePix,imageIn,twoBranchCellSpan,0.9)
      cCoef = self.C90
      eCoef = self.E90
      if valCaliper == 0.:
         valCaliper = valSpread
         cCoef = self.C
         eCoef = self.E
      if valCaliper > 4.*valSpread or valCaliper < 0.75*valSpread:
         valCaliper = self.valueCaliper(edgePix,imageIn,twoBranchCellSpan,0.7)
         cCoef = self.C70
         eCoef = self.E70
      if valCaliper > 4.*valSpread or valCaliper < 0.75*valSpread:
         valCaliper = self.valueCaliper(edgePix,imageIn,twoBranchCellSpan,0.55)
         cCoef = self.C55
         eCoef = self.E55
      if valCaliper > 4.*valSpread or valCaliper < 0.75*valSpread:
         valCaliper = self.valueCaliper(edgePix,imageIn,twoBranchCellSpan,0.49)
         cCoef = self.C49
         eCoef = self.E49
      if valCaliper < valSpread :
         valCaliper = valSpread
         cCoef = self.C
         eCoef = self.E

      # dX, dY from one kernel interpolated point to the next
      xCellLen = self.kernel[-1][0]/float(cycN)
      yCellLen = self.kernel[-1][1]/float(cycN)
      # physical-units version of same, in km
      realXScale = xCellLen*xS
      realYScale = yCellLen*yS
      # physical-units distance between succesive kernel interp. points
      realCellDirScale = np.sqrt(realXScale*realXScale+realYScale*realYScale)
      # omega defined as W''/dL; W'' is the twoBranchCellSpan*dL
      # so omega is unitless, simply count of cells.
      omega = twoBranchCellSpan
      # tau defined as topDepth/Dx, and has this relation:
      tau = self.A + self.B*omega
      if tau < 0.: tau = 0.
      topDepthEst = tau*realCellDirScale
      # sigma defined as S''/Strue, and also has this relation:
      if tau > 0.1:
         sigma = cCoef * np.power(tau,eCoef)      
      else:
         sigma = 0.
      # where S'' is same as valCaliper; sigma = valCaliper/Strue
      # implies Strue which is slipEst = valCaliper/sigma.
      if sigma != 0.:
         slipEst = valCaliper/sigma/cumScale
      else:
         slipEst = 0.
      slipCaliper = valCaliper/cumScale
      valf.write("%f %f %d %d %s\n"%(realCellDirScale,cumScale,edgePix[0],edgePix[1], " ".join("%f"%v for v in vals)))
      return topDepthEst,slipEst,slipCaliper

### end directionJump methods ###

class jtVals:
   """
   jtVals contains jumpTable items
   x, y (km) gradDir (deg), slip(mm), slipCaliper(mm), topDepth, edgePixRow, edgePixCol, lon, lat
   where slipCaliper is the rescaled data span from which slip is estimated.
   Note x,y are based on col, row and correspond to lon, lat respectively.
   """
   def __init__(self,x,y,gradDir,slip,slipCaliper, topDepth, eRow,eCol,limit):
      # Ix, etc are in subblock count (in sar2ccc, blockIndex and subraster.
      nlonpix = limit['NlonPix']
      nlatpix = limit['NlatPix']
      lonsubdelta = limit['LonDelta']
      lonsubmin = limit['LonMin'] + lonsubdelta/2.
      latsubdelta = limit['LatDelta']*nlatpix
      latsubmin = limit['LatMin'] + latsubdelta/2.
      minIx = limit['MinIx']
      minIy = limit['MinIy']

      lonsubdelta = limit['LonDelta']
      lonsubmin = limit['LonMin'] +  minIx*lonsubdelta + lonsubdelta/2.

      latsubdelta = limit['LatDelta']
      latsubmin = limit['LatMin'] +  minIy*latsubdelta + latsubdelta/2.

      self.lonPoint = lonsubmin + lonsubdelta * eCol
      self.latPoint = latsubmin + latsubdelta * eRow

      x = int(ceil((self.lonPoint - limit['RefLon'])/lonsubdelta)) * limit['GridX']
      y = int(ceil((self.latPoint - limit['RefLat'])/latsubdelta)) * limit['GridY']
      self.x = x
      self.y = y
      self.gradDir = gradDir
      self.slip = slip
      self.slipCaliper = slipCaliper
      self.topDepth = topDepth
      self.eRow = int(eRow)
      self.eCol = int(eCol)

class EdgeControl:
   """
   EdgeControl holds the controlling parameters for edge detection:
      gauss_dieff
      threshold_ratio
   """
   def __init__(self, gauss_dieoff,threshold_ratio,
         sigma, scale,attempt_flatten):
      self.gauss_dieoff = float(gauss_dieoff)
      self.threshold_ratio = float(threshold_ratio)
      self.sigma = float(sigma)
      self.scale = float(scale)
      self.attempt_flatten = attempt_flatten

def main(argv=None):
    startTime = time.time()
    if argv is None:
        argv = sys.argv
    commandline = ""
    for arg in argv: 
        commandline +=  arg + ' '
    commandline += '\n'

    try:
        try:
            opts, args = getopt.getopt(argv[1:], "a:hm:r:s:", ["help", "output="])
        except getopt.error as msg:
             raise Usage(msg)

        # option processing
        # Defaults (aperture, HiMmThresh,threshRatio):
        aperture = 5
        HiMmThresh = 2.
        threshRatio = 0.75
        sigmaInOriginalPixels = 1.
        for option, value in opts:
            if option in ("-a"):
                aperture = int(value)
            if option in ("-h", "--help"):
                raise Usage(__doc__)
            else:
                redof = open('REDO.edge', 'w')
                redof.write(commandline)
                redof.close()
            if option in ("-m"):
                HiMmThresh = float(value)
            if option in ("-r"):
                threshRatio = float(value)
            if option in ("-s"):
                sigmaInOriginalPixels = float(value)

        if len(args) != 3:
            raise Usage(__doc__)

        sim = open(args[0], 'r')
        line = sim.readline()
        sim.seek(0) # rewind file 
        itype, x,y,obs,sig,el,az = line.split()
        if 'j' in obs: 
           dT = DataType('C')
           obs = complex(obs)
        else: 
           dT = DataType('F')
           obs=float(obs)

        ixf = open(args[1], 'r')
        print('Index file: ', args[1])
        isumfile = args[2]
        print('Summary file: ', isumfile)
        # Items in isum.txt:
        # NlatPix: vertical (Lat) stride, typ 3 or 6, from  commandline -j arg
        # NlonPix: horizontal (Lon) stride, typ 3 or 6, from  commandline -g arg
        # MinIx,MaxIx,MinIy,MaxIy: box limits of input data, subgrid indices
        # LonMin,LonDelta,LatMin,LatDelta,Nlon,Nlat: subgrid coordinate parameters
        # GridX, GridY: subgrid spacing in flat-earth at ref, in km
        # RefLon, RefLat: reference coordinates for computing x, y


        ism = open(isumfile, "r")
        limit = get_keys(ism)
        sampling = limit['NlonPix']

        if dT.dataType == 'C':
            phase2mm = float(limit['Phase2mm'])

        # Set sigma for a pre-smooth step, roughly:
        # Sue got good results for 2 for nlonpix==3, so:
        sigmaInCurrentPixels = sigmaInOriginalPixels/sampling


        # now that we seek to fill holes, ok to initially set to 1.e20
        dimx = limit['MaxIx'] - limit['MinIx'] + 1
        dimy = limit['MaxIy'] - limit['MinIy'] + 1

        # like tranpose: x corresponds to column, y with row
        nrow, ncol = dimy,dimx

        # set initial image (image) to the empty value
        # later filled with values where known
        image = np.ones((nrow, ncol))*dT.emptyFlag
        elevAngle = np.zeros((nrow, ncol))

        # fill valid part of image with observations, one pixel at a time
        sim.seek(0) # rewind file for 2nd pass
        for line in sim:
           itype, x,y,obs,sig,el,az = line.split()
           idx, idy = ixf.readline().split()
           idx = int(idx)
           idy = int(idy)
           col = idx - limit['MinIx']
           row = idy - limit['MinIy']
           if dT.dataType == 'C':
              image[row, col] = complex(obs) 
              elevAngle[row,col] = el
           if dT.dataType == 'F':
              image[row, col] = float(obs) 
              elevAngle[row,col] = float(el)
        azimuth = float(az)

        nowTime = time.time(); print(('Data fill complete time: ',nowTime-startTime))

        # initialize cumScale.  Currently picks up trim, smooth operation scales
        cumScale  = 1

        # At this point image may be real or complex. 
        image,iScale,iShift = image2outs(image,"raw",limit,False)

        nowTime = time.time(); print(('Pre Trim time: ',nowTime-startTime))

        imageTrim = penTrim(image, nrow,ncol,dT)
        imageTrim = penTrim(imageTrim, nrow,ncol,dT)
        image,imageScale,iShift = image2outs(imageTrim,'trim',limit,False)
        #i2txt(imageTrim, "trim")
        #i2png(imageTrim, "trim")

        nowTime = time.time(); print(('Post Trim time: ',nowTime-startTime))

        del image

        imageTrima = copy.deepcopy(imageTrim)
        imagePatch = holeSmooth(imageTrim, nrow,ncol,dT,limit)

        nowTime = time.time(); print(('Post holeSmooth time: ',nowTime-startTime))
    

        imageIn,scPatch,scShift = image2outs(imagePatch,"patch",limit,True)
        #cvPatch,scPatch = iscale(imagePatch)
        cumScale *= scPatch

        #i2txt(imagePatch, "patch")
        #imageIn, imageScale = mscale(imagePatch)
        #i2png(imageIn, "patch")


        # set paraeters in EdgeControl object: before filter so can use ec.sigma
        ec = EdgeControl(gauss_dieoff = 0.0001, 
                                  threshold_ratio = threshRatio, 
                                  sigma = sigmaInCurrentPixels, 
                                  scale = cumScale,
                                  attempt_flatten = False)

        
        if dT.dataType == 'F':
           imageIn = nd.gaussian_filter(imageIn, ec.sigma)
        if dT.dataType == 'C':
           imageIn.real = nd.gaussian_filter(imageIn.real, ec.sigma)
           imageIn.imag = nd.gaussian_filter(imageIn.imag, ec.sigma)

        nowTime = time.time(); print(('Post gaussian_filter time: ',nowTime-startTime))
    
        #cvScImage, smoothScale = iPctScale(imageIn, 0.25)
        #cvScImage, smoothScale,smShift = image2outs(imageIn,"smoothed",limit,True,iPct=0.25)
        cvScImage, smoothScale,smShift = image2outs(imageIn,"smoothed",limit,True)
        cumScale *= smoothScale
        ec.scale = cumScale
        # final scaled image: cvScImage; corresponding cumScale
        #i2txt(cvScImage, "smoothed")
        #i2png(cvScImage, "imageSmoothed")

        # report parameters now that we have cumScale
        lp = open("log", "w")
        lp.write("aperture: %d -- highthresh: %f --"\
                "threshRatio : %f -- sigma: %f  "\
                "cumulative scale: %f\n"\
                     %(aperture, HiMmThresh,threshRatio, \
                       sigmaInOriginalPixels, cumScale))
        lp.close()
        csc = open("cumScale", "w")
        csc.write( "%f\n" %(cumScale))
        csc.close()

        if dT.dataType == 'F':
           highThreshold = HiMmThresh * ec.scale * pCal
        if dT.dataType == 'C':
           highThreshold = HiMmThresh # remaining calibration is in cCanny

        LoMmThresh = HiMmThresh*ec.threshold_ratio
        lowThreshold = ec.threshold_ratio*highThreshold

        # Compute gradient separately - purely for diagnostic
        if dT.dataType == 'C':
           cvFl = np.arctan2(cvScImage.imag,cvScImage.real)
        else:
           cvFl = cvScImage.astype(float)
        gradient_plots(cvFl, ec,dT,limit)

        print(( "High threshold: ",highThreshold))
        print(( "Low: ",lowThreshold))
        print(( "aperture:",aperture))

        if dT.dataType == 'F':
           cvEdgeMap = cv2.Canny(cvScImage, lowThreshold,highThreshold,
              apertureSize=aperture, L2gradient=False)
        if dT.dataType == 'C':
           print("Running cCanny")
           cvEdgeMap = cCanny(cvScImage, highThreshold,
              phase2mm,1./ec.scale,limit,dT) # find correct scale to pass!
           cvEdgeMap *= 255
           cvEdgeMap,iScale,iShift = image2outs(cvEdgeMap,'cvEdgeMap',limit,False)

        scratchEdgeMap = copy.copy(cvEdgeMap)


        nowTime = time.time(); print(('Post Canny time: ',nowTime-startTime))
    

        boolEdgeMap = cvEdgeMap > 128 # reversed from edgar1
        boolEdgeMap,iScale,iShift = image2outs(boolEdgeMap.astype(int),
                                       'boolEdgemap',limit,False)
        #i2txt(boolEdgeMap.astype(int),"boolEdgeMap")

        # no masking in blur case
        maskedEdgeMap = boolEdgeMap

        edgeList = []
        for row in range(nrow):
           for col in range(ncol):
              if maskedEdgeMap[row,col]:
                 edgeList.append((row,col))
        print('found',len(edgeList),'edge pixels of',len(cvEdgeMap)*len(cvEdgeMap[0]))

        ddepth = -1
        gX = cv2.Scharr(cvFl,ddepth,1,0)
        gY = cv2.Scharr(cvFl,ddepth,0,1)
        # first try at line drawing: first to last edge pixel 
        lineColor = 127
        # lineColor 127 is middle gray;
        origin = (0,0)
        
        maxRow,maxCol = cvFl.shape
        dJ = DirectionJump(4)
        jtList = []
        slipList = []
        topDList = []
        valf = open("valsFile.txt","w")
        for edgePix in edgeList:
            # omit pixels too close to boundary of grid (<= 3 pixels)
            # as gradient may be corrupted by points out of bounds
            if 2 < edgePix[0] < maxRow-3 and 2 < edgePix[1] < maxCol-3:
               xSpace,ySpace = limit["GridX"],limit["GridY"]
               dX,dY = gX[edgePix]*xSpace,gY[edgePix]*ySpace
               # not args dYdX, reversed to make clockwise from N
               #gradDir is a grid-defined direction (in row, col space)
               gradDir = geofunc.atan2o(gX[edgePix],gY[edgePix])
               trueGradDir = geofunc.atan2o(dX,dY)

               dKernel, dWeight,dLength = dJ.directionKernel(gradDir)

               topDepth,slip,slipCaliper  = dJ.slipWidthFit(xSpace,ySpace,
                            cumScale, edgePix,cvFl,valf)

               jtList.append(jtVals(edgePix[1]*xSpace,edgePix[0]*ySpace,
                   trueGradDir,slip,slipCaliper,topDepth,
                   edgePix[0],edgePix[1],limit))
               topDList.append(topDepth)
               slipList.append(slip)
        jtList.sort(key=lambda item: item.slip,reverse = False)

        jtf = open("jumpTableTotal.txt","w")
        # switch item 1, item 0 so we take row, col and turn into x, y
        # recalling row index movies in y direction, col in x.
        for px in jtList:
           if px.slip > 0.1: # avoid tiny slips
              #reverse y,x,and also Col, Row for easy plotting
              jtf.write('%f %f %f %f %f %f %f %f %f %f %f\n'\
                 %(px.x, px.y, px.gradDir, px.slip, px.slipCaliper,
                 px.topDepth,elevAngle[px.eRow,px.eCol],px.eCol,px.eRow,px.lonPoint,px.latPoint))
        jtf.close()


        # create inValid zone to cut down on artifacts: bad pixels
        # from original array, peninsulas trimmed, and buffer nkern around that.
        nkern = 8
        kernelBlock = np.ones((nkern,nkern),np.uint8) # a 4x4 array

        #i2txt(imageTrima, "trima")
        #i2png(imageTrima, "trima")
        imageTrima,iScale,iShift = image2outs(imageTrima,'trima',limit,False)

        badDataFlag = imageTrima > emptyFlag/2.
        maskTrim = np.uint8(badDataFlag)
        # maskTrim = maskTrim*255
        inValid = cv2.dilate(maskTrim,kernelBlock,iterations=1)
        inValid,iScale,iShift = image2outs(inValid,'inValid',limit,False)
        #i2txt(inValid, "inValid")
        #i2png(inValid, "inValid")

        
        # Gather into validJTList slip range information from valid pixels only
        validJTList = []
        validSlipList = []
        validTopList = []
        for px in jtList: 
           if inValid[px.eRow,px.eCol]:
              continue
           validJTList.append(px)
           validSlipList.append(px.slip)

        # sort enables selecting percentile clipping point
        validJTList.sort(key=lambda item: item.slip,reverse = False)

        dsRuns = [] # a list of class instances
        wMin,wMax = 1.,5.
        tMin,tMax = 0.005,0.5
        #(self, widMin,widMax,minTop,maxTop,TPercentile,minSlip,maxSlip,
        # colorMin,colorMax,labelN):

        dsRuns.append(DepthSlip(wMin,wMax,0.005,0.05,99,0.,0.,16,255,"0"))
        dsRuns.append(DepthSlip(wMin,wMax,0.005,0.05,0,10.,20.,16,255,"1"))
        dsRuns.append(DepthSlip(wMin,wMax,0.005,0.05,0,20.,40.,16,255,"2"))
        dsRuns.append(DepthSlip(wMin,wMax,0.005,0.05,0,40.,80.,16,255,"3"))
        dsRuns.append(DepthSlip(wMin,wMax,0.005,0.05,0,80.,160.,16,255,"4"))
        dsRuns.append(DepthSlip(wMin,wMax,0.005,0.05,0,8.,40.,16,255,"5"))
        dsRuns.append(DepthSlip(wMin,wMax,0.005,0.05,0,5.,20.,16,255,"6"))
        dsRuns.append(DepthSlip(wMin,wMax,0.005,0.05,0,8.,20.,16,255,"7"))
        dsRuns.append(DepthSlip(wMin,wMax,0.005,0.05,0,1.,21.,16,255,"8"))
        dsRuns.append(DepthSlip(wMin,wMax,0.005,0.05,0,3.,15.,16,255,"9"))

        k = 0
        for iRun in dsRuns:
           print(("DBG: doing run k",k))
           iRun.depthSlipPlots(validJTList,validSlipList,imageIn.shape,
              limit,elevAngle,dT)
           k += 1


        #DepthSlip args are
        # widMin,widMax,minTop,maxTop,TPercentile,minSlip,maxSlip,
        #      colorMin,colorMax,labelN
        sRun = DepthSlip(3,3,.005,.5, 99,0.,0.,16,255,"0")
        sRun.sleevedSlipPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(3,3,.005,.5, 0,10.,20.,16,255,"1")
        sRun.sleevedSlipPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(3,3,.005,.5, 0,20.,40.,16,255,"2")
        sRun.sleevedSlipPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(3,3,.005,.5, 0,40.,80.,16,255,"3")
        sRun.sleevedSlipPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(3,3,.005,.5, 0,80.,160.,16,255,"4")
        sRun.sleevedSlipPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(3,3,.005,.5, 0,8.,40.,16,255,"5")
        sRun.sleevedSlipPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(3,3,.005,.5, 0,5.,20.,16,255,"6")
        sRun.sleevedSlipPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(3,3,.005,.5, 0,8.,20.,16,255,"7")
        sRun.sleevedSlipPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(3,3,.005,.5, 0,1.,21.,16,255,"8")
        sRun.sleevedSlipPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(3,3,.005,.5, 0,3,15.,16,255,"9")
        sRun.sleevedSlipPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)


        minWid = .005*limit['NlonPix']/3.
        maxWid = 10.*minWid
        sRun = DepthSlip(1,8,minWid,maxWid, 99,0.,0.,16,255,"0")
        sRun.tdGrayPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(1,8,minWid,maxWid, 0,10.,20.,16,255,"1")
        sRun.tdGrayPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(1,8,minWid,maxWid, 0,20.,40.,16,255,"2")
        sRun.tdGrayPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(1,8,minWid,maxWid, 0,40.,80.,16,255,"3")
        sRun.tdGrayPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(1,8,minWid,maxWid, 0,80.,160.,16,255,"4")
        sRun.tdGrayPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(1,8,minWid,maxWid, 0,8.,40.,16,255,"5")
        sRun.tdGrayPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(1,8,minWid,maxWid, 0,5.,20.,16,255,"6")
        sRun.tdGrayPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(1,8,minWid,maxWid, 0,8.,20.,16,255,"7")
        sRun.tdGrayPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(1,8,minWid,maxWid, 0,1.,21.,16,255,"8")
        sRun.tdGrayPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)
        sRun = DepthSlip(1,8,minWid,maxWid, 0,3.,15.,16,255,"9")
        sRun.tdGrayPlot(maskedEdgeMap,validJTList,validSlipList,
           maskedEdgeMap.shape,limit,dT)

        nowTime = time.time(); print(('Post Scharr time: ',nowTime-startTime))
    
        gX,iScale,iShift = image2outs(gX,'gx',limit,False)
        gY,iScale,iShift = image2outs(gY,'gy',limit,False)
        #i2png(gX, "gX")
        #i2txt(gX, "gX")
        #i2png(gY, "gY")
        #i2txt(gY, "gY")
        # correct gradient to nonsquare pixels:
        # hardwire scaling for now, later from
        # lat2xy,lonDelta, latDelta, NlonPix:
        xSc = 26.
        ySc = 31. # suitable for .00005556 degreee spacing * 5,
        bCross = []
        lonDelta = limit["LonDelta"]
        latDelta = limit["LatDelta"]
        print(("using azimuth from data file:",azimuth))
        for pair in edgeList:
           x,y = pair
           Gx = np.sign(lonDelta)*gX[x,y]/xSc
           Gy = np.sign(latDelta)*gY[x,y]/ySc
           bVal = Gx*geofunc.coso(azimuth)-Gy*geofunc.sino(azimuth)
           bCross.append((x,y,bVal))
        # zero a pair of images, then fill one for bCross > 0., one for < 0.
        posCross = np.zeros((nrow,ncol))
        negCross = np.zeros((nrow,ncol))
        pCount = 0
        nCount = 0
        for item in bCross:
           x,y,b = item
           posCross[x,y] = (b > 0.)
           negCross[x,y] = (b < 0.)
        posCross,iScale,iShift = image2outs(posCross,'posCross',limit,False)
        negCross,iScale,iShift = image2outs(negCross,'negCross',limit,False)

        nowTime = time.time(); print(('Post Cross time: ',nowTime-startTime))
        
        prefix = 'edges'

        nkern = 1
        rootName = prefix + '+1'
        finalImage(nkern,posCross,limit,rootName,'white',dT)

        nkern = 1
        rootName = prefix + '-1'
        finalImage(nkern,negCross,limit,rootName,'black',dT)

        nkern = 3
        rootName = prefix + '+3'
        finalImage(nkern,posCross,limit,rootName,'white',dT)

        nkern = 3
        rootName = prefix + '-3'
        finalImage(nkern,negCross,limit,rootName,'black',dT)


        nkern = 8
        rootName = prefix + "8"
        # do kerning, make background transparent,call image2outs, hence
        #   makeGearthFile
        finalImage(nkern,maskedEdgeMap,limit,rootName,'black',dT)

        nkern = 7
        rootName = prefix + "7"
        finalImage(nkern,maskedEdgeMap,limit,rootName,'black',dT)

        nkern = 6
        rootName = prefix + "6"
        finalImage(nkern,maskedEdgeMap,limit,rootName,'black',dT)

        nkern = 5
        rootName = prefix + "5"
        finalImage(nkern,maskedEdgeMap,limit,rootName,'black',dT)

        nkern = 4
        rootName = prefix + "4"
        finalImage(nkern,maskedEdgeMap,limit,rootName,'black',dT)

        nkern = 3
        rootName = prefix + "3"
        finalImage(nkern,maskedEdgeMap,limit,rootName,'black',dT)

        nkern = 2
        rootName = prefix + "2"
        finalImage(nkern,maskedEdgeMap,limit,rootName,'black',dT)

        nkern = 1
        rootName = prefix + "1"
        finalImage(nkern,maskedEdgeMap,limit,rootName,'black',dT)

        chk = np.asarray(cvFl)
        print(("DBG chk",chk[0:4]))
        del cvScImage

        nowTime = time.time(); print(('Post kern file write time: ',nowTime-startTime))
    
    

        # RUSTY: histogram the data: count[], bin_edge[] define histogram
        count, bin_edge = np.histogram(chk,bins = 64)
        # take cumulative sum; later divide by last element for distribution
        cumul = np.cumsum(count)
        cumtop = cumul[-1]
        cumindex = 0
        print(("DBG types: ",type(bin_edge[0]),type(count[0]),type(cumul[0])))
        hp = open("hist.txt", "w")
        for e, h,c in zip(bin_edge,count,cumul):
           hp.write("%f %f %f\n"%(e, float(h),float(c)))

        hp.close()

        print("ScImage lo mean, hi: ",chk.min(),chk.mean(),chk.max())
        # may be needless extra scaled array here . . .
        print("minmaxloc", cv2.minMaxLoc(cvEdgeMap))
        #cv.SaveImage('edgemap.png', cvEdgeMap)
        #convert to transparent background

        nowTime = time.time(); print(('AllDone: ',nowTime-startTime))

    # handle raised Usage error; typically incorrect args,
    # and typically  prints documentation
    except Usage as err:
        print(sys.argv[0].split("/")[-1] + ": " + str(err.msg), file=sys.stderr)
        print("	 for help use --help", file=sys.stderr)
        return 2

# support expected standalone runs (but can also import as module)
if __name__ == "__main__":
    sys.exit(main()) 
