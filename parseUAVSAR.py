#!/usr/bin/env python
"""
Experimental: attempt to get centers of 6x6 blocks shifted by multiples of 2x2.
NAME
   parseUAVSAR -- reads UAVSAR files and produces simplex input data lines.
                  and similar correlation file.
                  Output deformation is in mm, in simplex format
      Corr are in range [0.,1.]

SYNOPSIS
   parseUAVSAR.py [-a|-s] [-i|-v] [-q] [-e] 
      [-m nEnvCols] [-n nEnvRows] [-g nCellCols] [-j nCellRows]
      [-k reflon] [-l reflat] [-p opf.polyFile.kml] 
      [-c corr_threshold] [-u spread_threshold] 
      [-o outputFile] <filePrefix> 

DESCRIPTION
   The parseUAVSAR.py script reads radar phase from UAVSAR files ending 
   .unw.grd or .int.grd (complex phase), using matching file ending .ann
   to interpret data pixel locations in lat/lon, and matching file ending
   .cor.grd to allow exclution of low-correlation pixels.  Pixel line-of-sight
   motion (cm) is written to the outputFile in simplex format, 
   one line per pixel. Typical use defines averaging blocks of -jx-g image cells
   surrounded by an environment of -nx-m cells inclusive: the cell is rejected
   if the thresholds are exceeded in this envorinment.

   The following options are available:

   -a    Average blocks of data to a single value (use large pixels)
   
   -s    Alternative to -a, write out the sampled value at the center of 
         each block of data (no averaging): the default when -a not specified.

   -i    Read phase data from file ending .int.grd, convert complex to phase

   -v    Alternative to -i, read phase data from file ending unw.grd

   -q    output complex-valued pixels in (a+bj) format, modified simplex

   -e    To support edge detection, write indices to indx.txt

   -k    specify reference longitude (default is PEG longitude), degrees

   -l    specify reference latitude (default is PEG latitude), degrees

   -m    use block size that is nEnvCols pixels in east-west direction 
         (default is 25)

   -n    use block size that is nEnvRows pixels in north-south direction  
         (default is 25)

   -g    horizontal stride: allow next block to be stride pixels shifted
         (default is -m value)

   -j    vertical stride: allow next block to be stride pixels shifted
         (default is -n value)

   -p    use kml file opf.polyFile.kml to define inclusive polygon. 
         Blocks of data (large pixels) with centers inside polygon will
         be represented in the output.

   -c    disregard pixels or blocks with representative correlation smaller
         than corr_threshold (default 0.25)

   -u    disregard blocks with phase standard deviation greater 
         than spread_threshold (default 1.0 cm)

   -x    combine block masking test -c, -u with "and" (default "or")

   -o    output file name specified to be <outputFile>.  Default is 
         "simplex_input_block.txt"

EXAMPLE:
    parseUAVSAR.py -a -m 6 -n 6 -g 2 -j 2 -c 0.4 -u 0.5 -x (-p xx.kml -e -o a.txt prefix))

    outputs  average of displacements on  2x2 blocks, screens if 6x6 environment has mean corr > 0.4 and std dev < 0.5 cm

MODS IN PROG for complex interferograms:
-a: we don't average phase, but we can average complex values on per-block basis.
-s: can support as-is (complex value, phase, whatever).
-i: no change; the new default, for complex interferograms; not -v
-e: no change (determine edge coords, write to index file).
-k, -l: (reference coords, not PEG): no change
-m, -n: (coherence-consideration block size): no change
-g, -j: (strides): no change; 
-c (coherence threshold): no change.
-u (phase std dev): disregard for now (default high?);
not clear if some complex variance might indicate trouble
-x: disregard (for now): mask with "and"
-o: outputFile must now contain complex-valued info
for complex edge detection operations. Do in standard python format: (234+1451j)

   
OUTPUT: 
    <stdout>: 
       brief diagnostic messages are written to stdout
    
    <outputFile>:
       Simplex input measurement lines written to outputFile indicated by -o; 
       else to file with default name simplex_input_block.txt

       Output format is usual Simplex observation format:
       type   x(km)      y(km)      SAR_LOS   sigma   elevation azimuth
       7   -18.980154 -25.461656 -51.517734 1.000000 27.800303 -5.319699

       Running "simplex -a" is recommended for this data (-a: chisquare is
       computed relative to the observed and computed data average.)
       This accounts for an unknown constant phase offset produced by UAVSAR.
"""

import sys
import getopt
import copy
import xml.dom.minidom
import numpy as np
import geogeo
import Lxy
from setpar import setpar
from daynum2k import daynum2k
import time

SVN_Id = "$Id: parseUAVSAR.py 162 2011-09-16 00:22:17Z jwparker $"

class Usage(Exception):
    def __init__(self, msg):
       self.msg = msg

class Timer:
   def __init__(self):
      self.startTime = time.time()
   def report(self,msg):
      print('TIMER:: ' + msg + ': ',time.time() - self.startTime)
          


def writeArgsToREDO(r_,argv):
   commandline = ""
   for arg in argv: commandline = commandline + arg + ' '
   commandline = commandline + '\n'
   r_.writelines(commandline)

def fileArray(fileName,nLon,nLat,numType,timer):
   """
   Open file and extract a numpy array.

   fileName: name or path of the file to open
   nLon, nLat: the dimensions of the array
   numType: 'F' for float, 'C' for complex64
   """
   with open(fileName,'r') as g_:
      timer.report('\t opening '+fileName)
      if numType == 'F':
         a = np.fromfile(g_,'<f')
      elif numType == 'C':
         a = np.fromfile(g_,'Complex64')
      else:
         raise("Bad numType: "+numType)
      timer.report('\tclosed '+fileName)

      if len(a) != nLat * nLon:
         raise("Bad file length! " + fileName)

      print("\nFrom " + fileName + " numType " + numType)
      print("rows, columns, size:",nLat,nLon,len(a))
      a = np.reshape(a,(nLat,nLon))
      return a

        

class OpFlag:
   """
   Defaults and names for all processing options, beginning with line args
   """
   def __init__(self):
      """
      Set Defaults,  in case user has no -m -n -c -u flags
      """
      self.nEnvCols=25
      self.nEnvRows=25
      self.nEnvRowsSet = False
      self.nEnvColsSet = False
      self.nCellCols = self.nEnvCols # default: env same as cell
      self.nCellRows = self.nEnvRows
      self.nCellColsSet = False
      self.nCellRowsSet = False
      self.corrThresh = 0.25
      self.pSdevThresh = 1.0 # default is one cm (internal values are cm)
      self.useUnwFile = True
      self.useIndx = False
      self.useMid = False 
      self.threshCombine = "Or"
      self.averageBlock = False 
      self.outputFile = "simplexInputBlock.txt"
      self.polyFile = ""
      self.writeComplex = False
      self.inLon = None
      self.inLat = None

   def parseLineArgs(self,opts):
      """
      Process all line arguments, overriding default values as encountered.
      """
      for option, value in opts:
         if option == "-a":
            print("-a",value)
            self.averageBlock = True
         if option == "-d":
            print("-d",value)
            self.useUnwFile = True
         if option == "-i":
            print("-i",value)
            self.useUnwFile = False 
         if option == "-e":
            print("-e")
            self.useIndx = True
         if option == "-q":
            print("-q")
            self.writeComplex = True
         if option == "-k":
            print("-k",value)
            self.inLon = float(value)
         if option == "-l":
            print("-l",value)
            self.inLat = float(value)
         if option == "-m":
            print("-m",value)
            self.nEnvCols = int(value)
            self.nEnvColsSet = True
         if option == "-n":
            print("-n",value)
            self.nEnvRows = int(value)
            self.nEnvRowsSet = True
         if option == "-g":
            print("-g",value)
            self.nCellCols = int(value)
            self.nCellColsSet = True
         if option == "-j":
            print("-j",value)
            self.nCellRows = int(value)
            self.nCellRowsSet = True
         if option == "-p":
            print("-p",value)
            self.polyFile = value
         if option == "-c":
            print("-c",value)
            self.corrThresh = float(value)
         if option == "-s":
            print("-s",value)
            self.averageBlock = False 
         if option == "-u":
            print("-u",value)
            self.pSdevThresh = float(value)
         if option == "-x":
            self.threshCombine = 'And'
         if option == "-v":
            self.useUnwFile = True 
         if option in ("-h", "--help"):
            raise Usage(__doc__)
         if option in ("-o", "--output"):
            print("-o",value)
            self.outputFile = value
         # if -m used but not -g:
         if self.nEnvColsSet and not self.nCellColsSet:
            self.nCellCols = self.nEnvCols
         # if -n  used but not -j:
         if self.nEnvRowsSet and not self.nCellRowsSet:
            self.nCellRows = self.nEnvRows
      #
      # Set subsidiary flags
      #
      self.useMid =  \
         (self.nCellCols != self.nEnvCols or self.nCellRows != self.nEnvRows)
      #
      # Override self.averageBlock if 1x1 pixel blocks: don't waste effort
      #
      if(self.nEnvCols*self.nEnvRows == 1):
         self.averageBlock = False


class Cobox:
   """
   Cobox is the min, max limits for lon and lat defining a box in coord space
   """
   def __init__(self, lonmin,lonmax,latmin,latmax):
      self.lonmin = float(lonmin)
      self.lonmax = float(lonmax)
      self.latmin = float(latmin)
      self.latmax = float(latmax)
   
class LinearStep:
   """
   LinearStep defines a linear sequence by initial, delta, count parameters
   Units are not defined: may be km, lon degrees, etc.
   """
   def __init__(self,x0,xDelta,Nx):
      """
      Input variables: 
      x0 is the initial value
      xDelta is the step size;
      Nx is the number of points defining the steps.
      hence there are Nx - 1 intervals (steps)
      """
      self.x0 = x0
      self.xDelta = xDelta
      self.Nx = Nx
      self.xMax = x0+xDelta*(Nx-1)
   def values(self):
      vals = np.linspace(self.x0,self.xMax,num=self.Nx)
      return vals
      

class CoordGrid:
   """
   CoordGrid defines a rectangular coordinate grid by start, delta, count parameters
   in longitude and latitude
   """
   def __init__(self,lon0,londelta,nlon,lat0,latdelta,nlat):
      """
      Input variables: 
      lon0,lat0 are the origin (may be lon, lat or km E, km N for example)
      londelta, latdelta are the x and y step size (in degrees or km)
      nlon, nlat are the number of samples (hence nlon-1 steps and nlat-1 steps)
      """
      
      # defines lon0, londelta, nlon,lat0,latdelta,nlat,nlonsteps,nlatsteps,lonmax,latmax.
      self.lon0 = lon0
      self.londelta = londelta
      self.nlon = nlon
      self.lat0 = lat0
      self.latdelta = latdelta
      self.nlat = nlat
      self.nlonsteps = nlon-1
      self.nlatsteps = nlat-1
      self.lonmax = lon0+self.nlonsteps*londelta
      self.latmax = lat0+self.nlatsteps*latdelta

   def pg():
      print(("CG: lon0",lon0))
      print(("CG: londelta",londelta))
      print(("CG: nlon",nlon))
      print(("CG: lat0",lat0))
      print(("CG: latdelta",latdelta))
      print(("CG: nlat",nlat))
      print(("CG: nlonstaps",nlon-1))
      print(("CG: nlatstaps",nlat-1))
      print(("CG: lonmax",self.lonmax))
      print(("CG: latmax",self.latmax))

class Grid:
   """
   Grid has elements and functions for a rectangular grid of lon,lat pts
   Includes origin (Peg point on flight path) peg or user input
   Includes data grid description cg, 
   and larger-block subgrid subg used in downsampling
   """
   def __init__(self,annVal,opf):
      """
      All values are determined by the "ann" file or the options
      """
      if opf.inLon == None: refLon = annVal["Peg Longitude"]
      else:                 refLon = opf.inLon

      if opf.inLat == None: refLat = annVal["Peg Latitude"]
      else:                 refLat = opf.inLat

      self.ref = geogeo.Coord(refLon,refLat)

      self.nEnvCols = opf.nEnvCols
      self.nEnvRows = opf.nEnvRows
      self.nCellCols = opf.nCellCols
      self.nCellRows = opf.nCellRows
      # define buffer as half the remainder when we remove cell from env
      # That way env = cell + top buffer + bottom buffer (and left, right)
      self.rowEnvBuf = (self.nEnvRows - self.nCellRows)//2
      self.colEnvBuf = (self.nEnvCols - self.nCellCols)//2
      # this implies that for iRow,jCol as indices into the subgrid
      # the slice of the base array corresponding to the block is
      # iRow*gr.nCellRows:(iRow+1)*gr.nCellRows,\
      # jCol*gr.nCellCols:(jCol+1)*gr.nCellCols
      # which has shape gr.nCellRows,gr.nCellCols

      lonmin = annVal["Ground Range Data Starting Longitude"]
      latmin = annVal["Ground Range Data Starting Latitude"]
      londelta = annVal["Ground Range Data Longitude Spacing"]
      latdelta = annVal["Ground Range Data Latitude Spacing"]
      nlon = annVal['Ground Range Data Longitude Samples']
      nlat = annVal['Ground Range Data Latitude Lines']

      self.cg = CoordGrid(lonmin,londelta,nlon,latmin,latdelta,nlat)

      # subgrid: starts at center of nEnvCols by nEnvRows block at min point
      # and extends for integer number of blocks, ignoring further data
      # Note that this defn of subg implies the cells are lon,lat 
      # not indices or km
      colShiftToCenter = opf.nCellCols//2
      rowShiftToCenter = opf.nCellRows//2
      lonsubmin = lonmin + colShiftToCenter*londelta
      latsubmin = latmin + rowShiftToCenter*latdelta
      lonsubdelta = londelta * opf.nCellCols
      latsubdelta = latdelta * opf.nCellRows
      subnlon = nlon // opf.nCellCols 
      subnlat = nlat // opf.nCellRows 

      self.subg = CoordGrid(lonsubmin,lonsubdelta,subnlon,latsubmin,latsubdelta,subnlat)

   def rrcoord_of_block (self,pt):
      """
      pt is a Coord 
      """
      ref = self.ref
      coords = [pt]
      (rco,jnk1,jnk2)=Lxy.dlat2xy(ref,coords)
      return rco[0]


   def coord_of_block (self,ipair):
      """
      ipair is an index pair - which block is in view. Integer pair.
      """
      ilon,ilat = ipair
      lonpt = self.subg.lon0 + float(ilon)*self.subg.londelta
      latpt = self.subg.lat0 + float(ilat)*self.subg.latdelta
      pt=geogeo.Coord(lonpt,latpt)
      return pt

   def xGrid (self,lat):
      """
      xGrid defines a linear sequence based on cg spacing and 
      a particular (input) latitude "lat"

      It represents the x position values at image points
      on a west to east profile line at the center latitude.
      USAGE:
         lseq,sseq = gr.xGrid(lat)
         xi = lseq.x0+lseq.xdelta*i
         xsubj = sseq.x0+sseq.xdelta*j
      """
      co = [geogeo.Coord(self.cg.lon0, lat),
            geogeo.Coord(self.cg.lon0+1.0, lat)]

      rco,jnk1,jnk2 = Lxy.dlat2xy(self.ref,co)
      partl_x_by_lon = rco[1].x - rco[0].x

      x0 = rco[0].x
      xdelta = self.cg.londelta*partl_x_by_lon
      nx = self.cg.nlon
      lseq = LinearStep(x0,xdelta,nx)

      xsubdelta = xdelta*self.nEnvCols
      xsub0 = x0 + xsubdelta/2.
      nxsub = self.subg.nlon
      sseq = LinearStep(xsub0,xsubdelta,nxsub)
      return lseq,sseq,
       
      
class DataBlock:
   """
   Set of block-reduced statistics from phase and correlation values
   """
   def __init__(self,c,p,pS,pMid,ig,igMid,ll,i,j):
      """
      Simply a structure; might consider adding data points of blocks here.
      c:correlation, p:phase, ig:complex interferogram, ll:latLon, 
      i,j: row, column
      """
      self.c,self.p,self.pS,self.pMid,\
            self.ig,self.igMid,self.ll,self.i,self.j =\
            c, p, pS, pMid, ig, igMid, ll, i, j

def righttest(pt,s):
    """ 
    Does point pt lie in region found to the right of segment s?
    Strategy: find place where pt lat line intersects extension
    of segment s.  If within s lat range, and if pt to right
    of that intersection point, True.
    """
    denom = s[1].lat - s[0].lat
    # Do not consder a horizontal segment.
    if denom == 0.0:
       return False
    # define segment extension as (Coord) s[0](1-t) + s[1])*t
    # where t is in [0,1] inside the segment.
    tinter = (pt.lat-s[0].lat)/denom
    sinterlon = (1.-tinter)*s[0].lon + tinter*s[1].lon
    if (pt.lat - s[0].lat)*(pt.lat - s[1].lat) > 0.0:
        # pt latitude not between s0, s1 lat's
        return False
    if pt.lon - sinterlon < 0: 
        return False
    else:
        return True 

class Segments:
   def __init__(self):
      self.ingested = []

   def ingest(self,coordPair):
      self.ingested.append(coordPair)

   def findLatBounds(self,subg):
      seglat = []
      for seg in self.ingested:
          seglat.append(seg[0].lat)
          seglat.append(seg[1].lat)
      if len(seglat) == 0:
         # provide for no-polygon case
         segtmp = subg.lat0,subg.latmax
         seglat0,seglat1 = min(segtmp),max(segtmp)
      else:
         seglat0,seglat1 = min(seglat),max(seglat)
      return((seglat0,seglat1))

   def kmlIngest(self,pfile):
      """
      Return a list of segments, each a pair of Coords
      Polygon may extend beyond borders of valid interferogram image.
      The polygon must be represented as a single kml file 
         containing ordered xml <coordinate> records specifying vertices.
      Usually  this file is prepared in a prior Google Earth session.
      """

      co = []
      try:
         pf=open(pfile,"r")
      except IOError:
         return
      doc = xml.dom.minidom.parse(pf)
      coorddom = doc.getElementsByTagName("coordinates")
      c = coorddom[0].toxml()
      c = c.replace("\t","")
      c = c.replace("<coordinates>\n","")
      c = c.replace("\n</coordinates>","")
      tripl = c.split()

      for tr in tripl:
         item=tr.split(',')
         co.append(geogeo.Coord(item[0],item[1]))

      for i in range(len(co)-1):
         self.ingest((co[i],co[i+1]))

def maskedStats(theList):
   aArr = np.array(theList)
   maskz = (aArr == 0.0)
   maskn = np.isnan(aArr)
   mask = maskz | maskn
   aMask = aArr[~mask]
   if len(aMask) >=1:
      theValue = aMask.mean()
      theStdev = aMask.std()
   else: 
      theValue = np.nan
      theStdev = np.nan
   return theValue, theStdev

def elmap(xpt,ypt,annVal):
   """
   find elevation angle (from pixel to satellite)
   based on the Peg coordinates , heading, altitude,
   and xpt, ypt (offsets in km)
   """
   import geofunc
   M_PER_KM = 1000.
   
   if "Average GPS Altitude" in annVal:
       aga = annVal["Average GPS Altitude"]
   elif "Global Average Altitude" in annVal:
       aga = annVal["Global Average Altitude"]
   else:
       print("No GPS or global altitude in .ann file!")
       exit(-1)

   ath = annVal["Average Terrain Height"]
   ph = annVal["Peg Heading"]
   # So the average height of the craft above ground is:
   delta_height_in_m = aga - ath
   delta_height_in_km = delta_height_in_m/M_PER_KM
   # Heading unit vector, as x, y (East, North) components:
   hvec = (geofunc.sino(ph),geofunc.coso(ph))
   # Craft peg point is above (0,0); 
   # horizontal part of craft to pixel vector is (xpt,ypt)
   # and so projection is x_ = x_ - p dot h h_
   # (pp is horix. distance from pegged heading line to pixel)
   pdoth = xpt*hvec[0]+ypt*hvec[1]
   pp = (xpt-pdoth*hvec[0],ypt-pdoth*hvec[1])
   magpp = np.sqrt(pp[0]*pp[0]+pp[1]*pp[1])
   # elevation angle from right triangle: ht is delta ht,
   # base is mag pp
   el = geofunc.atano(delta_height_in_km/magpp)
   return el
   
def importMetadataValues(fname):
   """
   Function importMetadataValues:
      fname: filename to open and use to extract values
   """
   aV = {}
   af = open(fname,'r')
   for line in af.readlines():
      aV = setpar('Ground Range Data Latitude Lines',line,aV,'int')
      aV = setpar('Ground Range Data Longitude Samples',line,aV,'int')
      aV = setpar('Ground Range Data Starting Latitude',line,aV,'float')
      aV = setpar('Ground Range Data Starting Longitude',line,aV,'float')
      aV = setpar('Ground Range Data Latitude Spacing',line,aV,'float')
      aV = setpar('Ground Range Data Longitude Spacing',line,aV,'float')
      aV = setpar('Center Wavelength',line,aV,'float')
      aV = setpar('Average GPS Altitude',line,aV,'float')
      aV = setpar('Global Average Altitude',line,aV,'float')
      aV = setpar('Average Terrain Height',line,aV,'float')
      aV = setpar('Peg Latitude',line,aV,'float')
      aV = setpar('Peg Longitude',line,aV,'float')
      aV = setpar('Peg Heading',line,aV,'float')
      aV = setpar('Radar Look Direction',line,aV,'string')
      aV = setpar('Time of Acquisition for Pass 1',line,aV,'string')
      aV = setpar('Time of Acquisition for Pass 2',line,aV,'string')
      aV = setpar('Start Time of Acquisition for Pass 1',line,aV,'string')
      aV = setpar('Start Time of Acquisition for Pass 2',line,aV,'string')
   return aV


def findSubrasterBlocksInPoly(gr,subRaster,segments):
   """
   findSubrasterBlocksInPoly compares the subraster index subRaster 
   with the lon,lat polygon segment[], using the lon, lat Grid gr
   subgrid subg to find block centers inside the polygon.

   Returns: subgrid_index[] list of indices along a latitude line
   List of typically contiguous (or chunk-contiguous) subgrid indices
   corresponding to longitudes, counting londelta's from edge of 
   image file.

   Rule: Longitude for center of subgrid block with subgrid_index ib
   is lon = gr.subg.lon0 + ib*gr.subg.londelta 
   """      
   slat = gr.subg.lat0 + subRaster*gr.subg.latdelta
   cutlon = []
   # find segments intersecting slat
   for s in segments.ingested:
      sdelta = s[1].lat - s[0].lat
      if abs(sdelta) < 1e-8: continue
      h = (slat - s[0].lat)/sdelta
      if h > 0. and h < 1.:
         c = (1.-h)*s[0].lon + h*s[1].lon
         cutlon.append(c)
   cutlon.sort()
   lenc = len(cutlon)
   subgrid_index = []
   if len(segments.ingested) == 0:
      # provide for no-poly case
      subgrid_index = list(range(gr.subg.nlon))
      return subgrid_index
      
   if lenc%2 != 0:
      print("Warning: findSubrasterBlocksInPoly detects polygon not closed")
   for i in range(0,lenc,2):
      b_start = int(np.ceil((cutlon[i] - gr.subg.lon0)/gr.subg.londelta))
      b_end = int(np.ceil((cutlon[i+1] - gr.subg.lon0)/gr.subg.londelta))
      b_start, b_end = max(b_start,0), min(b_end,gr.subg.nlon)
      b_range = list(range(b_start,b_end))
      subgrid_index = subgrid_index + b_range
   return subgrid_index

def fetchRasterStats(gr, corA, unwA, subRaster, segments,opf):
   """ 
   fetchRasterStats determines block slices of corA, unwA
   to compute block statistics including samples and averages 
   for all points in a subgrid raster (index subRaster) that lie in the bounds
   of the polygon bounded by segment[]. 

   Returns dataBlocks values for samples or block centers
   (correlation, phase std dev, phase, coordinate and indices)
   """
   dataBlocks = []
   
   # Find_subr_bounds finds subg indices for sg points in poly for this line
   blockIndexList = findSubrasterBlocksInPoly(gr,subRaster,segments) 
   lenb = len(blockIndexList)
   if lenb == 0:
      return dataBlocks

   # Rule:  gr.subg.latdelta = gr.nEnvRows*gr.cg.latdelta 
   # So center of first subgrid raster subRaster is offset a half-cell 
   # (of subgrid) from cg origin plus shift

   midsample =  gr.nCellRows//2, gr.nCellCols//2
   for blockIndex in blockIndexList:
      #    compute stats: cl, pSdev, ul, co: append to newcl (etc)
      #    NEW compute pMidBlock-based phase mean
      # the slice of the base array corresponding to the block is
      # iRow*gr.nCellRows:(iRow+1)*gr.nCellRows,\
      # jCol*gr.nCellCols:(jCol+1)*gr.nCellCols
      # which has shape gr.nCellRows,gr.nCellCols
      iRow = subRaster
      jCol = blockIndex

      ptLon = gr.subg.lon0 + jCol*gr.subg.londelta
      ptLat = gr.subg.lat0 + iRow*gr.subg.latdelta
      ll = geogeo.Coord(ptLon,ptLat)

      # we need to augment the cell by additional rows, columns above and below
      # so define the buffer (used for above and below, to left and to right)
      cellTop = iRow     * gr.nCellRows
      cellBot = (iRow+1) * gr.nCellRows
      cellLft = jCol     * gr.nCellCols
      cellRgt = (jCol+1) * gr.nCellCols

      envTop = cellTop - gr.rowEnvBuf
      envBot = cellBot + gr.rowEnvBuf
      envLft = cellLft - gr.colEnvBuf
      envRgt = cellRgt + gr.colEnvBuf
      
      rEnvSt,rEnvEnd = envTop,envBot
      cEnvSt,cEnvEnd = envLft,envRgt
      rSt,rEnd = cellTop,cellBot
      cSt,cEnd = cellLft,cellRgt
      cList = corA[rEnvSt:rEnvEnd,cEnvSt:cEnvEnd]
      pList = unwA[rEnvSt:rEnvEnd,cEnvSt:cEnvEnd]
      pMidList = unwA[rSt:rEnd,cSt:cEnd]

      if opf.useUnwFile: 
         ig,igMid = 0,0
         if opf.averageBlock:
            c, dummy = maskedStats(cList)
            p, pS = maskedStats(pList)
            pMid, dummy = maskedStats(pMidList)
         else:
            # overwrirte with midsample
            c = cList[midsample]
            p = pList[midsample]
            pS = 0.
            pMid = p
      else: # complex interferogram case
         if opf.averageBlock:
            c, dummy = maskedStats(cList)
            ig, dummy = maskedStats(iList)
            igMid,dummy = maskedStats(iMidList)
            p = np.atan2(iItem.imag, iItem.real)
            pMid = np.atan2(iMidItem.imag,iMidItem.real)
            # Find std dev from conjuage-based differences.
            # This method avoids branch cuts, but scatter beyond pi 
            # becomes unreliable.
            iSum = 0
            for item in iList:
               conPhase = np.angle(iItem * np.conj(item))
               iSum = iSum + conPhase*conPhase
            pS = np.sqrt(iSum/float(len(iList)-1))
         else:
            c = cList[midsample]
            ig = iList[midsample]
            igMid = iItem
            p = np.atan2(iItem.imag, iItem.real)
            pMid = pItem
      dataBlock = DataBlock(c,p,pS,pMid,ig,igMid,ll,iRow,jCol)
      dataBlocks.append(dataBlock)
      
   return dataBlocks

def analyzeValues(filePrefix,annVal,gr,segments,opf):
   """
   Read from cor and unw files into cb, ub array
   Carve into blocks, find mean, std dev
   
   filePrefix - the base tag of a UAVSAR data set: .ann, .unw.grd, . . .
   annVal - dictionary of metadata for this data set
   gr - Grid object
   segments - instance of an object containing a list of segments 
      of bounding polygon, as Coord pairs
   opf - OpFlag object: parseUAVSAR options controlling this run 

   Return arrays of correlation, phase spread, phase, coordinates, and indices
   """
   
   timer = Timer()

   nLon = gr.cg.nlon
   nLat = gr.cg.nlat

   corA = fileArray(filePrefix + '.cor.grd',nLon,nLat,'F',timer)

   intA,umag  = [],[]
   if(opf.useUnwFile == True):
      unwA = fileArray(filePrefix + '.unw.grd',nLon,nLat,'F',timer)

   else:
      intA = fileArray(file.prefix + '.unw.grd',nLon,nLat,'C',timer)
      unwA = [np.atan2(v.imag,v.real) for v in intA]
      umag = [np.sqrt(v.real*v.real + v.imag*v.imag) for v in intA]

   quaRow = nLat//4
   thrRow = 3*nLat//4
   midRow = nLat//2
   midlat = gr.cg.lat0  +  midRow*gr.cg.londelta
   qualat = gr.cg.lat0  +  midRow*gr.cg.londelta
   thrlat = gr.cg.lat0  +  midRow*gr.cg.londelta
   lseq,sseq = gr.xGrid(midlat)
   qlseq,sseq = gr.xGrid(qualat)
   tlseq,sseq = gr.xGrid(thrlat)

   midXs = lseq.values()
   quaXs = qlseq.values()
   thrXs = tlseq.values()
 
   pfa = open("midline.txt",'w')
   qfa = open("qualine.txt",'w')
   tfa = open("thrline.txt",'w')
   if(opf.useUnwFile == True):
       for xLon,cc,uu in zip(midXs,corA[midRow,:],unwA[midRow,:]):
           pfa.write( "%f %f %f\n"%(xLon,cc,uu) )
       for xLon,cc,uu in zip(quaXs,corA[quaRow,:],unwA[quaRow,:]):
           qfa.write( "%f %f %f\n"%(xLon,cc,uu) )
       for xLon,cc,uu in zip(thrXs,corA[thrRow,:],unwA[thrRow,:]):
           tfa.write( "%f %f %f\n"%(xLon,cc,uu) )
   else:
       for xLon,cc,uu,um,xx,yy in \
              zip(xLon,corA[midRow,:],unwA[midRow,:],umag[midRow,:],aival[midRow,:].real,aival[midRow,:].imag):
           pfa.write( "%f %f %f %f %f %f\n"%(xLon,cc,uu,mm,xx,yy) )

   timer.report('\tpost midline time')

   # block stats: but if not -a want to assign as center(ish) sample
   # at end of this block, want the same items written to unwf; and same lists returned.
   unwf = open("unwcor.txt","w")
   seglat0,seglat1 = segments.findLatBounds(gr.subg)
      
   delta = gr.subg.latdelta
   if delta < 0:
      # indices traverse latitudes in reverse order
      # first_seglat here means lat with lowest index number
      first_seglat,last_seglat = seglat1,seglat0
   else:
      first_seglat,last_seglat = seglat0,seglat1
   first_subraster = int(np.ceil((first_seglat-gr.subg.lat0)/delta)) 
   last_subraster = int(np.ceil((last_seglat-gr.subg.lat0)/delta)) 

   dataBlocks = []
   for subRaster in range(first_subraster,last_subraster):
      if subRaster < 0: continue

      # Analyze all the blocks in this subraster.
      rasterDataBlocks = fetchRasterStats(gr,corA,unwA,subRaster, segments,opf)

      if(gr.nEnvCols > 1 or gr.nEnvRows > 1):
         for dB in rasterDataBlocks:
            unwf.writelines("%f %f\n"%(dB.pS,dB.c))

      # "+" here denotes the list join operation:
      # iRow, jCol have been given us as indices of the mxn blocks.
      # Convert to represent the gxj blocks
      #if gr.nCellCols != gr.nEnvCols or gr.nCellRows != gr.nEnvRows:
      #   hMult = gr.nEnvCols // gr.nCellCols
      #   vMult = gr.nEnvRows // gr.nCellRows
      #   for dB in rasterDataBlocks:
      #      dB.j = hMult * dB.j 
      #   for dB in rasterDataBlocks:
      #      dB.i = vMult * dB.i
      dataBlocks.extend(rasterDataBlocks)

   return dataBlocks

def createSimplexBlock(dataBlocks, annVal, gr, opf):

   """
   When block mean cor and unwrapped phase std dev meet
   threshold criteria, create a type 7 line of simplex input.
   Make point of best correlation the relative (type -7) point.
   """

   alllines = []
   # determine phase sign based on aquisition time order
   t1 = annVal['Time of Acquisition for Pass 1'].split()[0]
   t2 = annVal['Time of Acquisition for Pass 2'].split()[0]
   if len(t1.split('-')) < 3:
      t1 = annVal['Start Time of Acquisition for Pass 1'].split()[0]
   if len(t2.split('-')) < 3:
      t2 = annVal['Start Time of Acquisition for Pass 2'].split()[0]

   tdiff = daynum2k(t1)-daynum2k(t2)
   if tdiff < 0:
       phaseSign = -1.0
   else:
       phaseSign = 1.0
   print("Phasesign: ",phaseSign)

   ccount = 0
   ucount = 0
   ulcount = 0
   # Items in isum.txt:
   # NlatPix: vertical (Lat) stride, typ 3 or 6, from  commandline -j arg
   # NlonPix: horizontal (Lon) stride, typ 3 or 6, from  commandline -g arg
   # MinIx,MaxIx,MinIy,MaxIy: box limits of input data, subgrid indices 
   # LonMin,LonDelta,LatMin,LatDelta, Nlon,Nlat: subgrid coordinate parameters 
   # GridX, GridY: subgrid spacing in flat-earth at ref, in km
   # RefLon, RefLat: the grid reference point (Peg or supplied -k -l)
   if opf.useIndx:
      ix = open("indx.txt","w")
      # isum has summary: min, max ix, iy
      isum = open("isum.txt","w")
      if gr.nEnvRows == gr.nCellRows:
         isum.writelines("NlatPix %d\n"%(gr.nEnvRows))
      else:
         corrnLatPix = gr.nCellRows
         isum.writelines("NlatPix %d\n"%(corrnLatPix))

      if gr.nEnvCols == gr.nCellCols:
         isum.writelines("NlonPix %d\n"%(gr.nEnvCols))
      else:
         corrnLonPix = gr.nCellCols
         isum.writelines("NlonPix %d\n"%(corrnLonPix))

      iRowList,jColList = [],[]
      for dB in dataBlocks:
         iRowList.append(dB.i)
         jColList.append(dB.j)
      isum.writelines("MinIx %d\n"%(min(jColList)))
      isum.writelines("MaxIx %d\n"%(max(jColList)))
      isum.writelines("MinIy %d\n"%(min(iRowList)))
      isum.writelines("MaxIy %d\n"%(max(iRowList)))
      isum.writelines("LonMin %.14f\n"%( gr.cg.lon0 + gr.cg.londelta*gr.nCellCols/2.))
      isum.writelines("LatMin %.14f\n"%( gr.cg.lat0 + gr.cg.latdelta*gr.nCellRows/2.))
      isum.writelines("LonDelta %.14f\n"%(gr.cg.londelta*gr.nCellCols))
      isum.writelines("LatDelta %.14f\n"%(gr.cg.latdelta*gr.nCellRows))
      isum.writelines("Nlon %d\n"%(gr.cg.nlon/gr.nCellCols))
      isum.writelines("Nlat %d\n"%(gr.cg.nlat/gr.nCellRows))
      dCoord = geogeo.Coord(  gr.cg.londelta*gr.nCellCols, 
                              gr.cg.latdelta*gr.nCellRows )
      pCoord = geogeo.Coord( gr.ref.lon+dCoord.lon, gr.ref.lat+dCoord.lat )
      
      (XYdeltas,jnk1,jnk2) = Lxy.dlat2xy(gr.ref,[pCoord])
      gridX, gridY = XYdeltas[0].x,XYdeltas[0].y
      isum.writelines("GridX %.14f\n"%(gridX))
      isum.writelines("GridY %.14f\n"%(gridY))
      # Wavelength, obs in cm according to current *.ann
      waveLn = annVal["Center Wavelength"]
      conv = 10. #cm to mm
      # phase is proportional to displacement in mm; 
      # such that phase = 2*pi corresponds to  displ = waveLn(mm) / 2
      # so phase/2pi = 2*displ/waveLn(mm)
      # so phase = displ/(2*waveLn(mm)
      # and displ = 2*waveLn(mm)*phase
      phase2mm = conv*phaseSign*waveLn/(4*np.pi)
      isum.writelines("Phase2mm %.14f\n"%(phase2mm))
      isum.writelines("RefLon %.14f\n"%(gr.ref.lon))
      isum.writelines("RefLat %.14f\n"%(gr.ref.lat))

   # may want exfile.txt covered by a flag as well - later.
   ex = open("exfile.txt","w")
   crfi = open("corfile.txt","w")
   # count the rejects from flags -c -u
   clCount = 0
   pSdevCount = 0
   
   for dB in dataBlocks:
         #if block meets corr, phase tests: write to simplex observation block.  
         # I believe this will filter out zero-length blocks, which

         if opf.useMid: statBase = dB.pMid
         else: statBase = dB.p
         #DBGLINE
         #if abs(statBase ) < 1e-8: 
         #    print("DBG wow, that's small:",statBase)
         if dB.c < opf.corrThresh:
             clCount += 1
         if dB.pS > opf.pSdevThresh: 
             pSdevCount += 1

         if (dB.c > opf.corrThresh): 
            if  not opf.useUnwFile or dB.pS < opf.pSdevThresh: 
               if not np.isnan(statBase):
                 #if not (-1.e-10 < statBase < 1.e-10):
                 if not statBase == 0.00:
                  rco = gr.rrcoord_of_block(dB.ll)
                  if opf.useIndx:
                     ix.writelines("%d %d\n"%(dB.j,dB.i)) #may want to reverse these all through edgar

                  # az is FROM pixel TO radar.
                  if annVal["Radar Look Direction"] == "Left":
                     az = annVal["Peg Heading"] + 90.0
                  else:
                     az = annVal["Peg Heading"] - 90.0

                  # elmap uses height difference and map vector
                  el = elmap(rco.x,rco.y,annVal)

                  # Wavelength, obs in cm according to current *.ann
                  waveLn = annVal["Center Wavelength"]

                  if opf.writeComplex: 
                     if opf.useMid: obs = dB.iMid
                     else: obs = db.i
                     sig = 1.0
                     strObs = str(obs)
                     line = "7 %f %f %s %f %f %f"%(rco.x,rco.y,strObs,sig,el,az)
                  else:    
                     if opf.useMid: phas = dB.pMid
                     else:           phas = dB.p
                     obs = phaseSign*phas*waveLn/(4.*np.pi)

                  # Assign 1 cm sigma based on Baja coseismic fit residuals std dev
                     sig = 1.0
                     conv = 10. # cm to mm
                     line = "7 %f %f %f %f %f %f"%(rco.x,rco.y,obs*conv,sig*conv,el,az)

                  alllines.append(line)

   print(("DIAGONSTIC: clCount, pSdevCount = ",clCount,pSdevCount))
   # for corfile, want no gaps
   for dB in dataBlocks:
   #for i in range(len(cList)):
         obs = dB.c
         linex = "%f %f %f"%(dB.ll.lon,dB.ll.lat,obs)
         crfi.writelines("%s\n"%(linex,))

   # for exfile, want no gaps
   for dB in dataBlocks:
   #for i in range(len(cList)):
         rco = gr.rrcoord_of_block(dB.ll)
         #if block meets corr, u testswrite to simplex observation block.  
          
         if annVal["Radar Look Direction"] == "Left":
            az = annVal["Peg Heading"] + 90.0
         else:
            az = annVal["Peg Heading"] - 90.0
         # elmap uses height difference and map vector
         el = elmap(rco.x,rco.y,annVal)
         # Wavelength, obs in cm according to current *.ann
         waveLn = annVal["Center Wavelength"]

         if gr.nCellCols != gr.nEnvCols or gr.nCellRows != gr.nEnvRows:
            obs = phaseSign * dB.pMid * waveLn/(4. * np.pi)
         else:
            obs = phaseSign * dB.p    * waveLn/(4. * np.pi)

         sig = 1.0
         conv = 10. # cm to mm
         linex = "%f %f %f %f %f %f %f %f"%(dB.ll.lon,dB.ll.lat,obs*conv,sig*conv,el,az,rco.x,rco.y)
         ex.writelines("%s\n"%(linex,))


   # provide for zero-point case:
   if len(alllines) == 0:
       print("No points meet all critera - halting.")
       sys.exit(-1)

   outf = open(opf.outputFile,"w")
   for line in alllines:
       outf.writelines("%s\n"%(line,))

def main(argv=None):

    timer = Timer()

    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "ac:eg:hij:k:l:m:n:o:p:qsu:vx", ["help", ])
        except getopt.error as msg:
             raise Usage(msg)

        # Always one arg left: the radar tag, used as a file root with suffixes
        if len(args) != 1: 
           raise Usage(__doc__)
        filePrefix = args[0]

        # Process the options, set option flags
        opf = OpFlag()
        opf.parseLineArgs(opts)

        # Write line commands to REDO file: repeat run via "source REDO"
        # Comes AFTER len(args) test: don't overwrite REDO file if a bad run
        with open('REDO.parseUAVSAR','w') as r_: 
           writeArgsToREDO(r_,argv)

        # Read polygon segments: either none, or closed polygon.
        # Note Segments is an object that holds the ingested segment list
        segments = Segments()
        segments.kmlIngest(opf.polyFile)
        
        # Read metadata into dictionary
        annVal = importMetadataValues(filePrefix + '.ann')
        
        timer.report('\t preliminaries')

        # Define pixel grid, subgrid (gr and gr.subg) 
        # mapping from h,v pixel indices to lat, lon
        gr = Grid(annVal,opf)

        # Read interferometry products and create subgrid summary
        dataBlocks = analyzeValues(filePrefix,annVal,gr,segments,opf)

        timer.report('##method analyzeValues DONE')

        # Write out in Simplex data block format
        createSimplexBlock(dataBlocks, annVal, gr, opf)

        timer.report('##method createSimplexBlock DONE')
    except Usage as err:
        print(sys.argv[0].split("/")[-1] + ": " + str(err.msg), file=sys.stderr)
        print("	 for help use --help", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main()) 

"""
   # For stride not block case, this index is over the large blocks
   jCol = []
   iRow = []
   for blockI in blockIndexList:
      # coord follows from subg definitions; note lon0, lat0 may be shifted
      # from original origin
      lonLatItem = geogeo.Coord(gr.subg.lon0 + blockI*gr.subg.londelta,
               gr.subg.lat0 + subRaster*gr.subg.latdelta)
      lonLat.append(lonLatItem)
      # Note these indices are for the m, n blocks
      jCol.append(blockI)
      iRow.append(subRaster)
"""      
