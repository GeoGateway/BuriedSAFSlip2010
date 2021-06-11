# BuriedSAFSlip2010
This BuriedSAFSlip2010 repository contains the data and software that supports the Earth and Space Science 2021 article Buried Aseismic Slip and Off-Fault Deformation on the Southernmost San Andreas Fault triggered by the 2010 El Mayor Cucapah Earthquake revealed by UAVSAR by Jay Parker, Andrea Donnellan, Roger Bilham, Lisa Grant Ludwig, Jun Wang, Marlon Pierce, Nicholas Mowery, and Susanne Janecke.

GNSS data for for Figures 1-2 and Table 2 are generated from the Jet Propulsion Laboratory GNSS Time Series public repository at 

https://sideshow.jpl.nasa.gov/post/series.html

using the GeoGateway science portal at https://geo-gateway.org (GNSS tab).

 Typical use of the software requires download of UAVSAR data files from the Alaska Satellite facility, for example for the line SanAnd_26514_09015-001_10028-005_0354d_s01_L090HH_01: download to a fresh directory the three files

http://uavsar.asfdaac.alaska.edu/UA_SanAnd_26514_09015-001_10028-005_0354d_s01_L090_01/SanAnd_26514_09015-001_10028-005_0354d_s01_L090HH_01.ann
http://uavsar.asfdaac.alaska.edu/UA_SanAnd_26514_09015-001_10028-005_0354d_s01_L090_01/SanAnd_26514_09015-001_10028-005_0354d_s01_L090HH_01.unw.grd
http://uavsar.asfdaac.alaska.edu/UA_SanAnd_26514_09015-001_10028-005_0354d_s01_L090_01/SanAnd_26514_09015-001_10028-005_0354d_s01_L090HH_01.cor.grd

The chief analysis scripts are parseUAVSAR.py and edgar3.3.py, described here in a Unix (macOSX) environment.  Several open-source common python libraries must be resident in the system, pre-loaded by the authors using the Anaconda distribution system for Python (v3.7.8), with all compuations performed on a Macbook Pro computer in the Terminal application.

Create and descend to a fresh subdirectory for the parseUAVSAR.py stage of the processing (cleaning and downsampling the unwrapped phase data). This is invoked as follows for the data generated for this article:

$ parseUAVSAR.py -e -a -m 18 -n 18 -g 6 -j 6 -c 0.3 -u 0.7 -o SS.txt ../SanAnd_26514_09015-001_10028-005_0354d_s01_L090HH_01

This generates local output files used for edge detection: SS.txt, indx.txt, isum.txt.

For the second stage create and descend to a fresh subdirectory.  Edge detection products and estimates of gradient direction, width and slip for this article are generated by the command

$ edgar3.3.py -a 5 -m 5 -r 0.75 -s 18 ../SS.txt ../indx.txt ../isum.txt

Selected output files and data generated by GeoGateway form the basis of most of the figures for the article, as follows:

Figures 1-2: GNSS station coordinates are for stations selected using the GeoGateway science portal, GNSS tab, with parameters 
Displacement (mode)
(bounding box):
Latitude: 33.2525
Longitude: -115.7190
Width: 2.1262
Height: 1.5113

Epoch 1: 2009-04-24
Epoch 2: 2010-04-13
Ref. Site CACT
Scale: 160
Av Win 1: 10 days
Av Win 2: 10 days
Output Prefix: CACTTry2

producing output files:
2CACTtry2_table.txt
2CACTtry2_horizontal.kml
2CACTtry2_vertical.kml

The GNSS monument locations in Figure 1 and the horizontal and Figure 2 b&c vertical displacement plots are directly created from these files.
Table 2:  The first three rows (Coseismic) and fifth row (velocity) come from JPL GNSS Time Series tables https://sideshow.jpl.nasa.gov/post/tables/table3.html  (which records times, amplitudes, and undertainty for three component time series breaks) and https://sideshow.jpl.nasa.gov/post/tables/table2.html (which records station velocities derived from time series with breaks removed).   Row 3 results from differencing the station displacements from the GeoGateway output file 2CACTtry2_table.txt, produced as described above.

Figure 3: Grayscale interferogram backgrounds are zoomed selections from the 26514 line dataset, edgar3.3.py product file patchDS.kmz, representing downsampled values of SanAnd_26514_09015-001_10028-005_0354d_s01_L090HH_01.unw.grd unwrapped phase converted to line-of-sight displacement.  In Figure 3a we have enhanced theh contrast to show detail. Figure 3c displays fault transverse samples at the coordinates given.

Figure 4: The orange-pink background displays the UAVSAR unwrapped phase image found in http://uavsar.asfdaac.alaska.edu/UA_SanAnd_26514_09015-001_10028-005_0354d_s01_L090_01/SanAnd_26514_09015-001_10028-005_0354d_s01_L090HH_01.unw.kmz.  The Figure 4a black/white linework displays edgar3.3.py files edges+3V.kmz (white) and edges-3V.kmz (black).  The Figure 4b heat-scale colored linework displays edgar3.3.py output file SlSlip2V.kmz.  

Figure 5: Values of width and slip are from (1-based) columns 6 and 4 of edgar3.3.py output file jumpTable2.txt.  In 5a w80 is column 6 (width in km) * 1000 (conversion to m) * 6 (relation of w to w80).  In the same file lon,lat locations are in the same file columns 10,11, and converted to Distance from Reference Point using simple spread-sheet conversion. The profiles in Figure 5c were extracted from the unwrapped interferogram using the GeoGateway UAVSAR Line-of-Sight tool, repeatedly setting the end coordinates at +/- 1 km positions perpendicular to the mean fault location and strike in the region A-B shown in Figure 5d.  The GeoGateway tool extracts pixels crossed by the profile line, with no subsampling or smoothing.

Figure 6: The yellow-blue background displays the UAVSAR unwrapped phase image found in http://uavsar.asfdaac.alaska.edu/UA_SanAnd_26516_09015-010_10028-007_0354d_s01_L090_01/SanAnd_26516_09015-010_10028-007_0354d_s01_L090HH_01.unw.grd. The Figure 6a black/white linework displays edgar3.3.py files edges+3V.kmz (white) and edges-3V.kmz (black).  The Figure 6b heat-scale colored linework displays edgar3.3.py output file SlSlip2V.kmz.  The grayscale background for Figure 6c is the edgar3.3.py output file smoothedDS.kmz. The blue fault traces are from the qfaults.kmz file as indicated in the figure caption.

Figure 7: Values of width and slip are from (1-based) columns 6 and 4 of edgar3.3.py output file jumpTable2.txt.

Figure 8: Heatscale colored detected edges are from edgar3.3.py output file SlSlip2V.kmz.  Fault traces are from qfaults.kmz and transcriptions from Bryant (2012,2015).

Figure 9: Values for width and slip are simply derived from columns 6 and 4 of edgar3.3.py output file jumpTable2.txt, combining SAF-local edge detection data from radar lines 26514, 21516.  Slip is converted to dextral using the cosine relation that relies on the local strike (column 3 is the phase gradient direction in degrees, strike is (column 3) + 90) and elevation angle (column 7).

Figure 10: spreadsheet calculation based on standard faulted elastic half-space, locked above 6.3 m depth.


