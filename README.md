PyStab
=======

Jan. 2016, F. Mompiou, CEMES-CNRS.

Largely inspired by the original work of L. Dupuy and insitu7 c++
program (laurent.dupuy@cea.fr).

Requirements
------------

-   Python 2.7 (with Numpy, Matplotlib and OpenCV 3). Can be installed
    through [Enthought Canopy](https://store.enthought.com/downloads/)
    or [Anaconda](http://continuum.io/downloads) distributions for
    instance (multiplatform). For OpenCV installation see the
    [doc](http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html).
    For Ubuntu distribution please refer to this [installation
    procedure](https://help.ubuntu.com/community/OpenCV) for OpenCV 3.
-   ffmpeg (optional). For Windows, see these
    [instructions](http://fr.wikihow.com/installer-FFmpeg-sur-Windows).
    For Ubuntu, see the [doc](https://doc.ubuntu-fr.org/ffmpeg) for
    install.

Use
---

    python pystab.py -i mymovie.avi 

to navigate in the video and draw roi. Use the trackbar and press any
key to navigate in the movie. Left click, drag and release to draw a
rectangle. Press any key to remove; ESC to exit. The x,y,w,h (position
and size) of the roi are printed in the command prompt (can be copied
and paste in the xml file, see below)

    python pystab -s myfile.xml 

to stabilize the video in a specific area (ROI) given a fixed area
(template). Output: a series of jpg pictures of the stabilized movie in
the `/output` directory, and a video preview (optional). If ffmpeg enable, an output movie can be
generated.

The myfile.xml file is of the type:

~~~~ {.xml}
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<root>
  <movie name="zy-interaction.avi" />
  <roi x="79" y="192" w="380" h="291"/>
  <template x="362" y="221" w="95" h="159"/>
  <newmovie start="108" end="500" />
  <spot inpaint="no" x="370" y="272" w="12" h="13" inp="10" 
start="80" end="140" radius="40" />
  <enhance smooth="20" deviation="5" denoise="3" />
  <contrast clahelimit="0.1" clahex="10" clahey="10"/>
  <scalebar mag="60" size="50" />
  <ffmpeg deinterlace="yes" output="zy-interaction-stab.avi" />
  <preview raw="no" stabilize="yes" enhance="yes"/>

</root>
~~~~

Mandatory:

-   `<movie>`: the movie name. Param: `name`

-   `<roi>`, `<template>`: a ROI and a template. Param: `x`,`y`,`w`,`h`
    the position (x,y) and size (width and height)

Optional:

-   `<newmovie>`: define the time interval of the output movie. Param:
    `start` and `end` the starting and ending points. `step` specifies the step between frames (useful for long video)

-   `<spot>`: remove camera imperfections either by subtracting the
    background image obtained during a rapid camera motion (Param:
    `start` and `end` the time interval of the rapid motion, `radius` a
    blurring radius), or by inpainting the spot at the location `x`,`y`
    of width and height `w`,`h` with a radius `inp`. The choice of
    inpainting or background subtraction can be made by setting
    `inpaint` to `"yes"` or `"no"`.

-   `<enhance>`: improve the video image. Param: `denoise`: time
    averaging to remove noise. Smoothing out the image trajectory is
    performed by a x and y running average given a certain time window
    (`smooth`) and a certain tolerance (`deviate`). Output:
    `trajectory.png` showing the x and y motion and the corresponding
    smoothed trajectory.

-   `<contrast>`: improve the contrast by making Contrast Limited
    Adaptive Histogram Equalization. Param: `clalimit` for contrast
    limiting, `clahex` and `clahey` for x, y tile size. See the
    [doc](http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#clahe-contrast-limited-adaptive-histogram-equalization).

-   `<scalebar>`: add a scale. Param: `mag`: image magnification x1000
    (for jeol2010), `size` of the scalebar in nm.

-   `<ffmpeg>`: Export the stabilized/enhanced movie. Param:
    `deinterlace` option enable if `"yes"`. `output`: the video name.

-   `<preview>`: Show a video preview. Param: if `raw`, `stabilize` and `enhance` are set to `"yes"`, the raw video, stabilized (with the template) video and enhance video are displayed.



