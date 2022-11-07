# README

This is a tool to make panoramic screenshots on PC.

## Install

You might want to set a new environment since this project depends on OpenCV.

Installing the dependencies:

```
pip install -r requirements.txt
```

Running:

```
python panorama_screenshots.py
```

- Click start to draw the area that will be captured.
- Right-click to confirm the area.
- Left click on the masked region outside the selected area to take a screenshot of the selected area. Move around to take multiple screenshots, which will later be stitched together to make a panoramic picture.
- Right click on the masked region to terminate.
- Click save to generate the panoramic picture, or click cancel to delete all the screenshots you made and exit.
- Wait for a while and find the final result as 0.png in ``./tempPics/``. Might not success if some screenshots youu made do not overlap.
