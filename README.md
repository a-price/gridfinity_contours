# SVG Outliner

TODO: use getAffineTransform and warpAffine with the 3 circles to transform the image after find_contours, before find_box

A Python application that converts images of objects on a black background to SVG outlines, with interactive circle detection and boundary selection.

## Features

- Load and display images
- Detect circles using Hough Transform with adjustable parameters
- Interactive selection/deselection of circles
- Manual boundary point selection
- Export to SVG format
- Real-time preview of detected circles and boundaries

## Installation

1. Install pyenv and virtualenv on your workstation.
2. Install python 3.11 via `pyenv install 3.11`.
3. Create a virtual environment: `pyenv virtualenv 3.11 gridfinity_contours`
4. Activate the virtual environment: `pyenv activate gridfinity_contours`
5. Install the required packages: `pip install -r requirements.txt`

## Usage

1. Run the application:
   ```
   python solid.py
   ```

2. Click "Load Image" to load an image file
3. Adjust the circle detection parameters as needed
4. Click "Detect Circles" to find circles in the image
5. Click on circles to select/deselect them (green = selected, red = unselected)
6. Click on the image to add boundary points (blue)
7. Click "Export SVG" to save the result as an SVG file

## Controls

- **Min Distance**: Minimum distance between circle centers
- **Param1 (edge detection)**: Higher values detect more circular edges
- **Param2 (threshold)**: Lower values detect more circles (including false positives)
- **Min/Max Radius**: Range of circle radii to detect
