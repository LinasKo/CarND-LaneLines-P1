# Finding Lane Lines on the Road
*Writeup v1.0*

Apologies for the mess - I intend to redo the writeup later, including both pictures and images of the project, a README for Github (or GitLab) as well as nicer writing style. For now, if you wish to see project imagery, please have a look at the Jupyter Notebook.

[Project Repository](https://github.com/LinasKo/CarND-LaneLines-P1)

## Goals
Officially, the project consists of several base goals set by Udacity. These involve:

* Making a pipeline that finds road lane lines in images.
* Extending the pipeline to find lines in videos.
* Writing up the project, including project steps, descriptions of the resulting image pipeline, and a discussion about potential downsides and improvements to the system. 

However, I have set myself some additional goals.

* Seek out new ways that could improve my working efficiency when using Jupyter Notebooks.
* Use both new knowledge from the course, and other skills to succeed in th optional challenge.
* Develop the system in a crystal-clear way, writing the code efficiently, understandably and maintainably. This will help update the project later on and present it on my website.

---

## Pipeline Description
From the very start, my goal was to compose the pipeline in a clear, maintainable way. I started off with a set of steps to extract lane lines from the images, and added additional components to it, to make it more robust. Ultimately, this allowed me to estimate the lines in the challenge video fairly clearly.

Since the components of the basic pipeline was suggested in the lectures, as well as the task description, I will not explicitly describe it. Instead, below you shall find all steps of the final pipeline that I used to obtain lane lines in all of the videos.

#### 1. Input
``` python
def process_img(img, line_history, line_equation_history, history_length):
```
My algorithm is applied as a filter on all frames of a video. Therefore, I take in one frame. Also, I take in a couple of empty python lists, to keep as history. These will be used in step 7 of the pipeline.

### 2. Thresholding
``` python
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = threshold(gray)
```
I convert the image to grayscale as I do not care much about colors. The yellow lines are mostly bright enough to appear as if they are white, when in grayscale. I threshold the image using a simple pixel brightness threshold.

Note that here I explicitly chose to define the parameters (e.g. threshold values) inside the component functions to keep the resulting code easier to read. You shall see this style applied in most parts of my code for this project.

### 3. Apply lane mask
``` python
	mask_binary = make_lane_mask(thresh.shape)
    masked_thresh = cv2.bitwise_and(thresh, mask_binary)
```
For this project I only need to consider the lane ahead. Inside `make_lane_mask` I define trapezoid mask, roughly spanning from the left side to the right on the bottom, and a small length in the horizon:
``` python
    top_left  = [int(shape[1] * 0.45), int(shape[0] * 0.59)]
    top_right = [int(shape[1] * 0.54), int(shape[0] * 0.59)]
    bot_right = [int(shape[1] * 0.95), int(shape[0] -1)]
    bot_left  = [int(shape[1] * 0.12), int(shape[0] -1)]
    vertices = np.array([[top_left, top_right, bot_right, bot_left]], dtype=np.int32)

```

The vertex locations are hand picked. I opened an arbitrary image editor, moused over the points that I wanted the trapezoid vertices to be, and noted down the coordinates. This later allowed me to calculate the offsets, relative to the size of the image.

### 4. Compute Hough Lines
``` python
	lines = hough_lines(masked_thresh)
```
Here, the code is very similar to the helper function, given to us at the start. `cv2.HoughLinesP` is used to compute the lines that are then returned.

### 5. Find Equations of all lines
``` python
	line_equations = lines_to_line_equations(lines)
```
Now, for future computations, I found the equations of the lines. the line equation is written as `y = mx + b`, where `m` is the slope and `b` is the constant factor. For this, I used the slope equation `m = (y2-y1) / (x2-x1)` and `b = y1 - m*x1`.

The only complication was vertical lines. When that happens, `(x2 - x1)` becomes zero, producing an error when computing the slope. I made sure that when `x2 == x1`, I add a small value `ɛ` to `x2` before computing the slope `m`. By trial and error I verified that I tend to not get any overflow errors when `ɛ = 1e-3`.

Therefore, the `line_equations` is simply an array that contains pairs `[m, b]`.

### 6. First outlier removal
``` python
    inlier_indices = find_inliers_by_bottom_edge_crossing(line_equations, gray.shape)
    line_equations = line_equations[inlier_indices]
    lines = lines[inlier_indices]
```
I have observed that in some more difficult cases I get lines that can't realistically be lane lines - they are almost parallel to the horizon. Since I assumed that the vehicle in this task will always drive between two lane lines, I find ranges of `x` values near the bottom of the screen, that are likely locations of lane lines.

I define two ranges around the bottom left and bottom right corners of the view mask trapezoid, defined in step 3. Then, I compute the `x` at which the line crosses the bottom of the image and discard the lines where `x` does not fall into the defined regions.

This type of outlier removal had a very significant positive effect on the accuracy of the line detection. 

### 7. Append to History
``` python
    line_history.append(lines)
    line_history = line_history[-history_length:]
    line_equation_history.append(line_equations)
    line_equation_history = line_equation_history[-history_length:]    
    lines = np.vstack(line_history)
    line_equations = np.vstack(line_equation_history)
```
At this point, I still have many lines of different equations, and not two neat lines, for both lanes. The lines are jittery and not robust to ground texture changes, as seen in the challenge video.

The simplest way to fix this is to track historical data. This will surely reduce jitteriness, at the expense of slower reaction speed to new data.

Here, I have decided that a reasonable history length is 10 frames, which, given that the frame rate of the videos is 25 fps, corresponds to a 400ms window from which the history is kept.

### 8. Second outlier removal
``` python
    w_line_equations, mean, std = whiten(line_equations)  # w_abc signifies that abc is whitened
    w_centroids = cluster_lines(w_line_equations)
    inlier_indices = find_inliers_by_centroid_distance(w_line_equations, w_centroids)
    line_equations = line_equations[inlier_indices]
    lines = lines[inlier_indices]
```
To cluster the lines into two groups, I can use **K-means**, a clustering method for finding centroids of data clusters (fairly simple, though a bit too long to be explain here). But instead of averaging the lines immediately, I spot a way to remove other outliers.

I plotted all line equations `[[m1, b1], [m2, b2], ...]` from one frame, as well as one run, and spotted that I can visually distinguish two clusters of data, and that there are some outliers that do not belong ot any of the clusters.

I then ran `k-means` to find the centroids of these two clusters. Note, that I whitened the data beforehand, as the distances between the slopes of the lines `m` are much lower than between the constant factors `b`. Then, I removed 10% of the data that is the farthest from any of the two clusters. This way, I got rid of the data that fits my model the least.

The advantage of this is that I can now cluster again, for even better results. However, this also means that if at some point a non-lane line is detected very strongly detected (many lines passing through), this method would destroy the correct data that I have. Still, in this case it does not seem likely.

### 9. Average lines
``` python
    centroids = cluster_lines(line_equations)
```
Having removed outliers from the historical data, I average the lines using the same `k-means` method. I fully expect the median to work much better than the mean, but due to time constraints I've decided to just stick with what I have already. 

### 10. Extend line equations into lines
``` python
    long_lines = line_equations_to_lines(centroids, gray.shape)
```
At this point I devised a way to convert the line equations back into segments. I intended to use it for a smarter averaging system (e.g. using weights for historical data), but ultimately did not manage to implement it correctly. You can find the code for it commented out in the notebook. 

Still, here I perform the conversion, so I could use the data for testing if I wish.

### 11. Overlay found lines
``` python
	line_img = np.zeros_like(img)
    if extrapolate:
        draw_extrapolated_lines(line_img, centroids)
        mask_color = make_lane_mask(line_img.shape)
        line_img = cv2.bitwise_and(line_img, mask_color)
    else:
        draw_lines(line_img, long_lines)

    out_img = weighted_img(line_img, img)
    return out_img
```
Lastly, I overlay the results onto the image. There's not much to say here except that the trapezoid mask needs to be applied onto the extrapolated lines, as they span the whole screen, in my drawing implementation.

### Results
Ultimately, I believe the pipeline fairly accurately classifies the challenge video, while also displaying excellent performance on all other provided videos.

I haven't found time to test my pipeline on other datasets, but I expect my results to hold, as long as my assumptions are satisfied. 

## Shortcomings
My method of extracting the lane lines is far from perfect. Off the top of my head, I can identify the following:

* It only considers one lane. The detector will not work when changing lanes or when further lanes need to be detected.
* I assume that the vehicle is between two lanes. When that is not the case, lane data from video could be ignored (as an outlier), the other lane can go out of detection mask area or another road feature could be detected as the lane (due to clustering expecting exactly 2 clusters).
* I assume that both lanes that I can see are fairly straight. If the car makes a sharp turn, the Hough Lines would give out significantly more noise.
* The view mask assumes a certain camera placement and angle. The constrains might not hold if the camera location or orientation on the vehicle is changed.
* The Algorithm can take up to 400 ms to respond to a change in irregular lane line data, due to my chosen history length.
* The found lines are straight and do not bend to fit the curvature of the road.

## Possible improvements to your pipeline
* There has to be a way to figure out if the road is turning. One could attempt to find curved lines, use the known angle of the camera to estimate the center of horizon, scan each individual patch of the line, etc.
* Function of determining the best line equation for each side now uses mean. It might be better to incorporate median instead. I have seen improved results from using median in other projects, instead of the mean.
* I have not used the Canny Edge Detector, nor blurred the image. I have tested both, but did not see the improvement over just using thresholding. This is especially true when the lines are striped and have large gaps between each patch. Still, there must be a way to use the edge information to improve the algorithm - maybe there's a way to join the two?
* I wonder if I could use image features to estimate motion or lines, e.g. SIFT, FAST.

## Completion of my Goals
I am happy to say that I have succeeded in the goals that I have set for myself. I am satisfied with the quality of the code I wrote, and my approach to writing it. I do think that I have managed to complete the optional challenge robustly.

Lastly, I have learned more about Jupyter! I have used a new *Nbextension* called `Execution Dependencies` that automatically executes cells that I tag as prerequisites to others. This has considerably sped up my work, but I think I still prefer to work with proper code IDEs. Maybe next time I'll try JupyterLab :)
