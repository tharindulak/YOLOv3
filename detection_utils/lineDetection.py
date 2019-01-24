import numpy as np
import cv2

results = []

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_detect_lines(img, lines, color=[0, 0, 255], thickness=3):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is not None:
        newLines = sorted(lines, key=getKey)
        tempLine = list(newLines)

        minTuple = (newLines[0][0][0], newLines[0][0][1], newLines[0][0][2], newLines[0][0][3])
        maxTuple = (newLines[0][0][0], newLines[0][0][1], newLines[0][0][2], newLines[0][0][3])

        tempLine.pop(0)
        lineStepTresh = 10
        i = 0
        isDashed = False
        if newLines is not None:
            for line in newLines:
                if i < len(tempLine):
                    if ((line[0][0] - lineStepTresh) <= tempLine[i][0][0] <= (line[0][0] + lineStepTresh)) and ((line[0][1] - lineStepTresh) <= tempLine[i][0][1] <= (line[0][1] + lineStepTresh)) and ((line[0][2] - lineStepTresh) <= tempLine[i][0][2] <= (line[0][2] + lineStepTresh)) and ((line[0][3] - lineStepTresh) <= tempLine[i][0][3] <= (line[0][3] + lineStepTresh)):
                        isDashed = True
                for x1,y1,x2,y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                    if i < len(tempLine):
                        if (tempLine[i][0][1] < minTuple[1]) :
                           minTuple = (tempLine[i][0][0], tempLine[i][0][1], tempLine[i][0][2], tempLine[i][0][3])
                        if (tempLine[i][0][3] > maxTuple[3]) :
                           maxTuple = (tempLine[i][0][0], tempLine[i][0][1], tempLine[i][0][2], tempLine[i][0][3])
                i = i + 1
            finalLine = (int((minTuple[0] + maxTuple[0])/2), int((minTuple[1] + maxTuple[1])/2), int((minTuple[2] + maxTuple[2])/2), int((minTuple[3] + maxTuple[3])/2))
            color = [0, 255, 0]
            cv2.line(img, (finalLine[0], finalLine[1]), (finalLine[2], finalLine[3]), color, 4)
        # print(isDashed)
        # print("Min and MAx")
        # print(minTuple)
        # print(maxTuple)
        results.append(minTuple)
        results.append(maxTuple)
        results.append(finalLine)
        results.append(isDashed)

def getKey(item):
    return item[0][0]

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_detect_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * a + img * b + c
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, c)

def process_frame(image, secondFrontVehilcle, nearestBackVehicle):
    global first_frame

    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #hsv = [hue, saturation, value]
    #more accurate range for yellow since it is not strictly black, white, r, g, or b

    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    kernel_size = 5
    gauss_gray = gaussian_blur(mask_yw_image,kernel_size)

    #same as quiz values
    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(gauss_gray,low_threshold,high_threshold)

    imshape = image.shape
    lower_left = [imshape[1]/18,imshape[0]]
    lower_right = [imshape[1]-imshape[1]/18,imshape[0]]
    top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
    top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]

    vertices = roi_coordinates_right_side(image, secondFrontVehilcle, nearestBackVehicle)
    roi_image = region_of_interest(canny_edges, vertices)

    #rho and theta are the distance and angular resolution of the grid in Hough space
    #same values as quiz
    rho = 2
    theta = np.pi/180
    #threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 50
    min_line_len = 50
    max_line_gap = 200

    line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
    result = weighted_img(line_image, image, a=0.8, b=1., c=0.)
    return result

def roi_coordinates_right_side(image, secondFrontVehilcle, nearestBackVehicle):
    imshape = image.shape

    if (secondFrontVehilcle is None) and (nearestBackVehicle is None):
        lower_left = [imshape[1] / 2, imshape[0]]
        lower_right = [imshape[1], imshape[0]]
        top_left = [imshape[1] / 2, imshape[0] / 2 + imshape[0] / 10]
        top_right = [imshape[1] / 2 + 1 * imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
    else:
        if nearestBackVehicle is None:
            top_y = secondFrontVehilcle.y2
        else:
            top_y = nearestBackVehicle.y2

        lower_left = [imshape[1] / 2, imshape[0]]
        lower_right = [imshape[1], imshape[0]]
        top_left = [imshape[1] / 2, top_y]
        top_right = [imshape[1] / 2 + 11 * imshape[1] / 50, top_y]

    # lower_left = [imshape[1] / 2, imshape[0]]
    # lower_right = [imshape[1] - imshape[1] / 9, imshape[0]]
    # top_left = [imshape[1] / 2, imshape[0] / 2 + imshape[0] / 10]
    # top_right = [imshape[1] / 2 + 1 * imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]

    #roi_marker(lower_left, lower_right, top_left, top_right, image)
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
    return vertices

def roi_marker(lower_left, lower_right, top_left, top_right, img):
    color = [0, 255, 0]
    cv2.line(img, (int(lower_left[0]), int(lower_left[1])), (int(lower_right[0]), int(lower_right[1])), color, 2)
    cv2.line(img, (int(top_right[0]), int(top_right[1])), (int(top_left[0]), int(top_left[1])), color, 2)
    cv2.line(img, (int(lower_left[0]), int(lower_left[1])), (int(top_left[0]), int(top_left[1])), color, 2)
    cv2.line(img, (int(top_right[0]), int(top_right[1])), (int(lower_right[0]), int(lower_right[1])), color, 2)

def roi_coordinates_left_side(image):
    imshape = image.shape
    lower_left = [imshape[1] / 9, imshape[0]]
    lower_right = [(imshape[1] - imshape[1] / 9)/2, imshape[0]]
    top_left = [imshape[1] / 2 - imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
    top_right = [(imshape[1] / 2 + imshape[1] / 8)/ 2, imshape[0] / 2 + imshape[0] / 10]
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
    return vertices

def lineDetector(image, secondFrontVehilcle, nearestBackVehicle):
    # for source_img in os.listdir("test_images/"):
    #     image = mpimg.imread("test_images/" + source_img)
    #     processed = process_frame(image)
    #     mpimg.imsave("test_images/annotated_" + source_img, processed)
    # return results
    image = process_frame(image, secondFrontVehilcle, nearestBackVehicle)
    # cv2.imshow('Object detector', image)
    # Press any key to close the image
    # cv2.waitKey(0)
    results.append(image)
    return results
    #cv2.destroyAllWindows()
#lineDetector(cv2.imread('test_images/1640.jpeg'))


