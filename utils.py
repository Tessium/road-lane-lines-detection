import os
from classes import *

# global variables to hold old values of line
OLD_LEFT = None
OLD_RIGHT = None
OLD_LINES = None


def get_chunks(video, chunk_size, length):
    """
    function to break video into chunks, using library ffmpeg

    :param video: path to desired video
    :param chunk_size: length of chunks
    :param length: total length of video
    :return:
    """

    # calculate number of chunks
    chunks = int(length // chunk_size) + 1

    # create dir for saving video chunks
    if not os.path.exists('videos1'):
        os.makedirs('videos1')

    # break video into chunks, using ffmpeg library
    for i in range(chunks):
        start = i * chunk_size
        if i == chunks - 1:
            os.system('ffmpeg -i {} -ss {} videos1\\vid_{}.mp4'.format(video, start, i + 1))
        else:
            os.system('ffmpeg -i {} -ss {} -t {} videos1\\vid_{}.mp4'.format(video, start, chunk_size, i + 1))


def mask_image(img, vertices):

    """
    Function used to mask image within vertices (cuts off lines out of polygon)

    :param img: image to mask
    :param vertices: 4 coordinates of polygon
    :return: masked image from given vertices
    """

    # creating empty image with the same shape as img
    mask = np.zeros_like(img)

    # filling empty image within the given vertices
    cv2.fillPoly(mask, vertices, 255)

    # AND operation results in leaving lines
    # only within the given vertices
    masked = cv2.bitwise_and(img, mask)

    # return masked image
    return masked


def blend(img, initial_img):

    """
    Function to blend two images

    :param img: image with lines
    :param initial_img: original image
    :return: blended image of initial and image with lines
    """

    # creating empty array with shape of img
    empty = np.zeros_like(img)

    # cutting image by half and changing positions of two halves
    # top to bottom, bottom to top
    empty = empty[empty.shape[0] // 2 : empty.shape[0]]
    img = img[:img.shape[0] // 2]
    img = np.vstack((empty, img))

    # expanding dimensions of one channel image, by creating
    # two more same shape empty channels and stacking by dimensions
    img = np.uint8(img)
    img = np.dstack((np.zeros_like(img), np.zeros_like(img), img))

    # add two images by custom alpha, betta and lambda values
    # in this case, alpha and betta are equal to 1
    # which means, both images shall keep their original colors
    return cv2.addWeighted(initial_img, .8, img, 1, 1)


def fix_lines(lines):

    """
    Function to fix absent lane lines
    Absent or wrong lines are determined by their biases

    :param lines: list of left and right lines
    :return: avg of current and old lines
    """

    # loading global left, right variables
    # they keep last successful frame lines
    global OLD_LEFT
    global OLD_RIGHT

    # if it is first frame and if both lines are normal
    # we save lines to global and return current lines
    if OLD_LEFT is None and OLD_RIGHT is None:
        if lines[0].bias > 0 > lines[1].bias:
            OLD_LEFT = Line(lines[0].x1, lines[0].y1, lines[0].x2, lines[0].y2)
            OLD_RIGHT = Line(lines[1].x1, lines[1].y1, lines[1].x2, lines[1].y2)
        return lines

    # if lines are wrong, we return last successful lines
    if lines[0].bias < 0 or lines[1].bias < -1000:
        return OLD_LEFT, OLD_RIGHT

    # variables for storing old and current lines
    # 1 dimension loaded with old lines
    # 2 dimension loaded with current ones
    med_left = np.zeros((2, 4))
    med_right = np.zeros((2, 4))

    med_left[0] += OLD_LEFT.get_coords()
    med_right[0] += OLD_RIGHT.get_coords()

    med_left[1] += lines[0].get_coords()
    med_right[1] += lines[1].get_coords()

    # if lines are normal, we save it to global
    if lines[0].bias > 0:
        OLD_LEFT = Line(lines[0].x1, lines[0].y1, lines[0].x2, lines[0].y2)
    if lines[1].bias < 0:
        OLD_RIGHT = Line(lines[1].x1, lines[1].y1, lines[1].x2, lines[1].y2)

    # return new left and right lines, by taking their mean
    return Line(*np.nanmean(med_left, axis=0)), Line(*np.nanmean(med_right, axis=0))


def get_vertices(w, h, cut=True):
    if cut:
        if w == 1024:
            vertices = np.array([[[450, 165], [600, 165], [w, h], [300, h]]])
        elif w == 800:
            vertices = np.array([[[350, 130], [450, 130], [w, h], [250, h]]])
        elif w == 640:
            vertices = np.array([[[300, 105], [370, 105], [w, h], [150, h]]])
        elif w == 400:
            vertices = np.array([[[180, 70], [250, 75], [w, h], [100, h]]])
        else:
            vertices = np.array([[[0, 0], [1280, 0], [1280, 720], [0, 720]]])
    else:
        if w == 1024:
            vertices = np.array([[[0, 165], [w, 165], [w, h], [0, h]]])
        elif w == 800:
            vertices = np.array([[[0, 130], [w, 130], [w, h], [0, h]]])
        elif w == 640:
            vertices = np.array([[[0, 105], [w, 105], [w, h], [0, h]]])
        elif w == 400:
            vertices = np.array([[[0, 70], [w, 75], [w, h], [0, h]]])
        else:
            vertices = np.array([[[0, 0], [1280, 0], [1280, 720], [0, 720]]])
    return vertices


def get_lines(frame):

    """
    function to extract hough lines from frame

    :param frame: frame to process
    :return: detected lines
    """

    # load old lines
    global OLD_LINES

    # take grayscale of frame, converts 3 channels to 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur grayscale image with 3x3 kernel matrix
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # detect edges of blurred and grayscaled image
    # with Canny method of edge detection
    edges = cv2.Canny(blur, 90, 180)

    # create vertices depending on image dimensions
    h, w = edges.shape

    # get vertices for current frame dimension
    vertices = get_vertices(w, h)

    # mask detected edges, to remove redundant ones and leave only lane lines edges
    edges = mask_image(edges, vertices)

    # detect hough lines on masked image with detected edges
    detected = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, minLineLength=20, maxLineGap=300)

    # if no lines detected (case with absent lane lines)
    # else, save current lines
    if detected is None:
        detected = OLD_LINES
    else:
        detected = [Line(i[0][0], i[0][1], i[0][2], i[0][3]) for i in detected]
        OLD_LINES = detected

    if detected is None:
        return None

    # filter lines based on absolute value of their slopes
    # removes redundant lines, such as, vertical or horizontal lines
    lines = []

    for line in detected:
        # if .25 <= np.abs(line.slope) <= 2.25:
        lines.append(line)

    # merge left lines to one left line
    # merge right lines to one right line
    lines = merge_lines(lines, gray.shape)

    # return merged lines
    return lines


def merge_lines(lines, img_shape):
    """
    function to merge lines into solid left and solid right

    :param lines: lines to divide by sides and merge
    :param img_shape: shape of image
    :return: merged solid two lines
    """

    def calc_line(side_lines, s=True):
        """
        function to calculate new line based on others

        :param side_lines: all lines from one side
        :param s: if True, left lines, else right
        :return: new line
        """

        # get median of biases and slopes of lines
        bias = np.median([i.bias for i in side_lines]).astype(int)
        slope = np.median([i.slope for i in side_lines])

        # print(img_shape[0])
        # calculate new coordinates of line depending on side
        if s:
            x1, y1 = 0, bias
            x2, y2 = -np.int32(np.round(bias / slope)), 0
        else:
            x1, y1 = 0, bias
            x2, y2 = np.int32(np.round((img_shape[0] - bias) / slope)), img_shape[0]

        # return new line
        return Line(x1, y1, x2, y2)

    # filter lines based on their slope
    # positive slopes are right lines
    # negative, left
    # and pass to the method for computing new line
    left = [i for i in lines if i.slope < 0]
    left = calc_line(left)

    right = [i for i in lines if i.slope > 0]
    right = calc_line(right, False)

    # return lines
    return left, right


def process(frame):

    """
    method for processing frame passed from main

    :param frame: desired frame
    :return: processed image
    """

    # cut horizontally image by half and take bottom part
    cut = frame[frame.shape[0] // 2 : frame.shape[0]]

    # decide vertices of polygon for ROI, depending on frame dimensions
    h, w, _ = frame.shape

    vertices = get_vertices(w, h, cut=False)

    # get hough lines from frame
    lines = get_lines(cut)

    if not lines:
        return frame, 0

    # smooth and fix absent lines
    lines = fix_lines(lines)

    # create empty image with shape of initial image
    # draw lines on it
    img = np.zeros(shape=(h, w))
    for line in lines:
        line.draw(img)

    # mask image with early created vertices
    # cuts off lines out of ROI
    masked = mask_image(img, vertices)

    # blend initial image and image with lines
    blended = blend(masked, frame)

    # return ready frame
    return blended, lines[0].slope, lines[1].slope
