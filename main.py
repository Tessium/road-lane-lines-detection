import time
from utils import *

# ignore all warnings coming from numpy
np.seterr(all='ignore')


def main(video, stream=True, frame_size=(1024, 768)):
    """
    main method for processing video
    breaks video into frames
    applies processing methods to each frame

    :param video: target video file
    :param stream: if set True streams, else writes to file
    :param frame_size: dimensions of frame to resize
    :return:
    """

    # initializing variables for:
    # counting number of processed frames
    # video write for saving
    frame_count = 0
    out = None

    # if stream = False
    # create video writer object with same fps and dimensions
    if not stream:
        out = cv2.VideoWriter('processed_{}x{}.mp4'.format(frame_size[0], frame_size[1]),
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              int(round(video.get_fps())),
                              frame_size)
    # get tick count
    timer = cv2.getTickCount()

    # while frame cursor did not reach the end
    while video.is_opened():

        ret, frame = video.read()

        # if frame retrieved
        if ret:

            # frame resized to desired dimensions
            frame = cv2.resize(frame, frame_size, cv2.INTER_AREA)

            # process the frame
            frame, left_slope, right_slope = process(frame)

            slope_sum = left_slope + right_slope

            # print(round(slope_sum, 2))
            # print(round(left_slope, 2), round(right_slope, 2))

            if slope_sum > .1:
                cv2.putText(frame, "Direction: Smooth Right", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 3)
                # print('right')
            elif slope_sum < -0.9:
                cv2.putText(frame, "Direction: Smooth Left", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 3)
                # print('left')
            else:
                cv2.putText(frame, "Direction: Forward", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 3)

            # if stream = True
            if stream:
                # calculate fps
                fps = cv2.getTickFrequency() // (cv2.getTickCount() - timer)
                # update tick count
                timer = cv2.getTickCount()

                # writes last calculated fps
                # writes total time passed in seconds
                # opens a window with streaming video
                cv2.putText(frame, "FPS: " + str(fps), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 3)
                cv2.imshow('Processing', frame)

                # if q pressed, closes the streaming window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # if stream = False
            # write frame into video
            # print status each 500 frames
            else:
                if frame_count % 500 == 0:
                    print('Processed {} frames out of {}'.format(frame_count, video.get_frame_count()))
                out.write(frame)
                frame_count += 1

        # if frame not retrieved
        else:
            break

    # if saving, close writer
    if not stream:
        out.release()


if __name__ == '__main__':
    # divide video in chunks, if needed
    # video = Video('video.mp4')
    # get_chunks('video.mp4', 20, video.len)

    # create Video object
    video = Video('video.mp4')

    # call main method with desired arguments
    # possible dimensions:
    # 1024x768 | 800x600 | 640x480 | 400x300
    main(video, stream=False, frame_size=(1024, 768))

    # release video VideoCapture object
    video.release()

    # close all windows if any is open
    cv2.destroyAllWindows()
