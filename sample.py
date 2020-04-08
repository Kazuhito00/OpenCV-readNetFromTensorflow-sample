#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import time

import cv2 as cv


def main():
    # グラフ読み込み ###########################################################
    pb_filepath = os.path.join(os.getcwd(), 'model',
                               'frozen_inference_graph.pb')
    pbtxt_filepath = os.path.join(os.getcwd(), 'model', 'graph.pbtxt')

    cvNet = cv.dnn.readNetFromTensorflow(pb_filepath, pbtxt_filepath)

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    fps = 10

    while True:
        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        debug_image = copy.deepcopy(frame)
        rows = debug_image.shape[0]
        cols = debug_image.shape[1]

        # 手検出実施 ###########################################################
        blob_image = cv.dnn.blobFromImage(
            debug_image, size=(512, 512), swapRB=True, crop=False)
        cvNet.setInput(blob_image)
        cvOut = cvNet.forward()

        for detection in cvOut[0, 0, :, :]:
            class_id = detection[1]
            score = float(detection[2])
            if score < 0.75:
                continue

            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            # バウンディングボックス描画 #######################################
            cv.putText(debug_image,
                       str(class_id) + ":" + '{:.3f}'.format(score),
                       (int(left), int(top) - 15), cv.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 255, 0), 2, cv.LINE_AA)
            cv.rectangle(
                debug_image, (int(left), int(top)), (int(right), int(bottom)),
                (0, 255, 0),
                thickness=2)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # FPS調整 #############################################################
        elapsed_time = time.time() - start_time
        sleep_time = max(0, ((1.0 / fps) - elapsed_time))
        time.sleep(sleep_time)

        # 画面反映 #############################################################
        cv.putText(
            debug_image,
            "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow(' ', debug_image)
        cv.moveWindow(' ', 100, 100)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
