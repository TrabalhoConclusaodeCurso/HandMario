#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
import cv2
import multiprocessing as _mp
from pygame import mixer
from utils import load_graph, mario, detect_hands, predict
from config import ORANGE, RED, GREEN

tf.flags.DEFINE_integer("width", 640, "Screen width")
tf.flags.DEFINE_integer("height", 480, "Screen height")
tf.flags.DEFINE_float("threshold", 0.6, "Threshold for score")
tf.flags.DEFINE_float("alpha", 0.3, "Transparent level")
tf.flags.DEFINE_string("pre_trained_model_path", "pretrained_model.pb", "Path to pre-trained model")

FLAGS = tf.flags.FLAGS

def main():
    graph, sess = load_graph(FLAGS.pre_trained_model_path)
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FLAGS.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FLAGS.height)
    mp = _mp.get_context("spawn")
    v = mp.Value('i', 0)
    lock = mp.Lock()
    process = mp.Process(target=mario, args=(v, lock))
    process.start()
    mixer.init()
    mixer.music.load("super-mario-bros-4293.mp3")
    mixer.music.play(999)
    while True:
        #playsound('super-mario-bros-4293.mp3',0)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_hands(frame, graph, sess)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = predict(boxes, scores, classes, FLAGS.threshold, FLAGS.width, FLAGS.height)

        if len(results) == 1:
            #mixer.music.init(44100, -16, 1, 512)
            #sounds["start"] = mixer.music.load("smb_jump-super.wav")
            #sounds["end"] = mixer.music.load("resources/sounds/gameover.ogg")

            x_min, x_max, y_min, y_max, category = results[0]
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
            cv2.circle(frame, (x, y), 5, RED, -1)

            if category == "Open" and x <= FLAGS.width / 3:
                #playsound('smb_jump-super.wav')
                #sounds["start"].mixer.play() (1)
                action = 7  # Left jump
                text = "Jump left"

            elif category == "Closed" and x <= FLAGS.width / 3:
                action = 6  # Left
                text = "Run left"
            elif category == "Open" and FLAGS.width / 3 < x <= 2 * FLAGS.width / 3:
                #playsound('smb_jump-super.wav',0)
                action = 5  # Jump
                text = "Jump"
            elif category == "Closed" and FLAGS.width / 3 < x <= 2 * FLAGS.width / 3:
                action = 0  # Do nothing
                text = "Stay"
            elif category == "Open" and x > 2 * FLAGS.width / 3:
                #playsound('smb_jump-super.wav',0)
                action = 2  # Right jump
                text = "Jump right"
            elif category == "Closed" and x > 2 * FLAGS.width / 3:
                action = 1  # Right
                text = "Run right"
            else:
                action = 0
                text = "Stay"
            with lock:
                v.value = action
            cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (int(FLAGS.width / 3), FLAGS.height), ORANGE, -1)
        cv2.rectangle(overlay, (int(2 * FLAGS.width / 3), 0), (FLAGS.width, FLAGS.height), ORANGE, -1)
        cv2.addWeighted(overlay, FLAGS.alpha, frame, 1 - FLAGS.alpha, 0, frame)
        #out.write(frame)
        cv2.imshow('Detection', frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
