import cv2 as cv 
from argparse import ArgumentParser
import numpy as np
import pickle



if __name__ == "__main__":
    command_buffer = pickle.load(open("command_buffer.pkl", "rb"))
    image_buffer = pickle.load(open("image_buffer.pkl", "rb"))
    image_buffer_length: int = image_buffer.shape[0]
    print(image_buffer.shape())
    command_buffer_length: int = command_buffer.shape[0]
    assert image_buffer_length == command_buffer_length, "buffers are not the same size"
    cv.initFont

    codec = cv.VideoWriter_fourcc(*'MP4V')
    out = cv.VideoWriter("quadraped_video", codec, 50, image_buffer.shape[1:2])

    for i in range(image_buffer_length)
        img = image_buffer[i])
        commands = command_buffer[i]
        cv2.putText(image, 
                    f"Vx {command_buffer[0]}, Vy {comand_buffer[1]}, rZ {command_buffer[2]}",
                    (5,5), cv.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 2) 
        out.write(img)





    



    

    






