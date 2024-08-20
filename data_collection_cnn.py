import numpy as np
import pyrealsense2 as rs
import cv2
import os


def cut_region_at_distance(depth_image,color_image,min_depth = 0,max_depth = 0.8, region = False):
                
                mask = np.logical_and(depth_image > min_depth * 1000, depth_image < max_depth * 1000)

                # Convert mask to uint8
                mask = mask.astype(np.uint8) * 255
                masked_color_image = cv2.bitwise_and(color_image, color_image, mask=mask)
                return masked_color_image



#Initialize
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, framerate = 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, framerate = 30)

pipeline.start(config)

#create align object
align_to = rs.stream.color
align = rs.align(align_to)


try:
    while True:
            try:
                
                # Wait for a new frame
                frames = pipeline.wait_for_frames()

                aligned_frames = align.process(frames)
                # Get color and depth frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame :
                    continue

                

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.25), cv2.COLORMAP_JET)


                masked_color_image = cut_region_at_distance(depth_image, color_image,min_depth = 0,max_depth = 0.8, region = False)
                
      
                  
                scale_factor = 0.5
                resized_image = cv2.resize(masked_color_image,None,fx =  scale_factor,fy = scale_factor)
          

                cv2.imshow('Color Image', resized_image)
           
            
        
               
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except RuntimeError as e:
                print("Error:", e)
                continue


finally:
    # Stop streaming
    pipeline.stop()

 
    cv2.destroyAllWindows()
    