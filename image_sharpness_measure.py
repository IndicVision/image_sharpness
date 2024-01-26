import cv2
import functools
import ffmpeg
import numpy as np
import os
import pandas as pd
import time

from PIL import Image
from rembg import remove, new_session

ROOT_DIR = "/Users/shankar/Dropbox/temp_VBN/"
RUN_FRAME_EXTRACTION = True
CREATE_THUMBS = False
RUN_REMBG = True


def runtime_monitor(input_function):
    """ Runtime decorator function. """

    @functools.wraps(input_function)
    def runtime_wrapper(*args, **kwargs):
        tic_cpu = time.process_time()
        tic_wall = time.perf_counter()
        return_value = input_function(*args, **kwargs)
        toc_cpu = time.process_time()
        toc_wall = time.perf_counter()

        func_name = input_function.__name__
        proc_time_cpu = toc_cpu - tic_cpu
        proc_time_wall = toc_wall - tic_wall
        print(f"   Finished executing {func_name} in CPU time {proc_time_cpu:.3f} seconds and wall-clock time {proc_time_wall:.3f} seconds")
        return return_value
    return runtime_wrapper


def change_to_directory(dest_folder):
    """Checks if child directory exists and changes to it."""
    
    assert os.path.exists(os.path.dirname(dest_folder)), 'parent directory does not exist'

    if os.path.basename(dest_folder) not in os.listdir(os.path.dirname(dest_folder)):
        os.mkdir(dest_folder)

    os.chdir(dest_folder)
    
@runtime_monitor
def extract_frames(input_path, fps=2.0, scale_factor=1.0, output_folder_name="frames"):
    """ Extracts frames from a video.
    
    Inputs:
        input_path:     path to video file
        fps:            frame rate in fps
        scale_factor:     scale factor for resizing
    
    """
    parent_dir = os.path.dirname(input_path)
    frames_dir = os.path.join(parent_dir, output_folder_name)
    change_to_directory(frames_dir)

    stream = ffmpeg.input(input_path)
    stream = ffmpeg.filter(stream, 'fps', fps=fps, round='up')
    stream = ffmpeg.filter(stream, 'scale', **{'w': f"iw*{scale_factor}", 'h': '-1'})
    stream = ffmpeg.output(stream, '%d_frame.png', loglevel='quiet')
    ffmpeg.run(stream)

    print('ffmpeg frames written')
    return


def get_image_sizes(folder_path):
    image_sizes = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img = Image.open(os.path.join(folder_path, filename))
            image_sizes[filename] = img.size

    return image_sizes

@runtime_monitor
def create_thumbs(parent_dir, scale_factor=0.1):
    """ Creates thumbnails that are 10% of original size."""

    # Load the image
    frames_dir = os.path.join(parent_dir, 'rembg_frames')
    thumbs_dir = os.path.join(parent_dir, 'rembg_thumbs')
    if "thumbs" not in os.listdir(parent_dir):
        os.mkdir(os.path.join(parent_dir, "rembg_thumbs"))

    list_of_frames = [frame for frame in os.listdir(frames_dir) if frame.endswith('.png')]
    
    for frame in list_of_frames:

        image = cv2.imread(os.path.join(frames_dir, frame))

        # Check if image was successfully loaded
        if image is None:
            print("Error: Could not load image {os.path.basename(frame)}")
        else:
            # Calculate the new dimensions
            width = int(image.shape[1] * scale_factor)
            height = int(image.shape[0] * scale_factor)
            new_dimensions = (width, height)

            # Resize the image
            resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

            # Save the resized image
            resized_image_name = os.path.splitext(os.path.basename(frame))[0] + "_th.png"
            resized_image_path = os.path.join(thumbs_dir, resized_image_name)
            cv2.imwrite(resized_image_path, resized_image)

    print(f"image resizing done")


@runtime_monitor
def write_rembg_frames(parent_dir, session, input_folder_name='thumbs'):
    """ 
    Input: parent_dir contains a folder called frames, that contain frames extracted from a video

    Output: a new folder called 'rembg_frames' is created at the same level as frames to store rembg frames
    """
    
    frames_dir = os.path.join(parent_dir, input_folder_name)
    rembg_dir = os.path.join(parent_dir, 'rembg_' + input_folder_name)
    change_to_directory(rembg_dir)

    list_of_frames = [file for file in os.listdir(frames_dir) if file.endswith('.png')]
    print('Writing REMBG frames ...')
    print('Number of frames: ', len(list_of_frames))

    if RUN_REMBG:
        for i, file in enumerate(list_of_frames):
            print('\r--- frame_id: %3d ---' % i, end='')
            input_path = os.path.join(frames_dir, file)
            stem = file.split('.png')[0]
            output_path = os.path.join(rembg_dir, stem + "_rembg.png")

            with open(input_path, 'rb') as i:
                with open(output_path, 'wb') as o:
                    input = i.read()
                    output = remove(input, session=session)
                    o.write(output)
    
    print("rembg frames done")
                    

def calculate_VoL(image_path):
    """ Calculates the variance of the Laplacian, a measure of image sharpness. """
    # Load image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Calculate the variance (sharpness)
    sharpness = np.var(laplacian)

    return sharpness


def extract_leading_number(filename):
    return int(''.join(filter(str.isdigit, filename)))


def find_files(directory, ending='.mp4'):
    mp4_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(ending):
                mp4_files.append(os.path.join(root, file))
    return mp4_files


@runtime_monitor
def write_sharpness_data(frames_dir, df, image_type="Full"):
    """ Computes and writes out sharpness data of all PNG images in the folder to a Pandas Dataframe."""

    frames = sorted([file for file in os.listdir(frames_dir) if file.endswith('.png')], key=extract_leading_number)
    paths = [os.path.join(frames_dir, frame) for frame in frames]

    print('Calculating sharpness ...')
    sharpness_vals = [calculate_VoL(path) for path in paths]

    print('Reading resolutions ...')
    resolutions = [Image.open(path).size for path in paths]

    image_type = [image_type]*len(frames)
    new_data = {'Path': os.path.dirname(os.path.relpath(frames_dir, ROOT_DIR)),
                'Frame': frames,
                'Type': image_type,
                'Resolution': resolutions,
                'Variance of Laplacian': sharpness_vals}
    df = df.append(pd.DataFrame(new_data), ignore_index=True)

    print("Data appended to Pandas dataframe")

    return df


# parent_dir = "/Users/shankar/Dropbox_Personal/HealWell/video_shoots/Selvamohan/Jan20_2024/top"

if __name__ == "__main__":

    # Grab all video files in the folder
    my_video_files = find_files(ROOT_DIR)

    # Read pickle file, creating it if needed
    pkl_file = os.path.join(ROOT_DIR, 'sharpness.pkl')
    try:
        df = pd.read_pickle(pkl_file)
    except(FileNotFoundError):
        columns = ['Path', 'Frame', 'Type', 'Resolution', 'Variance of Laplacian']
        df = pd.DataFrame(columns=columns)
    

    # if pkl_file not in os.listdir(ROOT_DIR):
    #     columns = ['Path', 'Type', 'Resolution', 'Variance of Laplacian']
    #     df = pd.DataFrame(columns=columns)
    #     df.to_pickle(pkl_file)
    # df = pd.read_pickle(pkl_file)

    if RUN_REMBG:
        session = new_session()

    for i, fpath in enumerate(my_video_files):
        file_name = os.path.basename(fpath)
        parent_dir = os.path.dirname(fpath)
        
        print(f'-------\n({i})\n{file_name}')
        print(parent_dir)

        if RUN_FRAME_EXTRACTION:
            extract_frames(fpath, scale_factor=0.1, output_folder_name='thumbs')
        if RUN_REMBG:
            write_rembg_frames(parent_dir, session, input_folder_name='thumbs')
        # if CREATE_THUMBS:
        #     create_thumbs(parent_dir)

        # Write sharpness values for full frames

        # frames_dir = os.path.join(parent_dir, 'rembg_frames')
        # df = write_sharpness_data(frames_dir, df, type="Full")

        thumbs_dir = os.path.join(parent_dir, 'rembg_thumbs')
        df = write_sharpness_data(thumbs_dir, df, image_type="Thumbnail")

    # Add a relative path column
    # df['rel_path'] = df['Path'].apply(lambda x: os.path.dirname(os.path.relpath(x, ROOT_DIR)))

    df.to_pickle(pkl_file)