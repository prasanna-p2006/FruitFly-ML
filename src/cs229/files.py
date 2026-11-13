import os
import os.path
import cv2
import numpy as np

class CapProps:
    def __init__(self, width, height, fps):
        self.width=width
        self.height=height
        self.fps = fps

    @property
    def t_ms(self):
        return int(round(1e3/self.fps))

def get_dir(*args, mkdir_p=True):
    """
    Returns the absolute path to the relative directory specified by arg0, arg1, etc.
    """

    path_to_this_file = os.path.realpath(os.path.expanduser(__file__))
    dir_name = os.path.dirname(path_to_this_file)
    parent_dir = os.path.dirname(dir_name)
    if os.path.basename(parent_dir) == 'src':
        top_dir = os.path.dirname(parent_dir)
    else:
        top_dir = parent_dir

    args_list = list(args)
    # Map 'output' requests to 'outputs' or 'models'
    if args_list and args_list[0] == 'output':
        if len(args_list) > 1 and args_list[1] == 'models':
            args_list = ['models'] + args_list[2:]
        else:
            args_list[0] = 'outputs'

    # find path to directory
    join_args = [top_dir] + args_list
    path = os.path.join(*join_args)

    # make the directory if needed/desired
    if mkdir_p:
        os.makedirs(path, exist_ok=True)

    # return the path
    return path

def get_file(*args, mkdir_p=False):
    # get path to the directory (and make directory if needed/desired)
    dir = get_dir(*args[:-1], mkdir_p=mkdir_p)

    # construct the path to the file
    path = os.path.join(dir, args[-1])

    return path

def read_video(name, backend='msmf'):
    """
    Open video file using specified backend.
    
    Parameters:
    - name: Video filename (e.g. 'test4.mp4')
    - backend: 'msmf' (cv2.CAP_MSMF - Windows Media Foundation, recommended for Windows),
               'ffmpeg' (cv2.CAP_FFMPEG), or 'auto' (cv2.CAP_ANY).
    """
    # Support both 'input/videos' and 'input/video'
    video_dir = 'videos' if os.path.isdir(get_dir('input', 'videos', mkdir_p=False)) else 'video'
    file_path = get_file('input', video_dir, name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")

    api_pref = cv2.CAP_ANY
    if backend == 'msmf':
        api_pref = cv2.CAP_MSMF
    elif backend == 'ffmpeg':
        api_pref = cv2.CAP_FFMPEG

    cap = cv2.VideoCapture(file_path, api_pref)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file with backend {backend}: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 35.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    props = CapProps(width=width, height=height, fps=fps)

    return cap, props

def write_video(name, props):
    file_path = get_file('output', 'videos', name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(file_path, fourcc, props.fps, (props.width, props.height))

    return writer
