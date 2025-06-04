import subprocess
import os

def convert_frames_to_video(
    input_folder="frames",
    output_file="output.mp4",
    framerate=30,
    codec="libx264",
    pattern="frame_%04d.png"
):
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Folder '{input_folder}' does not exist.")

    cmd = [
        "ffmpeg",
        "-framerate", str(framerate),
        "-i", os.path.join(input_folder, pattern),
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        output_file
    ]

    print("Running ffmpeg to create video...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("ffmpeg failed:")
        print(result.stderr.decode())
