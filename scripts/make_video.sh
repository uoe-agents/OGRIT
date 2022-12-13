python ../../scripts/make_video.py .
ffmpeg -y -framerate 25 -pattern_type glob -i 'img*.png' -c:v libx264 out.mp4