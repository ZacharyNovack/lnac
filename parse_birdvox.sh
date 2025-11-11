for f in /mnt/arrakis_data/pnlong/lnac/birdvox/unit06/*.flac; do
  echo "Processing $f"
  ffmpeg -i "$f" -f segment -segment_time 60 -c copy "${f%.flac}_%03d.flac"
done
