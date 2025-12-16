# for f in /mnt/arrakis_data/pnlong/lnac/birdvox/unit06/*.flac; do
#   echo "Processing $f"
#   ffmpeg -i "$f" -f segment -segment_time 60 -c copy "${f%.flac}_%03d.flac"
# done

for f in /mnt/arrakis_data/pnlong/lnac/birdvox/unit06/split_data/*.flac; do
  ffmpeg -i "$f" -c:a flac -map 0 -y "/mnt/arrakis_data/pnlong/lnac/birdvox/unit06/fixed_${f##*/}"
  # oversave file
  mv "/mnt/arrakis_data/pnlong/lnac/birdvox/unit06/fixed_${f##*/}" "$f"
done
