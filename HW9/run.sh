if [ ! -f "lena.bmp" ]; then
    wget http://cv2.csie.ntu.edu.tw/CV/_material/lena.bmp -O lena.bmp
fi
python3 hw9.py