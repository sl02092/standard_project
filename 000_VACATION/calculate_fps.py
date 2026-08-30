import glob
import cv2

for filepath in sorted(glob.glob("C:\\repo\\VACATION\\videos\\*.mp4")):
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{filepath}: {fps:.2f} FPS")
    cap.release()