import cv2
from ultralytics import YOLO, solutions
from ultralytics.utils.plotting import Annotator

# Load the model
model = YOLO("./data/yolov8x.pt")

cap = cv2.VideoCapture("./data/data.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

zones = [
    [(32, 1726), (1716, 1678), (1696, 1098), (784, 1134), (28, 1718)],
    [(2408, 1674),(3732, 1550),(2876, 1122),(2096, 1170),(2404, 1662)]
]

counters = [
    solutions.ObjectCounter(
        view_img=False,
        reg_pts=zone,
        names=model.names,
        draw_tracks=False,
        line_thickness=2,
    )
    for zone in zones
]

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)

    # Count objects in each region
    for counter in counters:
        im0 = counter.start_counting(im0, results)

    out.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
