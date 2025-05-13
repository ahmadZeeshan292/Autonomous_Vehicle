import cv2
from AutonomousVehicle import SelfDrivingVehicle

if __name__ == "__main__":
    path = r'C:\Users\Ahmad Zeeshan\OneDrive\Desktop\semestor\6th semestor\DIP\Project\Dataset\DIP Project Videos\PXL_20250325_043754655.TS.mp4'
    video = cv2.VideoCapture(path)

    ret, frame = video.read()
    scaling_factor = 1
    frame = cv2.resize(frame, (int(1280 * scaling_factor), int(720 * scaling_factor)), interpolation=cv2.INTER_AREA)

    car = SelfDrivingVehicle(frame, scaling_factor)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, (int(1280 * scaling_factor), int(720 * scaling_factor)), interpolation=cv2.INTER_AREA)
        car_o = car.forward(frame)

        cv2.imshow("output", car_o)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()
