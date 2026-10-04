import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchreid

from ultralytics import YOLO


# =========================
# MODELS
# =========================

yolo = YOLO("yolo11n.pt")

reid_model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

reid_model.eval()


# =========================
# REID PREPROCESSING
# =========================

reid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# TARGET STATE
# =========================

target_id = None
target_embedding = None

current_result = None
current_frame = None

lost_frames = 0

# Number of frames target can be missing
# before we start ReID recovery.
LOST_FRAME_GRACE = 10

# Experimental threshold.
# We will tune this after testing.
REID_THRESHOLD = 0.60


# =========================
# GET REID EMBEDDING
# =========================

def get_embedding(person_crop):

    if person_crop.size == 0:
        return None

    person_tensor = reid_transform(person_crop)
    person_tensor = person_tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = reid_model(person_tensor)

    return embedding


# =========================
# TARGET SELECTION
# =========================

def select_target(event, x, y, flags, param):

    global target_id
    global target_embedding
    global current_result
    global current_frame
    global lost_frames

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if current_result is None or current_frame is None:
        return

    if current_result.boxes.id is None:
        return

    for box in current_result.boxes:

        track_id = int(box.id[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if x1 <= x <= x2 and y1 <= y <= y2:

            # Save BoT-SORT ID
            target_id = track_id

            # Crop selected person
            person_crop = current_frame[y1:y2, x1:x2]

            # Generate ReID embedding
            target_embedding = get_embedding(person_crop)

            lost_frames = 0

            print(f"Selected target: ID {target_id}")
            print("Target embedding generated!")
            print(f"Embedding shape: {target_embedding.shape}")

            break


# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

window_name = "TargetLock - Target Selection"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, select_target)


# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_frame = frame.copy()

    # =========================
    # YOLO + BOT-SORT
    # =========================

    results = yolo.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[0],
        verbose=False
    )

    current_result = results[0]

    annotated_frame = current_result.plot()


    # =========================
    # CHECK IF TARGET IS VISIBLE
    # =========================

    target_visible = False

    if (
        target_id is not None
        and current_result.boxes.id is not None
    ):

        for box in current_result.boxes:

            track_id = int(box.id[0])

            if track_id == target_id:

                target_visible = True
                lost_frames = 0

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    annotated_frame,
                    "TARGET",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                break


    # =========================
    # TARGET LOST
    # =========================

    if target_id is not None and not target_visible:

        lost_frames += 1


    # =========================
    # REID SEARCH MODE
    # =========================

    if (
        target_id is not None
        and target_embedding is not None
        and lost_frames >= LOST_FRAME_GRACE
    ):

        cv2.putText(
            annotated_frame,
            "SEARCHING FOR TARGET...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2
        )

        best_similarity = -1
        best_candidate_id = None

        if current_result.boxes.id is not None:

            for box in current_result.boxes:

                candidate_id = int(box.id[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                candidate_crop = frame[y1:y2, x1:x2]

                candidate_embedding = get_embedding(candidate_crop)

                if candidate_embedding is None:
                    continue

                similarity = F.cosine_similarity(
                    target_embedding,
                    candidate_embedding
                ).item()

                print(
                    f"Target vs Candidate {candidate_id}: "
                    f"{similarity:.4f}"
                )

                # Show similarity above candidate
                cv2.putText(
                    annotated_frame,
                    f"Similarity: {similarity:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                # Keep highest similarity
                if similarity > best_similarity:

                    best_similarity = similarity
                    best_candidate_id = candidate_id


        # =========================
        # REIDENTIFICATION
        # =========================

        if (
            best_candidate_id is not None
            and best_similarity >= REID_THRESHOLD
        ):

            old_target_id = target_id

            target_id = best_candidate_id

            lost_frames = 0

            print(
                f"TARGET RE-IDENTIFIED! "
                f"Old ID {old_target_id} -> "
                f"New ID {target_id} "
                f"(similarity: {best_similarity:.4f})"
            )


    # =========================
    # SHOW FRAME
    # =========================

    cv2.imshow(window_name, annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()