import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchreid

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

REID_THRESHOLD = 0.60

# Number of consecutive strong matches required
# before accepting a re-identification.
REID_CONFIRMATIONS_REQUIRED = 3

# Frames target can disappear before entering ReID search.
LOST_FRAME_GRACE = 10


# ============================================================
# MODELS
# ============================================================

print("Loading YOLO...")

yolo = YOLO("yolo11n.pt")

print("Loading OSNet ReID...")

reid_model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

reid_model.eval()

print("TargetLock models loaded.")


# ============================================================
# REID PREPROCESSING
# ============================================================

reid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# TARGET STATE
# ============================================================

target_id = None

# Original / current appearance representation
target_embedding = None

# Last known target location
target_bbox = None

# Frames target has been missing
lost_frames = 0

# ReID confirmation counter
reid_confirmation_count = 0

# Whether we are currently searching
searching = False


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")


# ============================================================
# REID FUNCTION
# ============================================================

def get_embedding(person_crop):

    if person_crop is None or person_crop.size == 0:
        return None

    tensor = reid_transform(person_crop).unsqueeze(0)

    with torch.no_grad():
        embedding = reid_model(tensor)

    # Normalize embedding so cosine similarity is stable.
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding


# ============================================================
# FIND TARGET
# ============================================================

def select_target(result, frame, x, y):

    global target_id
    global target_embedding
    global target_bbox
    global lost_frames
    global searching
    global reid_confirmation_count

    if result.boxes.id is None:
        return False

    for box in result.boxes:

        track_id = int(box.id[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if x1 <= x <= x2 and y1 <= y <= y2:

            crop = frame[y1:y2, x1:x2]

            embedding = get_embedding(crop)

            if embedding is None:
                return False

            target_id = track_id
            target_embedding = embedding
            target_bbox = (x1, y1, x2, y2)

            lost_frames = 0
            searching = False
            reid_confirmation_count = 0

            print()
            print("=" * 50)
            print(f"TARGET SELECTED: ID {target_id}")
            print("Target appearance embedding saved.")
            print(f"Embedding dimensions: {embedding.shape}")
            print("=" * 50)
            print()

            return True

    return False


# ============================================================
# MOUSE CALLBACK
# ============================================================

current_result = None
current_frame = None


def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if current_result is not None and current_frame is not None:

            select_target(
                current_result,
                current_frame,
                x,
                y
            )


window_name = "TargetLock"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame could not be read.")
        break

    current_frame = frame.copy()

    # --------------------------------------------------------
    # DETECTION + TRACKING
    # --------------------------------------------------------

    results = yolo.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[0],
        verbose=False
    )

    current_result = results[0]

    annotated = frame.copy()

    # Draw normal detections
    if current_result.boxes is not None:

        for box in current_result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if box.id is not None:

                track_id = int(box.id[0])

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (100, 100, 100),
                    1
                )

                cv2.putText(
                    annotated,
                    f"ID {track_id}",
                    (x1, max(y1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1
                )


    # --------------------------------------------------------
    # CHECK WHETHER TARGET IS STILL TRACKED
    # --------------------------------------------------------

    target_visible = False

    if (
        target_id is not None
        and current_result.boxes.id is not None
    ):

        for box in current_result.boxes:

            track_id = int(box.id[0])

            if track_id == target_id:

                target_visible = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                target_bbox = (x1, y1, x2, y2)

                lost_frames = 0
                searching = False
                reid_confirmation_count = 0

                # ------------------------------------------------
                # TARGET BOX
                # ------------------------------------------------

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    4
                )

                cv2.putText(
                    annotated,
                    "TARGET LOCKED",
                    (x1, max(y1 - 12, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                break


    # --------------------------------------------------------
    # TARGET LOST
    # --------------------------------------------------------

    if target_id is not None and not target_visible:

        lost_frames += 1

        if lost_frames >= LOST_FRAME_GRACE:

            searching = True


    # --------------------------------------------------------
    # REID SEARCH
    # --------------------------------------------------------

    if (
        searching
        and target_embedding is not None
        and current_result.boxes.id is not None
    ):

        cv2.putText(
            annotated,
            "SEARCHING FOR TARGET",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2
        )

        best_similarity = -1
        best_candidate_id = None
        best_candidate_bbox = None

        # -----------------------------------------------
        # Compare EVERY visible person
        # -----------------------------------------------

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

            # Display similarity
            cv2.putText(
                annotated,
                f"ReID {similarity:.2f}",
                (x1, min(y2 + 20, frame.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

            print(
                f"Candidate ID {candidate_id} "
                f"| similarity = {similarity:.4f}"
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_candidate_id = candidate_id
                best_candidate_bbox = (
                    x1,
                    y1,
                    x2,
                    y2
                )


        # -----------------------------------------------
        # MULTI-FRAME CONFIRMATION
        # -----------------------------------------------

        if (
            best_candidate_id is not None
            and best_similarity >= REID_THRESHOLD
        ):

            reid_confirmation_count += 1

            cv2.putText(
                annotated,
                f"REID CONFIRMING "
                f"{reid_confirmation_count}/"
                f"{REID_CONFIRMATIONS_REQUIRED}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            if (
                reid_confirmation_count
                >= REID_CONFIRMATIONS_REQUIRED
            ):

                old_id = target_id

                target_id = best_candidate_id
                target_bbox = best_candidate_bbox

                lost_frames = 0
                searching = False
                reid_confirmation_count = 0

                print()
                print("=" * 60)
                print("TARGET RE-IDENTIFIED")
                print(
                    f"Old track ID: {old_id}"
                )
                print(
                    f"New track ID: {target_id}"
                )
                print(
                    f"ReID similarity: "
                    f"{best_similarity:.4f}"
                )
                print("=" * 60)
                print()

        else:

            # Candidate wasn't consistently strong enough.
            reid_confirmation_count = 0


    # --------------------------------------------------------
    # STATUS UI
    # --------------------------------------------------------

    if target_id is None:

        cv2.putText(
            annotated,
            "CLICK A PERSON TO SELECT TARGET",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    elif searching:

        cv2.putText(
            annotated,
            "REID RECOVERY ACTIVE",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )

    else:

        cv2.putText(
            annotated,
            f"TARGET ID: {target_id}",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow(window_name, annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("TargetLock stopped.")