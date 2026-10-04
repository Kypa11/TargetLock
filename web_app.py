import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchreid

import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from ultralytics import YOLO


# ============================================================
# APP
# ============================================================

app = FastAPI(title="TargetLock")


# ============================================================
# MODELS
# ============================================================

print("Loading YOLO...")

yolo = YOLO("yolo11n.pt")


print("Loading OSNet...")

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

# OSNet expects the person image in this format.
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

# The BoT-SORT ID of the person we are currently following.
target_id = None

# Saved appearance representation of the target.
# We use this later if BoT-SORT loses them.
target_embedding = None

# How many frames the target has been missing.
lost_frames = 0

# Whether we are actively looking for the target using ReID.
searching = False

# We don't immediately trust one ReID match.
# The match has to be strong for several frames.
reid_confirmation_count = 0


# ============================================================
# SETTINGS
# ============================================================

# Target can disappear for this many frames before
# we switch from normal tracking to ReID recovery.
LOST_FRAME_GRACE = 10

# Minimum cosine similarity needed for a possible match.
REID_THRESHOLD = 0.60

# Number of consecutive good matches required before
# we actually accept the new track ID.
REID_CONFIRMATIONS_REQUIRED = 3


# ============================================================
# HELPER - RESET TARGET
# ============================================================

def reset_target():

    global target_id
    global target_embedding
    global lost_frames
    global searching
    global reid_confirmation_count

    # Clear everything related to the current target.
    target_id = None
    target_embedding = None

    lost_frames = 0
    searching = False
    reid_confirmation_count = 0

    print("TARGET RESET")


# ============================================================
# REID
# ============================================================

def get_embedding(person_crop):

    # Sometimes YOLO can give us an invalid crop.
    if person_crop is None or person_crop.size == 0:
        return None

    tensor = reid_transform(person_crop).unsqueeze(0)

    with torch.no_grad():

        embedding = reid_model(tensor)

    # Normalize the embedding so cosine similarity
    # behaves consistently.
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding


# ============================================================
# MOBILE WEB INTERFACE
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>TargetLock</title>

    <style>

        body {
            margin: 0;
            background: #111;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }

        h1 {
            margin: 18px 0 5px;
        }

        p {
            color: #aaa;
        }

        #camera {
            width: 95%;
            max-width: 900px;
            border-radius: 12px;
            margin-top: 15px;
        }

        button {
            margin: 8px;
            padding: 14px 22px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }

        #start {
            background: #2196f3;
            color: white;
        }

        #select {
            background: #00c853;
            color: white;
        }

        #stop {
            background: #f44336;
            color: white;
        }

        #status {
            margin-top: 12px;
            font-size: 18px;
            font-weight: bold;
        }

    </style>

</head>


<body>

    <h1>🎯 TargetLock</h1>

    <p>AI Persistent Subject Tracking</p>


    <button id="start">
        Open Camera
    </button>


    <button id="select">
        Select / Re-select Target
    </button>


    <button id="stop">
        Stop Tracking
    </button>


    <div id="status">
        Camera off
    </div>


    <video
        id="camera"
        autoplay
        playsinline>
    </video>


    <!-- I use this canvas to grab frames from the phone camera
         and send them to the Python backend. -->
    <canvas
        id="canvas"
        style="display:none;">
    </canvas>


    <script>

        // ========================================================
        // ELEMENTS
        // ========================================================

        const video = document.getElementById("camera");

        const canvas = document.getElementById("canvas");

        const startButton =
            document.getElementById("start");

        const selectButton =
            document.getElementById("select");

        const stopButton =
            document.getElementById("stop");

        const status =
            document.getElementById("status");


        // ========================================================
        // APP STATE
        // ========================================================

        let stream = null;

        // True when I am waiting for the user to tap
        // the person they want to track.
        let selecting = false;

        // True while frames are being sent to Python.
        let running = false;


        // ========================================================
        // OPEN CAMERA
        // ========================================================

        startButton.onclick = async () => {

            try {

                // Every time I open the camera, I want
                // a completely fresh target.
                await fetch("/reset", {
                    method: "POST"
                });

                selecting = false;


                // Ask the phone for camera access.
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "user"
                    },
                    audio: false
                });


                video.srcObject = stream;


                status.innerText =
                    "Camera active — select a target";


                running = true;


                // Start sending frames to the backend.
                processFrame();


            } catch (error) {

                console.error(error);

                status.innerText =
                    "Camera permission failed";

            }

        };


        // ========================================================
        // SELECT / RE-SELECT TARGET
        // ========================================================

        selectButton.onclick = () => {

            if (!running) {

                status.innerText =
                    "Open the camera first";

                return;
            }


            // The next tap on the video will choose
            // the person I want to follow.
            selecting = true;


            status.innerText =
                "Tap the person you want to track";

        };


        // ========================================================
        // STOP TRACKING
        // ========================================================

        stopButton.onclick = async () => {

            // Stop the target on the Python side.
            await fetch("/reset", {
                method: "POST"
            });


            selecting = false;


            status.innerText =
                "Camera active — select a target";

        };


        // ========================================================
        // TAP VIDEO TO SELECT PERSON
        // ========================================================

        video.onclick = async (event) => {

            // If I am not currently selecting a target,
            // ignore normal taps on the video.
            if (!selecting) {
                return;
            }


            const rect =
                video.getBoundingClientRect();


            // Convert the phone screen tap into
            // coordinates from the actual camera frame.
            const x =
                (event.clientX - rect.left)
                * video.videoWidth
                / rect.width;


            const y =
                (event.clientY - rect.top)
                * video.videoHeight
                / rect.height;


            const response = await fetch(
                `/select?x=${x}&y=${y}`,
                {
                    method: "POST"
                }
            );


            const result =
                await response.json();


            if (result.selected) {

                status.innerText =
                    "🎯 TARGET LOCKED";

                selecting = false;

            } else {

                status.innerText =
                    "No person detected there. Try again.";

            }

        };


        // ========================================================
        // SEND CAMERA FRAME TO PYTHON
        // ========================================================

        async function processFrame() {

            if (!running) {
                return;
            }


            // Match the canvas size to the camera frame.
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;


            const context =
                canvas.getContext("2d");


            // Copy the current camera frame into
            // the hidden canvas.
            context.drawImage(
                video,
                0,
                0,
                canvas.width,
                canvas.height
            );


            // Turn the frame into a JPEG.
            canvas.toBlob(async (blob) => {

                if (!blob) {

                    requestAnimationFrame(processFrame);

                    return;
                }


                const formData =
                    new FormData();


                formData.append(
                    "file",
                    blob,
                    "frame.jpg"
                );


                try {

                    // Send the camera frame to FastAPI.
                    const response =
                        await fetch(
                            "/process",
                            {
                                method: "POST",
                                body: formData
                            }
                        );


                    const result =
                        await response.json();


                    // ------------------------------------------------
                    // SHOW CURRENT TARGET STATE
                    // ------------------------------------------------

                    if (result.status === "locked") {

                        status.innerText =
                            "🎯 TARGET LOCKED";

                    }


                    else if (result.status === "lost") {

                        status.innerText =
                            "⚠️ TARGET LOST";

                    }


                    else if (result.status === "searching") {

                        status.innerText =
                            "🔎 REID SEARCHING...";

                    }


                    else {

                        status.innerText =
                            "Camera active — select a target";

                    }


                } catch (error) {

                    console.error(error);

                }


                // Keep processing frames.
                requestAnimationFrame(processFrame);


            }, "image/jpeg", 0.75);

        }

    </script>

</body>

</html>
"""


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return HTML


# ============================================================
# RESET TARGET
# ============================================================

@app.post("/reset")
def reset():

    # This is used when I open the camera,
    # stop tracking, or want to start fresh.
    reset_target()

    return {
        "reset": True
    }


# ============================================================
# SELECT TARGET
# ============================================================

@app.post("/select")
def select_target(x: float, y: float):

    global target_id
    global target_embedding
    global lost_frames
    global searching
    global reid_confirmation_count

    # Make sure I actually have a processed frame
    # to use for selecting the person.
    if latest_result is None or latest_frame is None:

        return {
            "selected": False
        }


    # If YOLO has not assigned track IDs yet,
    # I cannot select a tracked person.
    if latest_result.boxes.id is None:

        return {
            "selected": False
        }


    # Check every detected person.
    for box in latest_result.boxes:

        track_id = int(box.id[0])


        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        # Check if the user tapped inside this person's box.
        if x1 <= x <= x2 and y1 <= y <= y2:

            crop = latest_frame[
                y1:y2,
                x1:x2
            ]


            # Save this person's appearance.
            embedding = get_embedding(crop)


            if embedding is None:

                return {
                    "selected": False
                }


            # This person is now my target.
            target_id = track_id

            target_embedding = embedding


            # Reset the recovery state.
            lost_frames = 0
            searching = False
            reid_confirmation_count = 0


            print(
                f"TARGET SELECTED: ID {target_id}"
            )


            return {
                "selected": True,
                "target_id": target_id
            }


    # The user tapped somewhere that was not
    # inside a detected person.
    return {
        "selected": False
    }


# ============================================================
# LATEST FRAME STATE
# ============================================================

latest_result = None

latest_frame = None


# ============================================================
# PROCESS FRAME
# ============================================================

@app.post("/process")
async def process_frame(
    file: UploadFile = File(...)
):

    global latest_result
    global latest_frame

    global target_id
    global target_embedding

    global lost_frames
    global searching
    global reid_confirmation_count


    # ========================================================
    # READ IMAGE
    # ========================================================

    data = await file.read()


    image_array = np.frombuffer(
        data,
        dtype=np.uint8
    )


    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        return {
            "status": "error"
        }


    # Save the most recent frame.
    # The /select endpoint uses this when
    # the user taps someone.
    latest_frame = frame


    # ========================================================
    # YOLO + BOT-SORT
    # ========================================================

    results = yolo.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[0],       # 0 = person
        verbose=False
    )


    result = results[0]

    latest_result = result


    # ========================================================
    # CHECK IF TARGET IS STILL VISIBLE
    # ========================================================

    target_visible = False


    if (
        target_id is not None
        and result.boxes.id is not None
    ):

        for box in result.boxes:

            track_id = int(box.id[0])


            if track_id == target_id:

                # BoT-SORT still knows who my target is.
                target_visible = True


                # Reset the lost counter.
                lost_frames = 0

                searching = False

                reid_confirmation_count = 0

                break


    # ========================================================
    # TARGET LOST
    # ========================================================

    if target_id is not None and not target_visible:

        lost_frames += 1


        # For a few frames, I consider the target
        # temporarily lost.
        #
        # I don't want to immediately run ReID because
        # BoT-SORT might recover the same person by itself.
        if lost_frames < LOST_FRAME_GRACE:

            searching = False


        # Once the grace period is over,
        # start actively looking for the target.
        elif lost_frames >= LOST_FRAME_GRACE:

            searching = True


    # ========================================================
    # REID SEARCH
    # ========================================================

    if (
        searching
        and target_embedding is not None
        and result.boxes.id is not None
    ):

        best_similarity = -1

        best_candidate_id = None


        # Look at EVERY person currently visible.
        for box in result.boxes:

            candidate_id = int(box.id[0])


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            crop = frame[
                y1:y2,
                x1:x2
            ]


            candidate_embedding = \
                get_embedding(crop)


            if candidate_embedding is None:
                continue


            # Compare this person's appearance
            # to the saved target appearance.
            similarity = F.cosine_similarity(
                target_embedding,
                candidate_embedding
            ).item()


            print(
                f"Candidate {candidate_id}: "
                f"{similarity:.3f}"
            )


            # Keep the strongest candidate.
            if similarity > best_similarity:

                best_similarity = similarity

                best_candidate_id = candidate_id


        # ====================================================
        # CONFIRM REID MATCH
        # ====================================================

        if (
            best_candidate_id is not None
            and best_similarity >= REID_THRESHOLD
        ):

            # One strong match is not enough.
            reid_confirmation_count += 1


            if (
                reid_confirmation_count
                >= REID_CONFIRMATIONS_REQUIRED
            ):

                old_id = target_id


                # BoT-SORT gave the returning target
                # a new ID, so update our target ID.
                target_id = best_candidate_id


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
            # Start the confirmation process again.
            reid_confirmation_count = 0


    # ========================================================
    # STATUS
    # ========================================================

    if target_id is None:

        # Nobody has been selected yet.
        status = "ready"


    elif searching:

        # Target has been gone long enough that
        # we are actively using ReID.
        status = "searching"


    elif lost_frames > 0:

        # Target disappeared, but we haven't started
        # ReID yet because we are still inside the grace period.
        status = "lost"


    else:

        # BoT-SORT currently has the target.
        status = "locked"


    return {
        "status": status,
        "target_id": target_id
    }