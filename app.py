# app.py - Flask backend for the object detection web app
# uses gRPC to talk to tensorflow serving and draws bounding boxes on images

import os
import time
import uuid
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, redirect, url_for, flash, Response

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

# prometheus stuff for monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# setup flask
app = Flask(__name__)
app.secret_key = "mlops-secret-key-change-later"

# only allow these file types
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# folder paths
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "originals")
RESULTS_FOLDER = os.path.join(app.root_path, "static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# tf serving connection settings
TF_SERVING_HOST = os.environ.get("TF_SERVING_HOST", "localhost")
TF_SERVING_PORT = os.environ.get("TF_SERVING_PORT", "8500")
TF_SERVING_ENDPOINT = f"{TF_SERVING_HOST}:{TF_SERVING_PORT}"

# model settings
MODEL_NAME = os.environ.get("MODEL_NAME", "ssd_mobilenet_v2")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# coco labels - these are the 90 classes the model can detect
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
    33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
    51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange",
    56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
    61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
    75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush"
}

# colours for bounding boxes
BBOX_COLOURS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFD700", "#FF00FF",
    "#00FFFF", "#FF6347", "#32CD32", "#8A2BE2", "#FF8C00"
]

# ---- metrics storage ----
# keeps track of inference stats in memory
metrics_store = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_inference_time_ms": 0.0,
    "avg_inference_time_ms": 0.0,
    "total_objects_detected": 0,
    "class_distribution": defaultdict(int),
    "recent_inferences": [],
    "service_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "last_inference_time": None,
    "model_name": MODEL_NAME,
    "confidence_threshold": CONFIDENCE_THRESHOLD
}

# prometheus metrics - these get scraped by prometheus every 15s
PROM_REQUESTS = Counter("inference_requests_total", "Total inference requests", ["status"])
PROM_LATENCY = Histogram("inference_latency_ms", "Inference latency in ms",
                         buckets=[50, 100, 200, 500, 1000, 2000, 5000])
PROM_DETECTIONS = Counter("objects_detected_total", "Objects detected by class", ["class_name"])


# ---- helper functions ----

def allowed_file(filename):
    # check if the file extension is one we accept
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    # load image, convert to RGB, make it a numpy array with batch dimension
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)
    input_tensor = np.expand_dims(image_np, axis=0)  # add batch dimention
    return input_tensor, image


def run_inference_grpc(input_tensor):
    # send image to tf serving over grpc and get back detections
    channel = grpc.insecure_channel(
        TF_SERVING_ENDPOINT,
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024)
        ]
    )
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # build the grpc request
    grpc_request = predict_pb2.PredictRequest()
    grpc_request.model_spec.name = MODEL_NAME
    grpc_request.model_spec.signature_name = "serving_default"
    grpc_request.inputs["input_tensor"].CopyFrom(
        tf.make_tensor_proto(input_tensor, dtype=tf.uint8)
    )

    # send it with a 30 second timeout
    response = stub.Predict(grpc_request, timeout=30.0)

    # pull out the results
    boxes = tf.make_ndarray(response.outputs["detection_boxes"])
    classes = tf.make_ndarray(response.outputs["detection_classes"])
    scores = tf.make_ndarray(response.outputs["detection_scores"])
    num_detections = int(tf.make_ndarray(response.outputs["num_detections"])[0])
    channel.close()

    return {
        "detection_boxes": boxes[0],
        "detection_classes": classes[0],
        "detection_scores": scores[0],
        "num_detections": num_detections
    }


def draw_bounding_boxes(image, detections):
    # draw boxes and labels on the image for each detection
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    # try to load a nice font, fall back to default if it doesnt exist
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    for idx, det in enumerate(detections):
        colour = BBOX_COLOURS[idx % len(BBOX_COLOURS)]

        # convert normalised coords to pixel coords
        ymin, xmin, ymax, xmax = det["box"]
        left = int(xmin * width)
        top = int(ymin * height)
        right = int(xmax * width)
        bottom = int(ymax * height)

        # draw the box
        draw.rectangle([left, top, right, bottom], outline=colour, width=3)

        # draw label with background
        label = f'{det["class_name"]} ({det["score"]:.1%})'
        text_bbox = draw.textbbox((left, top), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        draw.rectangle([left, top - th - 6, left + tw + 6, top], fill=colour)
        draw.text((left + 3, top - th - 4), label, fill="white", font=font)

    return annotated


def update_metrics(inference_time_ms, detections, success=True):
    # update the metrics store and prometheus counters after each inference
    metrics_store["total_requests"] += 1

    if success:
        metrics_store["successful_requests"] += 1
        metrics_store["total_inference_time_ms"] += inference_time_ms
        metrics_store["avg_inference_time_ms"] = (
            metrics_store["total_inference_time_ms"] / metrics_store["successful_requests"]
        )
        metrics_store["total_objects_detected"] += len(detections)
        for det in detections:
            metrics_store["class_distribution"][det["class_name"]] += 1

        # keep last 15 inferences for the log
        metrics_store["recent_inferences"].append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "inference_time_ms": round(inference_time_ms, 2),
            "objects_detected": len(detections),
            "classes": [d["class_name"] for d in detections]
        })
        if len(metrics_store["recent_inferences"]) > 15:
            metrics_store["recent_inferences"].pop(0)

        metrics_store["last_inference_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # update prometheus
        PROM_REQUESTS.labels(status="success").inc()
        PROM_LATENCY.observe(inference_time_ms)
        for det in detections:
            PROM_DETECTIONS.labels(class_name=det["class_name"]).inc()
    else:
        metrics_store["failed_requests"] += 1
        PROM_REQUESTS.labels(status="failure").inc()


# ---- routes ----

@app.route("/")
def index():
    # home page with upload form
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # handle image upload, run inference, show results

    # check if a file was actually uploaded
    if "image" not in request.files:
        flash("No file selected!", "error")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected!", "error")
        return redirect(url_for("index"))

    # check extension
    if not allowed_file(file.filename):
        flash("Invalid file type! Only .jpg, .jpeg and .png allowed.", "error")
        return redirect(url_for("index"))

    # save with a unique name so files dont overwrite each other
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{uuid.uuid4().hex[:8]}.{ext}"
    original_name = file.filename

    original_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(original_path)

    try:
        # preprocess and run inference
        input_tensor, pil_image = preprocess_image(original_path)

        start = time.time()
        raw_results = run_inference_grpc(input_tensor)
        inference_ms = (time.time() - start) * 1000

        # filter by confidence threshold
        detections = []
        for i in range(raw_results["num_detections"]):
            score = float(raw_results["detection_scores"][i])
            if score >= CONFIDENCE_THRESHOLD:
                class_id = int(raw_results["detection_classes"][i])
                detections.append({
                    "class_name": COCO_LABELS.get(class_id, f"class_{class_id}"),
                    "score": score,
                    "box": raw_results["detection_boxes"][i].tolist()
                })

        # draw boxes and save
        annotated = draw_bounding_boxes(pil_image, detections)
        result_name = f"result_{safe_name}"
        annotated.save(os.path.join(RESULTS_FOLDER, result_name))

        # update metrics
        update_metrics(inference_ms, detections, success=True)

        return render_template("results.html", results={
            "image_name": original_name,
            "original_image": f"originals/{safe_name}",
            "result_image": f"results/{result_name}",
            "detections": detections,
            "inference_time_ms": round(inference_ms, 2),
            "num_detections": len(detections)
        })

    except grpc.RpcError as e:
        logger.error("gRPC error: %s", e)
        update_metrics(0, [], success=False)
        flash("Could not connect to model server. Is TF Serving running?", "error")
        return redirect(url_for("index"))

    except Exception as e:
        logger.error("Error: %s", str(e))
        update_metrics(0, [], success=False)
        flash(f"Something went wrong: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/metrics")
def metrics():
    # metrics dashboard page
    class_dist = dict(metrics_store["class_distribution"])
    sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)

    return render_template("metrics.html", metrics={
        "total_requests": metrics_store["total_requests"],
        "successful_requests": metrics_store["successful_requests"],
        "failed_requests": metrics_store["failed_requests"],
        "avg_inference_time_ms": round(metrics_store["avg_inference_time_ms"], 2),
        "total_objects_detected": metrics_store["total_objects_detected"],
        "class_distribution": sorted_classes,
        "recent_inferences": metrics_store["recent_inferences"],
        "service_start_time": metrics_store["service_start_time"],
        "last_inference_time": metrics_store["last_inference_time"] or "None yet",
        "model_name": metrics_store["model_name"],
        "confidence_threshold": metrics_store["confidence_threshold"],
        "success_rate": (
            round(metrics_store["successful_requests"] / metrics_store["total_requests"] * 100, 1)
            if metrics_store["total_requests"] > 0 else 0
        )
    })


@app.route("/prometheus_metrics")
def prometheus_metrics():
    # endpoint that prometheus scrapes for metrics data
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    logger.info("Starting app - TF Serving at %s, Model: %s", TF_SERVING_ENDPOINT, MODEL_NAME)
    app.run(host="0.0.0.0", port=5000, debug=True)
