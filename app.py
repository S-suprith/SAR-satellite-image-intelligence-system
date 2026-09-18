import os
import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, send_file, url_for
from werkzeug.utils import secure_filename

from model import UNetColorization

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# --------------------------------------------------
# FLASK APP
# --------------------------------------------------

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = UNetColorization().to(device)

model.load_state_dict(
    torch.load(
        "color_model.pth",
        map_location=device
    )
)

model.eval()


# --------------------------------------------------
# FOLDERS
# --------------------------------------------------

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --------------------------------------------------
# LAND FEATURE DETECTION
# --------------------------------------------------

def detect_land_features(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    total_pixels = image.shape[0] * image.shape[1]

    # Vegetation
    veg_mask = cv2.inRange(
        hsv,
        (35, 40, 40),
        (85, 255, 255)
    )

    # Water
    water_mask = cv2.inRange(
        hsv,
        (90, 50, 50),
        (130, 255, 255)
    )

    # Roads
    road_mask = cv2.inRange(
        gray,
        150,
        200
    )

    # Buildings
    building_mask = cv2.inRange(
        gray,
        200,
        255
    )

    veg_pixels = cv2.countNonZero(veg_mask)
    water_pixels = cv2.countNonZero(water_mask)
    road_pixels = cv2.countNonZero(road_mask)
    building_pixels = cv2.countNonZero(building_mask)

    vegetation = (veg_pixels / total_pixels) * 100
    water = (water_pixels / total_pixels) * 100
    roads = (road_pixels / total_pixels) * 100
    buildings = (building_pixels / total_pixels) * 100

    other = 100 - (
        vegetation +
        water +
        roads +
        buildings
    )

    # Prevent negative percentage
    other = max(0, other)

    return (
        vegetation,
        water,
        roads,
        buildings,
        other
    )


# --------------------------------------------------
# PIE CHART
# --------------------------------------------------

def create_pie_chart(
    veg,
    water,
    roads,
    buildings,
    other
):

    labels = [
        "Vegetation",
        "Water",
        "Roads",
        "Buildings",
        "Other"
    ]

    sizes = [
        veg,
        water,
        roads,
        buildings,
        other
    ]

    colors = [
        "green",
        "blue",
        "gray",
        "orange",
        "#8B4513"
    ]

    plt.figure(figsize=(6, 6))

    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={
            "edgecolor": "black"
        }
    )

    plt.title("Satellite Land Distribution")

    chart_path = os.path.join(
        OUTPUT_FOLDER,
        "land_chart.png"
    )

    plt.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return chart_path


# --------------------------------------------------
# PDF REPORT
# --------------------------------------------------

def create_pdf_report(
    original_image,
    processed_image,
    chart,
    veg,
    water,
    roads,
    buildings,
    other
):

    report_path = os.path.join(
        OUTPUT_FOLDER,
        "satellite_report.pdf"
    )

    styles = getSampleStyleSheet()

    elements = []

    # Title
    elements.append(
        Paragraph(
            "Satellite Image Intelligence Report",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(1, 20)
    )

    # Land Distribution
    elements.append(
        Paragraph(
            "Land Distribution Analysis",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            f"Vegetation : {veg:.2f} %",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Water : {water:.2f} %",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Roads : {roads:.2f} %",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Buildings : {buildings:.2f} %",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Other : {other:.2f} %",
            styles["Normal"]
        )
    )

    elements.append(
        Spacer(1, 30)
    )

    # Original Image
    elements.append(
        Paragraph(
            "Original SAR Image",
            styles["Heading2"]
        )
    )

    elements.append(
        Image(
            original_image,
            width=350,
            height=350
        )
    )

    elements.append(
        Spacer(1, 20)
    )

    # Colorized Image
    elements.append(
        Paragraph(
            "AI Colorized Image",
            styles["Heading2"]
        )
    )

    elements.append(
        Image(
            processed_image,
            width=350,
            height=350
        )
    )

    elements.append(
        Spacer(1, 20)
    )

    # Chart
    elements.append(
        Paragraph(
            "Land Distribution Chart",
            styles["Heading2"]
        )
    )

    elements.append(
        Image(
            chart,
            width=350,
            height=350
        )
    )

    # Build PDF
    pdf = SimpleDocTemplate(
        report_path
    )

    pdf.build(elements)

    return report_path


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_image():

    # Check file
    if "image" not in request.files:
        return redirect("/")

    file = request.files["image"]

    # Check filename
    if file.filename == "":
        return redirect("/")

    filename = secure_filename(
        file.filename
    )

    # Input path
    input_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(input_path)

    # --------------------------------------------------
    # READ SAR IMAGE
    # --------------------------------------------------

    img = cv2.imread(
        input_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        return "Unable to read image", 400

    img = cv2.resize(
        img,
        (256, 256)
    )

    # --------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------

    L = img / 255.0

    L_tensor = torch.tensor(
        L,
        dtype=torch.float32
    )

    L_tensor = (
        L_tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------

    with torch.no_grad():

        pred_ab = model(
            L_tensor
        )

    pred_ab = (
        pred_ab
        .squeeze(0)
        .cpu()
        .numpy()
    )

    pred_ab = np.transpose(
        pred_ab,
        (1, 2, 0)
    )

    # --------------------------------------------------
    # CONVERT LAB TO BGR
    # --------------------------------------------------

    L_uint8 = (
        L * 255
    ).astype(
        np.uint8
    )

    AB = (
        pred_ab + 1
    ) * 128

    AB = np.clip(
        AB,
        0,
        255
    ).astype(
        np.uint8
    )

    lab = np.zeros(
        (256, 256, 3),
        dtype=np.uint8
    )

    lab[:, :, 0] = L_uint8
    lab[:, :, 1:] = AB

    colorized = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # --------------------------------------------------
    # SAVE COLORIZED IMAGE
    # --------------------------------------------------

    output_filename = (
        "processed_" + filename
    )

    output_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        output_filename
    )

    cv2.imwrite(
        output_path,
        colorized
    )

    # --------------------------------------------------
    # LAND FEATURE ANALYSIS
    # --------------------------------------------------

    (
        veg,
        water,
        roads,
        buildings,
        other
    ) = detect_land_features(
        colorized
    )

    # --------------------------------------------------
    # CREATE PIE CHART
    # --------------------------------------------------

    chart_path = create_pie_chart(
        veg,
        water,
        roads,
        buildings,
        other
    )

    # --------------------------------------------------
    # CREATE PDF REPORT
    # --------------------------------------------------

    report_path = create_pdf_report(
        input_path,
        output_path,
        chart_path,
        veg,
        water,
        roads,
        buildings,
        other
    )

    # --------------------------------------------------
    # STATIC URLS
    # --------------------------------------------------

    input_url = url_for(
        "static",
        filename=f"uploads/{filename}"
    )

    output_url = url_for(
        "static",
        filename=f"outputs/{output_filename}"
    )

    chart_url = url_for(
        "static",
        filename="outputs/land_chart.png"
    )

    report_filename = os.path.basename(
        report_path
    )

    report_url = url_for(
        "download_file",
        filename=report_filename
    )

    # --------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------

    return render_template(
        "index.html",

        input_image=input_url,

        output_image=output_url,

        vegetation=round(
            veg,
            2
        ),

        water=round(
            water,
            2
        ),

        roads=round(
            roads,
            2
        ),

        buildings=round(
            buildings,
            2
        ),

        other=round(
            other,
            2
        ),

        chart=chart_url,

        report=report_url
    )


# --------------------------------------------------
# DOWNLOAD FILE
# --------------------------------------------------

@app.route(
    "/download/<filename>"
)
def download_file(filename):

    path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        filename
    )

    if not os.path.exists(path):
        return "File not found", 404

    return send_file(
        path,
        as_attachment=True
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )