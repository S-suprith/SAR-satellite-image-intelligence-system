import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, send_file
from werkzeug.utils import secure_filename
from model import UNetColorization

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet


app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetColorization().to(device)
model.load_state_dict(torch.load("color_model.pth", map_location=device))
model.eval()


UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ----------------------------------
# Land Feature Detection
# ----------------------------------
def detect_land_features(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    total_pixels = image.shape[0] * image.shape[1]

    veg_mask = cv2.inRange(hsv,(35,40,40),(85,255,255))
    water_mask = cv2.inRange(hsv,(90,50,50),(130,255,255))
    road_mask = cv2.inRange(gray,150,200)
    building_mask = cv2.inRange(gray,200,255)

    veg_pixels = cv2.countNonZero(veg_mask)
    water_pixels = cv2.countNonZero(water_mask)
    road_pixels = cv2.countNonZero(road_mask)
    building_pixels = cv2.countNonZero(building_mask)

    vegetation = (veg_pixels / total_pixels) * 100
    water = (water_pixels / total_pixels) * 100
    roads = (road_pixels / total_pixels) * 100
    buildings = (building_pixels / total_pixels) * 100

    other = 100 - (vegetation + water + roads + buildings)

    return vegetation, water, roads, buildings, other


# ----------------------------------
# Damage Detection
# ----------------------------------
def detect_damage(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    damage_mask = cv2.inRange(gray,0,60)

    damage_pixels = cv2.countNonZero(damage_mask)

    total_pixels = image.shape[0] * image.shape[1]

    damage_percent = (damage_pixels / total_pixels) * 100

    if damage_percent > 5:
        status = "Possible structural damage detected"
    else:
        status = "Natural landscape"

    return round(damage_percent,2), status


# ----------------------------------
# Segmentation Overlay
# ----------------------------------
def create_segmentation_overlay(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    overlay = image.copy()

    veg_mask = cv2.inRange(hsv,(35,40,40),(85,255,255))
    water_mask = cv2.inRange(hsv,(90,50,50),(130,255,255))
    road_mask = cv2.inRange(gray,150,200)
    building_mask = cv2.inRange(gray,200,255)
    farm_mask = cv2.inRange(hsv,(20,40,40),(35,255,255))

    overlay[veg_mask>0] = (0,255,0)
    overlay[water_mask>0] = (255,0,0)
    overlay[road_mask>0] = (128,128,128)
    overlay[building_mask>0] = (0,0,255)
    overlay[farm_mask>0] = (0,255,255)

    result = cv2.addWeighted(image,0.6,overlay,0.4,0)

    return result


# ----------------------------------
# Pie Chart
# ----------------------------------
def create_pie_chart(veg, water, roads, buildings, other):

    labels = ["Vegetation","Water","Roads","Buildings","Other"]
    sizes = [veg, water, roads, buildings, other]

    colors = ["green","blue","gray","orange","#8B4513"]

    plt.figure(figsize=(6,6))

    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor':'black'}
    )

    plt.title("Satellite Land Distribution")

    chart_path = "static/outputs/land_chart.png"

    plt.savefig(chart_path, dpi=300)
    plt.close()

    return chart_path


# ----------------------------------
# NEW PDF REPORT
# ----------------------------------
def create_pdf_report(original_image, processed_image, overlay_image, chart,
                      veg, water, roads, buildings, other,
                      damage, status):

    report_path = "static/outputs/satellite_report.pdf"

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("Satellite Image Intelligence Report", styles['Title']))
    elements.append(Spacer(1,20))

    elements.append(Paragraph("Land Distribution Analysis", styles['Heading2']))

    elements.append(Paragraph(f"Vegetation : {veg:.2f} %", styles['Normal']))
    elements.append(Paragraph(f"Water : {water:.2f} %", styles['Normal']))
    elements.append(Paragraph(f"Roads : {roads:.2f} %", styles['Normal']))
    elements.append(Paragraph(f"Buildings : {buildings:.2f} %", styles['Normal']))
    elements.append(Paragraph(f"Other : {other:.2f} %", styles['Normal']))

    elements.append(Spacer(1,20))

    elements.append(Paragraph("Damage Detection", styles['Heading2']))
    elements.append(Paragraph(f"Destroyed Area : {damage:.2f} %", styles['Normal']))
    elements.append(Paragraph(f"Status : {status}", styles['Normal']))

    elements.append(Spacer(1,30))

    elements.append(Paragraph("Original Satellite Image", styles['Heading2']))
    elements.append(Image(original_image, width=350, height=350))

    elements.append(Spacer(1,20))

    elements.append(Paragraph("AI Colorized Image", styles['Heading2']))
    elements.append(Image(processed_image, width=350, height=350))

    elements.append(Spacer(1,20))

    elements.append(Paragraph("Land Segmentation Map", styles['Heading2']))
    elements.append(Image(overlay_image, width=350, height=350))

    elements.append(Spacer(1,20))

    elements.append(Paragraph("Land Distribution Chart", styles['Heading2']))
    elements.append(Image(chart, width=350, height=350))

    pdf = SimpleDocTemplate(report_path)
    pdf.build(elements)

    return report_path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return redirect("/")

    file = request.files["image"]

    filename = secure_filename(file.filename)

    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(input_path)

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(256,256))

    L = img / 255.0
    L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_ab = model(L_tensor)

    pred_ab = pred_ab.squeeze(0).cpu().numpy()
    pred_ab = np.transpose(pred_ab,(1,2,0))

    L = (L*255).astype(np.uint8)
    AB = (pred_ab+1)*128

    lab = np.zeros((256,256,3))
    lab[:,:,0] = L
    lab[:,:,1:] = AB
    lab = lab.astype(np.uint8)

    colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    output_filename = "processed_"+filename
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

    cv2.imwrite(output_path,colorized)


    overlay_img = create_segmentation_overlay(colorized)

    overlay_filename = "overlay_"+filename
    overlay_path = os.path.join(app.config["OUTPUT_FOLDER"],overlay_filename)

    cv2.imwrite(overlay_path,overlay_img)


    veg, water, roads, buildings, other = detect_land_features(colorized)

    damage_percent, damage_status = detect_damage(colorized)

    chart_path = create_pie_chart(veg, water, roads, buildings, other)


    report_path = create_pdf_report(
        input_path,
        output_path,
        overlay_path,
        chart_path,
        veg,
        water,
        roads,
        buildings,
        other,
        damage_percent,
        damage_status
    )


    return render_template(
        "index.html",
        input_image=input_path,
        output_image=output_path,
        overlay_image=overlay_path,
        vegetation=round(veg,2),
        water=round(water,2),
        roads=round(roads,2),
        buildings=round(buildings,2),
        other=round(other,2),
        damage=damage_percent,
        damage_status=damage_status,
        chart=chart_path,
        report=report_path
    )


@app.route("/download/<filename>")
def download_file(filename):

    path = os.path.join(app.config["OUTPUT_FOLDER"], filename)

    if not os.path.exists(path):
        return "File not found",404

    return send_file(path,as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)