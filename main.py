from flask import Flask, render_template, request, send_from_directory, jsonify
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the trained model
model = load_model('models/model.h5')

# Class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type and get probabilities
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0] * 100  # Convert to percentage

    if class_labels[predicted_class_index] == 'notumor':
        result = "No Tumor"
    else:
        result = f"Tumor: {class_labels[predicted_class_index]}"
    return result, confidence_score, predictions[0]  # Return raw predictions

# Generate PDF report with image, optimized for one page with dim background
def generate_pdf_report(patient_name, age, sex, result, confidence, image_path, filename):
    pdf_path = os.path.join(UPLOAD_FOLDER, f"{os.path.splitext(filename)[0]}_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Create a canvas to set background color
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFillColorRGB(0.95, 0.95, 0.95)  # Dim gray background (RGB: 242, 242, 242)
    c.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    c.showPage()

    # Title with larger font
    story.append(Paragraph("MRI Tumor Detection Report", styles['Heading1']))
    story.append(Spacer(1, 12))

    # Patient Information with adjusted spacing
    story.append(Paragraph(f"Patient Name: {patient_name}", styles['Normal']))
    story.append(Paragraph(f"Age: {age}", styles['Normal']))
    story.append(Paragraph(f"Sex: {sex}", styles['Normal']))
    story.append(Spacer(1, 6))

    # Diagnosis with larger font
    story.append(Paragraph(f"Diagnosis: {result}", styles['Heading2']))
    story.append(Paragraph(f"Confidence Level: {confidence:.2f}%", styles['Normal']))
    story.append(Spacer(1, 6))

    # Add Image with larger size
    img = Image(image_path, width=300, height=300)
    img.drawHeight = 300 * img.drawHeight / img.drawWidth
    img.drawWidth = 300
    story.append(img)
    story.append(Spacer(1, 6))

    # Disclaimer with readable font
    story.append(Paragraph("Disclaimer: This report is for informational purposes only and should be reviewed by a qualified medical professional for an accurate diagnosis.", styles['Italic']))
    story.append(Spacer(1, 12))

    # Ensure single page by building directly
    doc.build(story)
    return pdf_path

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle patient info and file upload
        patient_name = request.form.get('patient_name', 'Unknown')
        age = request.form.get('age', 'N/A')
        sex = request.form.get('sex', 'N/A')
        file = request.files.get('file')
        if file:
            # Clear previous uploads if requested
            if request.form.get('clear'):
                for f in os.listdir(UPLOAD_FOLDER):
                    os.remove(os.path.join(UPLOAD_FOLDER, f))
                return render_template('index.html', result=None)

            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence, _ = predict_tumor(file_location)

            # Generate PDF report
            pdf_path = generate_pdf_report(patient_name, age, sex, result, confidence, file_location, file.filename)
            pdf_url = f'/uploads/{os.path.basename(pdf_path)}'

            # Return result along with image path and PDF URL
            return render_template('index.html', result=result, confidence=confidence, file_path=f'/uploads/{file.filename}', patient_name=patient_name, age=age, sex=sex, pdf_url=pdf_url)

    return render_template('index.html', result=None)

# Route to serve uploaded files and PDF
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)