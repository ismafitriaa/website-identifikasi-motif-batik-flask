from flask import Flask, render_template, request, redirect, url_for
from flask_mysqldb import MySQL
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import base64

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'batik'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
mysql = MySQL(app)

# Function to calculate total pages
def calculate_total_pages(total_records, records_per_page):
    return (total_records + records_per_page - 1) // records_per_page

@app.route("/")
def main():
    cur = mysql.connection.cursor()
    # Get the keyword from the query string
    keyword = request.args.get('keyword', '')

    # If keyword is provided, filter the results
    if keyword:
        cur.execute("SELECT COUNT(*) FROM data WHERE namaBatik LIKE %s", ('%' + keyword + '%',))
    else:
        cur.execute("SELECT COUNT(*) FROM data")

    total_records = cur.fetchone()[0]

    page = request.args.get('page', 1, type=int)
    records_per_page = 6
    total_pages = calculate_total_pages(total_records, records_per_page)
    offset = (page - 1) * records_per_page

     # If keyword is provided, filter the results
    if keyword:
        cur.execute("SELECT * FROM data WHERE namaBatik LIKE %s LIMIT %s OFFSET %s", ('%' + keyword + '%', records_per_page, offset))
    else:
        cur.execute("SELECT * FROM data LIMIT %s OFFSET %s", (records_per_page, offset))

    gambar = cur.fetchall()
    cur.close()
    
    images = []
    for row in gambar:
        image_blob = row[2]
        image_base64 = base64.b64encode(image_blob).decode('utf-8')
        images.append((row[0], row[1], image_base64, row[3]))

    return render_template('index.html', data=images, page=page, total_pages=total_pages, keyword=keyword)

def classify_motif(image_path, model):
    # Load and preprocess the input image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Move the input image to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
    
    # Convert output probabilities to predicted class
    _, predicted = torch.max(output, 1)
    
    # Map the predicted class index to the class name
    class_names = ['Alas-Alasan', 'Angkucamala Puspa Padwa', 'Ayam Puger', 'Babon Angrem', 'Bali', 'Banyumasan', 'Bondhet',
                   'Buket Pakis', 'Buntal', 'Candi Baruna', 'Candi Jago', 'Cemukiran', 'Ceplok', 'Ceplok Burba', 'Ceplok Candi Luhur',
                   'Ceplok Lung Kestlop', 'Daniswara Jiwatrisna Nirbaya', 'Daniswara Jiwatrisna Patibrata', 'Gandring Wirasena Hambangun Negari',
                   'Ista Malang Kucecwara', 'Kastara Cakra Gama', 'Lasem', 'Liris Cemeng', 'Lung-Lungan', 'Malang Heritage', 'Meru', 'Ole-Ole',
                   'Parang Barong', 'Parang Barong Seling Huk', 'Parang Curiga', 'Parang Gendreh', 'Parang Kesit Barong', 'Parang Klithik Glebag Seruni',
                   'Parang Klithik Seling Ceplok', 'Parang Kusumo', 'Parang Sarpo', 'Parang Sondher', 'Peksi Manyura', 'Rujak Sente',
                   'Sapanti Nata', 'Sawat Manak', 'Sawat Suri', 'Sekar Jagad', 'Sekartaji Prameswari Juwita', 'Semen Giring', 'Sido Luhur', 'Sido Mukti',
                   'Sido Mulyo', 'Srikaton', 'Tambal', 'Topeng Gandring Wirasena', 'Tuntrum', 'Wahyu Tumurun']
    
    # Check if motif detected
    if predicted.item() < len(class_names):
        predicted_class = class_names[predicted.item()]
    else:
        predicted_class = "Tidak ada motif yang terdeteksi"
    
    return predicted_class

@app.route("/upload", methods=['POST'])
def upload():
    if 'gambar' not in request.files:
        return redirect(url_for('main'))
    
    file = request.files['gambar']
    if file.filename == '':
        return redirect(url_for('main'))
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Save the uploaded image
        
        # Load the classification model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=False)
        old_conv = model.features[0]
        model.features[0] = nn.Sequential(
                nn.Dropout(p=0.25),
                old_conv)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 53)
        model.load_state_dict(torch.load('models/model.pt', map_location=device))
        model.eval()
        model = model.to(device)

        # Classify motif
        prediction = classify_motif(file_path, model)

        # Get the page parameter from the URL, default to 1 if not provided
        page = request.args.get('page', 1, type=int)

        cur = mysql.connection.cursor()
        cur.execute("SELECT COUNT(*) FROM data")
        total_records = cur.fetchone()[0]  # Get total number of records

        # Get the page parameter from the URL, default to 1 if not provided
        page = request.args.get('page', 1, type=int)

        # Assuming 6 records per page, calculate total pages
        records_per_page = 6
        total_pages = calculate_total_pages(total_records, records_per_page)

        uploaded_image_url = '/' + file_path.replace('\\', '/')

        cur.execute("SELECT COUNT(*) FROM data")

        total_records = cur.fetchone()[0]

        page = request.args.get('page', 1, type=int)
        offset = (page - 1) * records_per_page

        cur.execute("SELECT * FROM data LIMIT %s OFFSET %s", (records_per_page, offset))

        gambar = cur.fetchall()
        cur.close()
        
        images = []
        for row in gambar:
            image_blob = row[2]
            image_base64 = base64.b64encode(image_blob).decode('utf-8')
            images.append((row[0], row[1], image_base64, row[3]))

        return render_template('index.html', data=images, uploaded_image=uploaded_image_url, motif=prediction, page=page, total_pages=total_pages)
        
    return redirect(url_for('main'))


if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
