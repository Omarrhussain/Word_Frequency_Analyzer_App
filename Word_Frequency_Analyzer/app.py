from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import os
import matplotlib.pyplot as plt
from analyzer import WordAnalyzer
from nltk_analyzer import NLTKWordAnalyzer
from werkzeug.utils import secure_filename

Upload_Folder = 'uploads'
Chart_Folder = 'charts'
Chart_Name = 'word_frequency.png'
Chart_Path = os.path.join(Chart_Folder, Chart_Name)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Upload_Folder
app.config['CHART_FOLDER'] = Chart_Folder

os.makedirs(Upload_Folder, exist_ok=True)
os.makedirs(Chart_Folder, exist_ok=True)

#route to serve chartimages
@app.route('/charts/<path:filename>')
def serve_chart(filename):
    return send_from_directory(Chart_Folder, filename)

@app.route("/",methods=["GET","POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        analyzer_type = request.form.get('analyzer_type', 'simple')  
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            chart_filename = f'word_frequency_{analyzer_type}.png'
            chart_path = os.path.join(Chart_Folder, chart_filename)
            
            if analyzer_type == 'nltk':
                analyzer = NLTKWordAnalyzer()
                word_counts = analyzer.analyze(filepath, top_n=10)
                top_words = dict(word_counts)
                plt.figure(figsize=(8, 5))
                plt.bar([word for word, _ in word_counts], [count for _, count in word_counts])
                plt.xticks(rotation=45)
                plt.title('Word Frequency Distribution (NLTK)')
                plt.xlabel('Words')
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
            else:  
                analyzer = WordAnalyzer()
                analyzer.analyze(filepath)
                top_words = analyzer.get_top_words()
                analyzer.create_chart(chart_path, 10)
            chart_url = url_for('serve_chart', filename=chart_filename)
            return render_template('index.html', 
                               top_words=top_words, 
                               chart_path=chart_url,
                               analyzer_type=analyzer_type)        
    return render_template('index.html', 
                         top_words=None, 
                         chart_path=None, 
                         analyzer_type='simple')       
 
if __name__ == "__main__":
    app.run(debug=True)