from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session, flash
import numpy as np
from PIL import Image
import io
import os
import json
import logging
from datetime import datetime, timedelta
import base64
from werkzeug.utils import secure_filename
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
import torch.nn as nn

# Email imports and SMTP configuration
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# SMTP settings (update these with your actual sender email and password)
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SENDER_EMAIL = 'veerareddygajulapalli@gmail.com'  # Replace with your sender email
SENDER_PASSWORD = 'Veera@123'       # Replace with your email password or app password
RECEIVER_EMAIL = '99220040560@klu.ac.in'      # The recipient email

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
app.permanent_session_lifetime = timedelta(days=365)  # Set session lifetime to 1 year
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set max file size to 16MB
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
app.config['REQUEST_TIMEOUT'] = None  # Disable request timeout

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load language files with error handling
try:
    # Load class indices
    with open('class_indices.json', 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    logger.info("Class indices loaded successfully")

    # Load weed information from multiple files
    weed_info = {'en': {}, 'te': {}, 'ta': {}, 'hi': {}}
    for i in range(1, 5):
        try:
            with open(f'weed_info_part{i}.json', 'r', encoding='utf-8') as f:
                part_data = json.load(f)
                # Merge each language section separately
                for lang in part_data:
                    if lang in weed_info:
                        weed_info[lang].update(part_data[lang])
                logger.info(f"Loaded weed_info_part{i}.json successfully")
                logger.info(f"Part {i} contains languages: {list(part_data.keys())}")
                logger.info(f"Part {i} contains weeds: {list(part_data.get('en', {}).keys())}")
        except FileNotFoundError:
            logger.error(f"Could not find weed_info_part{i}.json")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing weed_info_part{i}.json: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading weed_info_part{i}.json: {str(e)}")
            raise

    # Log the total number of weeds loaded for each language
    for lang in weed_info:
        logger.info(f"Total weeds loaded for {lang}: {len(weed_info[lang])}")
        logger.info(f"Weeds available in {lang}: {list(weed_info[lang].keys())}")
    
    # Validate language data
    required_languages = {'en', 'te', 'ta', 'hi'}
    available_languages = set(weed_info.keys())
    if not required_languages.issubset(available_languages):
        missing_languages = required_languages - available_languages
        logger.warning(f"Missing translations for languages: {missing_languages}")
    
    # Load chemical products information
    with open('chemical_products.json', 'r', encoding='utf-8') as f:
        chemical_products = json.load(f)
    logger.info("Chemical products loaded successfully")

    # Load team information
    with open('team_info.json', 'r', encoding='utf-8') as f:
        team_info = json.load(f)
    logger.info("Team information loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading data files: {str(e)}")
    raise

# Load translations
try:
    with open('translations.json', 'r', encoding='utf-8') as f:
        translations = json.load(f)
    logger.info("Translations loaded successfully")
except Exception as e:
    logger.error(f"Error loading translations: {str(e)}")
    translations = {}

def get_language():
    """Get current language with fallback to English"""
    return session.get('language', 'en')

def translate(key, lang=None):
    """Get translated text for a key"""
    if lang is None:
        lang = get_language()
    try:
        # Split the key by dots to navigate nested dictionary
        keys = key.split('.')
        value = translations.get(lang, translations.get('en', {}))
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        # Fallback to English if translation not found
        try:
            value = translations['en']
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            # Return the key if no translation found
            return key

@app.context_processor
def utility_processor():
    """Make translations available in templates"""
    return dict(
        translate=translate,
        get_language=get_language
    )

class WeedClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WeedClassifier, self).__init__()
        
        # Use ResNet50 for better accuracy
        self.resnet = torchvision.models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-4]:
            param.requires_grad = False
            
        # Modify the final layers for better classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize the new layers
        for m in self.resnet.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.resnet(x)

def create_model_session():
    """Initialize PyTorch model"""
    try:
        model_path = os.path.join('models', 'weed_classifier.pth')
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None

        # Load model with PyTorch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create model instance
        num_classes = len(class_indices)
        logger.info(f"Creating model with {num_classes} classes")
        model = WeedClassifier(num_classes)
        
        # Load state dictionary
        logger.info(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        try:
            model.load_state_dict(state_dict)
            logger.info("Model state dict loaded successfully")
        except Exception as e:
            logger.error(f"Error loading state dict: {str(e)}")
            return None
        
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        logger.info("PyTorch model loaded successfully and set to eval mode")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Initialize model at startup
model = create_model_session()

def preprocess_image(image):
    """Preprocess image for PyTorch model input"""
    try:
        # Define transforms - same as validation transform used during training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        img_tensor = transform(image)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        return img_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def get_translated_data(data_dict, lang, fallback='en'):
    """Get translated data with fallback to English"""
    if not data_dict:
        return {}
    
    # Try requested language
    if lang in data_dict and data_dict[lang]:
        return data_dict[lang]
    
    # Fall back to English if requested language not available
    if fallback in data_dict and data_dict[fallback]:
        logger.warning(f"Translation not found for language {lang}, falling back to {fallback}")
        return data_dict[fallback]
    
    logger.error(f"No translation found for data in any language")
    return {}

def predict_weed(image):
    """Predict using PyTorch model"""
    try:
        global model
        if model is None:
            logger.info("Model not initialized, attempting to load...")
            model = create_model_session()
            if model is None:
                raise RuntimeError("Failed to initialize PyTorch model")

        # Add confidence threshold
        CONFIDENCE_THRESHOLD = 75.0  # 75% confidence threshold

        # Add morphological group mapping
        morphological_groups = {
            # Broadleaf weeds
            "Mollugo_verticillata": "Broadleaf",  # Carpetweeds
            "Cirsium_arvense": "Broadleaf",  # Canada thistle
            "Eclipta_prostrata": "Broadleaf",
            "Ipomoea_purpurea": "Broadleaf",  # Morningglory
            "Amaranthus_palmeri": "Broadleaf",  # PalmerAmaranth
            "Sida_spinosa": "Broadleaf",  # Prickly Sida
            "Portulaca_oleracea": "Broadleaf",  # Purslane
            "Ambrosia_artemisiifolia": "Broadleaf",  # Ragweed
            "Senna_obtusifolia": "Broadleaf",  # Sicklepod
            "Euphorbia_maculata": "Broadleaf",  # SpottedSpurge
            "Anoda_cristata": "Broadleaf",  # SpurredAnoda
            "Coronopus_didymus": "Broadleaf",  # Swinecress
            "Amaranthus_tuberculatus": "Broadleaf",  # Waterhemp
            "Amaranthus_spinosus": "Broadleaf",  # Amaranthus
            "Lianeblue": "Broadleaf",
            "Lianemargoze": "Broadleaf",
            "Lianepocpoc": "Broadleaf",
            "Osseille": "Broadleaf",
            "Pourpier": "Broadleaf",
            # Grass weeds
            "Digitaria_sanguinalis": "Grass",  # Crabgrass
            "Eleusine_indica": "Grass",  # Goosegrass
            "Brachypodium_sylvaticum": "Grass",
            # Sedge weeds
            "Nutsedge": "Sedge"  # Purple/Yellow Nutsedge
        }

        # Preprocess image
        logger.info("Preprocessing image...")
        input_data = preprocess_image(image)
        
        # Move to same device as model
        device = next(model.parameters()).device
        input_data = input_data.to(device)
        logger.info(f"Input shape: {input_data.shape}")
        
        # Run prediction with gradient disabled
        logger.info("Running prediction...")
        with torch.no_grad():
            outputs = model(input_data)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top prediction and confidence
        confidence, class_idx = torch.max(probabilities, dim=1)
        conf_value = float(confidence[0].item()) * 100
        
        # Check if confidence is below threshold
        if conf_value < CONFIDENCE_THRESHOLD:
            error_messages = {
                'en': 'The uploaded image does not appear to be a weed that this system can identify. Please ensure you are uploading a clear image of a weed.',
                'te': 'అప్లోడ్ చేసిన చిత్రం ఈ వ్యవస్థ గుర్తించగల కలుపు మొక్క కాదు. దయచేసి స్పష్టమైన కలుపు మొక్క చిత్రాన్ని అప్లోడ్ చేయండి.',
                'ta': 'பதிவேற்றிய படம் இந்த அமைப்பால் அடையாளம் காண முடியாத களையாக தெரிகிறது. தெளிவான களை படத்தை பதிவேற்றவும்.',
                'hi': 'अपलोड की गई छवि एक खरपतवार नहीं लगती जिसे यह सिस्टम पहचान सकता है। कृपया खरपतवार की एक स्पष्ट छवि अपलोड करें।'
            }
            lang = session.get('language', 'en')
            raise ValueError(error_messages.get(lang, error_messages['en']))
        
        # Get the predicted class name
        idx_to_class = {v: k for k, v in class_indices.items()}
        dataset_name = idx_to_class[int(class_idx[0].item())]
        logger.info(f"Predicted class: {dataset_name}")
        
        # Map dataset name to weed_info name
        name_mapping = {
            "Amaranthus": "Amaranthus_spinosus",
            "Brachypodium sylvaticum": "Brachypodium_sylvaticum",
            "Carpetweeds": "Mollugo_verticillata",
            "Cirsium arvense": "Cirsium_arvense",
            "Cotton": "Cotton",
            "Crabgrass": "Digitaria_sanguinalis",
            "Eclipta": "Eclipta_prostrata",
            "Goosegrass": "Eleusine_indica",
            "Morningglory": "Ipomoea_purpurea",
            "Nutsedge": "Nutsedge",
            "PalmerAmaranth": "Amaranthus_palmeri",
            "Prickly Sida": "Sida_spinosa",
            "Purslane": "Portulaca_oleracea",
            "Ragweed": "Ambrosia_artemisiifolia",
            "Sicklepod": "Senna_obtusifolia",
            "SpottedSpurge": "Euphorbia_maculata",
            "SpurredAnoda": "Anoda_cristata",
            "Swinecress": "Coronopus_didymus",
            "Waterhemp": "Amaranthus_tuberculatus",
            "liane blue": "Lianeblue",
            "liane margoze": "Lianemargoze",
            "liane pocpoc": "Lianepocpoc",
            "osseille": "Osseille",
            "pourpier": "Pourpier"
        }
        
        weed_name = name_mapping.get(dataset_name, dataset_name)
        logger.info(f"Mapped weed name: {weed_name}")
        
        # Get morphological group
        morphological_group = morphological_groups.get(weed_name, "Unknown")
        logger.info(f"Morphological group: {morphological_group}")
            
        # Get weed characteristics
        lang = get_language()
        logger.info(f"Current language: {lang}")
        
        # Get weed info in current language
        weed_data = weed_info.get(lang, {})
        if not weed_data:
            logger.warning(f"No data found for language {lang}, falling back to English")
            weed_data = weed_info.get('en', {})
        
        # Get specific weed info
        weed_info_specific = weed_data.get(weed_name, {})
        if not weed_info_specific:
            logger.error(f"No data found for weed: {weed_name}")
            logger.info(f"Available weeds: {list(weed_data.keys())}")
            return [{
                'class': dataset_name,
                'scientific_name': weed_name,
                'morphological_group': morphological_group,
                'confidence': round(conf_value, 2),
                'characteristics': [],
                'cultural_methods': [],
                'mechanical_methods': [],
                'chemical_methods': [],
                'biological_methods': [],
                'image_url': ''
            }]
        
        return [{
            'class': dataset_name,
            'scientific_name': weed_name,
            'morphological_group': morphological_group,
            'confidence': round(conf_value, 2),
            'characteristics': weed_info_specific.get('characteristics', []),
            'cultural_methods': weed_info_specific.get('control_methods', {}).get('cultural', []),
            'mechanical_methods': weed_info_specific.get('control_methods', {}).get('mechanical', []),
            'chemical_methods': weed_info_specific.get('control_methods', {}).get('chemical', []),
            'biological_methods': weed_info_specific.get('control_methods', {}).get('biological', []),
            'image_url': weed_info_specific.get('image_url', '')
        }]
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html', language=get_language())

@app.route('/search')
def search():
    lang = get_language()
    return render_template('search.html', language=lang)

@app.route('/result/<weed_name>')
def result(weed_name):
    lang = get_language()
    
    # Get translated weed data
    weed_data = get_translated_data(weed_info, lang)
    if not weed_data:
        return render_template('error.html',
                             message="No weed information available",
                             language=lang)
    
    # Get specific weed information
    weed_info_specific = weed_data.get(weed_name, {})
    if not weed_info_specific:
        return render_template('error.html',
                             message=f"Information not found for weed: {weed_name}",
                             language=lang)
    
    # Split control methods into categories
    cultural_methods = []
    mechanical_methods = []
    chemical_methods = []
    biological_methods = []
    
    current_category = None
    for method in weed_info_specific.get('control_methods', []):
        if method.startswith('**'):
            current_category = method.replace('**', '').strip().lower()
        elif current_category:
            if 'cultural' in current_category:
                cultural_methods.append(method)
            elif 'mechanical' in current_category:
                mechanical_methods.append(method)
            elif 'chemical' in current_category:
                chemical_methods.append(method)
            elif 'biological' in current_category:
                biological_methods.append(method)
    
    return render_template('result.html',
                         weed_name=weed_name,
                         image_url=weed_info_specific.get('image_url', ''),
                         confidence=weed_info_specific.get('confidence', 0),
                         characteristics=weed_info_specific.get('characteristics', []),
                         cultural_methods=cultural_methods,
                         mechanical_methods=mechanical_methods,
                         chemical_methods=chemical_methods,
                         biological_methods=biological_methods,
                         language=lang)

@app.route('/supplements')
def supplements():
    """Route for supplements page"""
    # Get language
    lang = get_language()
    
    # Get weed parameter from URL
    weed_name = request.args.get('weed', None)
    
    # Load chemical products data
    with open('chemical_products.json', 'r', encoding='utf-8') as f:
        chemical_products_data = json.load(f)
    
    # If weed name is provided, filter products for that weed
    chemical_products = []
    if weed_name:
        # Get products for the specified weed
        weed_data = chemical_products_data.get(weed_name, {})
        
        # Handle both array format and object with 'products' property
        if isinstance(weed_data, list):
            # Direct array of products
            weed_products = weed_data
        else:
            # Object with 'products' property
            weed_products = weed_data.get('products', [])
            
        for product in weed_products:
            # Extract purchase links from either format
            purchase_links = product.get('purchase_links', {})
            # If old format links exist and purchase_links doesn't have that property, add them
            if 'amazon_link' in product and not purchase_links.get('amazon'):
                if purchase_links is None:
                    purchase_links = {}
                purchase_links['amazon'] = product['amazon_link']
            if 'agribegri_link' in product and not purchase_links.get('agribegri'):
                if purchase_links is None:
                    purchase_links = {}
                purchase_links['agribegri'] = product['agribegri_link']
            if 'flipkart_link' in product and not purchase_links.get('flipkart'):
                if purchase_links is None:
                    purchase_links = {}
                purchase_links['flipkart'] = product['flipkart_link']
            
            chemical_products.append({
                'name': product.get('name', ''),
                'description': product.get('description', ''),
                'active_ingredient': product.get('active_ingredient', ''),
                'usage': product.get('usage', ''),
                'image_url': product.get('image_url', ''),
                'purchase_links': purchase_links,
                # Also keep the old link format for compatibility
                'amazon_link': product.get('amazon_link', ''),
                'agribegri_link': product.get('agribegri_link', ''),
                'flipkart_link': product.get('flipkart_link', '')
            })
    else:
        # Get all products
        for weed, data in chemical_products_data.items():
            # Handle both array format and object with 'products' property
            if isinstance(data, list):
                # Direct array of products
                products = data
            else:
                # Object with 'products' property
                products = data.get('products', [])
                
            if isinstance(products, list):
                for product in products:
                    # Extract purchase links from either format
                    purchase_links = product.get('purchase_links', {})
                    # If old format links exist and purchase_links doesn't have that property, add them
                    if 'amazon_link' in product and not purchase_links.get('amazon'):
                        if purchase_links is None:
                            purchase_links = {}
                        purchase_links['amazon'] = product['amazon_link']
                    if 'agribegri_link' in product and not purchase_links.get('agribegri'):
                        if purchase_links is None:
                            purchase_links = {}
                        purchase_links['agribegri'] = product['agribegri_link']
                    if 'flipkart_link' in product and not purchase_links.get('flipkart'):
                        if purchase_links is None:
                            purchase_links = {}
                        purchase_links['flipkart'] = product['flipkart_link']
                    
                    chemical_products.append({
                        'name': product.get('name', ''),
                        'description': product.get('description', ''),
                        'active_ingredient': product.get('active_ingredient', ''),
                        'usage': product.get('usage', ''),
                        'weed': weed,
                        'image_url': product.get('image_url', ''),
                        'purchase_links': purchase_links,
                        # Also keep the old link format for compatibility
                        'amazon_link': product.get('amazon_link', ''),
                        'agribegri_link': product.get('agribegri_link', ''),
                        'flipkart_link': product.get('flipkart_link', '')
                    })
    
    # Get application guidelines data
    application_guidelines = {}
    
    return render_template('supplements.html', 
                          chemical_products=chemical_products, 
                          application_guidelines=application_guidelines,
                          weed_name=weed_name)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Handle contact form submission
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')

        # Prepare the email
        email_subject = f"Contact Form Submission: {subject}"
        email_body = f"Name: {name}\nEmail: {email}\nSubject: {subject}\nMessage:\n{message}"

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = email_subject
        msg.attach(MIMEText(email_body, 'plain'))

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
            logger.info(f"Contact form email sent to {RECEIVER_EMAIL}")
            flash('Your message has been sent successfully!', 'success')
        except Exception as e:
            logger.error(f"Failed to send contact form email: {e}")
            flash('There was an error sending your message. Please try again later.', 'danger')

        return redirect(url_for('contact'))

    return render_template('contact.html',
                         team_members=team_info['team_members'],
                         address=team_info['address'],
                         phone=team_info['phone'],
                         email=team_info['email'],
                         language=session.get('language', 'en'))

@app.route('/change_language')
def change_language():
    lang = request.args.get('lang', 'en')
    if lang in {'en', 'te', 'ta', 'hi'}:
        session['language'] = lang
        logger.info(f"Language changed to: {lang}")
        # Get the current page URL to redirect back
        current_page = request.referrer or url_for('index')
        return redirect(current_page)
    else:
        logger.warning(f"Unsupported language requested: {lang}")
        lang = 'en'
        session['language'] = lang
    return redirect(url_for('index'))

@app.route('/change_crop')
def change_crop():
    crop = request.args.get('crop', 'cotton').lower()
    session['crop'] = crop
    if crop != 'cotton':
        warning_messages = {
            'en': 'This application only supports weed identification for cotton crops. The weed database and control methods are specifically designed for cotton cultivation.',
            'te': 'ఈ అప్లికేషన్ ప్రత్తి పంటలకు మాత్రమే కలుపు మొక్కల గుర్తింపును సహాయం చేస్తుంది. కలుపు మొక్కల డేటాబేస్ మరియు నియంత్రణ పద్ధతులు ప్రత్తి సాగుకు ప్రత్యేకంగా రూపొందించబడ్డాయి.',
            'ta': 'இந்த பயன்பாடு பருத்தி பயிர்களுக்கு மட்டுமே களை அடையாளத்தை ஆதரிக்கிறது. களை தரவுத்தளம் மற்றும் கட்டுப்பாட்டு முறைகள் பருத்தி சாகுபடிக்காக வடிவமைக்கப்பட்டுள்ளன.',
            'hi': 'यह एप्लिकेशन केवल कपास की फसलों के लिए खरपतवार पहचान का समर्थन करती है। खरपतवार डेटाबेस और नियंत्रण विधियाँ विशेष रूप से कपास की खेती के लिए डिज़ाइन की गई हैं।'
        }
        lang = session.get('language', 'en')
        flash(warning_messages.get(lang, warning_messages['en']))
    return redirect(request.referrer or url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('result.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('result.html', error='No file selected')
        
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get predictions
        predictions = predict_weed(image)
        
        # Get the prediction details
        prediction = predictions[0]  # We know we'll always have a prediction now
        scientific_name = prediction['scientific_name']
        common_name = prediction['class']
        confidence = prediction['confidence']
        morphological_group = prediction['morphological_group']
        
        # Format the weed name for display
        display_name = f"{common_name} ({scientific_name})"
        if common_name == scientific_name:
            display_name = scientific_name
        
        return render_template('result.html',
                             weed_name=display_name,
                             confidence=confidence,
                             morphological_group=morphological_group,
                             characteristics=prediction['characteristics'],
                             cultural_methods=prediction['cultural_methods'],
                             mechanical_methods=prediction['mechanical_methods'],
                             chemical_methods=prediction['chemical_methods'],
                             biological_methods=prediction['biological_methods'],
                             image_url=prediction['image_url'],
                             language=get_language())
                             
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return render_template('result.html',
                             error=str(e),
                             weed_name='',
                             confidence=0,
                             morphological_group='',
                             characteristics=[],
                             cultural_methods=[],
                             mechanical_methods=[],
                             chemical_methods=[],
                             biological_methods=[],
                             language=get_language())

@app.route('/identify', methods=['POST'])
def identify():
    try:
        # Get weed information in the current language
        lang = get_language()
        weed_data = get_translated_data(weed_info, lang)
        
        # If no translation found in current language, fall back to English
        if not weed_data or scientific_name not in weed_data:
            logger.warning(f"Translation not found for {scientific_name} in {lang}, falling back to English")
            weed_data = weed_info.get('en', {})
        
        weed_info_specific = weed_data.get(scientific_name, {})
        
        # Get characteristics and control methods from the translated data
        characteristics = weed_info_specific.get('characteristics', [])
        control_methods = weed_info_specific.get('control_methods', {})
        
        # Extract control methods by category
        cultural_methods = control_methods.get('cultural', [])
        mechanical_methods = control_methods.get('mechanical', [])
        chemical_methods = control_methods.get('chemical', [])
        biological_methods = control_methods.get('biological', [])
        
        # Format the weed name for display
        display_name = f"{common_name} ({scientific_name})"
        if common_name == scientific_name:
            display_name = scientific_name
            
        return render_template('result.html',
                             weed_name=display_name,
                             confidence=confidence,
                             characteristics=characteristics,
                             cultural_methods=cultural_methods,
                             mechanical_methods=mechanical_methods,
                             chemical_methods=chemical_methods,
                             biological_methods=biological_methods,
                             image_url=weed_info_specific.get('image_url', ''),
                             language=lang)
    except Exception as e:
        logger.error(f"Error in identify route: {str(e)}")
        return render_template('error.html', 
                             error="An error occurred while processing your request. Please try again.",
                             language=get_language())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
