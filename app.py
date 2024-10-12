import os
from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import io
import logging
import traceback
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for
app = Flask(__name__,static_folder='static')

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load your model
MODEL_PATH = 'C:/Users/JR/Desktop/soil using flask v2/soil_classification_modelv1.h5'
try:
    model = load_model(MODEL_PATH, compile=False)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    logging.error(traceback.format_exc())

# Define a confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# In-memory storage for predicted soil types
predicted_soil_types = {}

# Helper function to preprocess the image for the model
def preprocess_image(img):
    try:
        img = img.resize((224, 224))  # Adjust size as per your model's requirement
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize the image if required by the model
        return img
    except Exception as e:
        logging.error(f"Error during image preprocessing: {str(e)}")
        logging.error(traceback.format_exc())
        raise e

# Helper function to calculate soil quality based on NPK values
def calculate_soil_quality(nitrogen, phosphorus, potassium):
    # Define thresholds (these can be adjusted based on domain knowledge)
    N = nitrogen * .33
    P = phosphorus * .33
    K = potassium * .33
    
    npk = N+P+K
    
    if npk < .33:
        return 'Poor'
    elif .66 < npk >=.33:
        return 'Moderate'
    else:
        return 'High' 
    
  #  if nitrogen < 10 and phosphorus < 5 and potassium < 10:
   #     return 'Poor'
   # elif 10 <= nitrogen < 20 and 5 <= phosphorus < 15 and 10 <= potassium < 20:
   #     return 'Moderate'
   # else:
   #     return 'High'
  
# Route to serve the view images page
@app.route('/view-images')
def view_images():
    upload_dir = os.path.join('static', 'uploads')
    images = []
    
    if os.path.exists(upload_dir):
        for filename in os.listdir(upload_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
                # Retrieve the soil type based on the filename, using an empty string as default if not found
                soil_type = predicted_soil_types.get(filename, "Unknown Soil Type")
                images.append({
                    'id': len(images) + 1,
                    'image_url': url_for('static', filename=f'uploads/{filename}'),
                    'soil_type': soil_type
                })
    
    return render_template('view_images.html', images=images)

   
# Route to serve the real-time soil assessment page
@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

# Route to serve the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to serve the upload page
@app.route('/upload')
def upload():
    return render_template('upload.html')

# Route to handle file upload and make predictions
@app.route('/upload-image', methods=['POST'])
def upload_image():
    image_file = request.files.get('image')

    if not image_file:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Ensure the 'static/uploads' directory exists
        upload_dir = os.path.join('static', 'uploads')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        # Save the uploaded image temporarily
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(upload_dir, filename)
        image_file.save(image_path)

        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_preprocessed = preprocess_image(img)

        # Make predictions using the model
        predictions = model.predict(img_preprocessed)

        # Assuming the model outputs [soil_type_probs, npk_values]
        soil_type_probs = predictions[0][0].tolist()
        npk_values = predictions[1][0].tolist()

        # Get the index of the highest probability for soil type
        soil_classes = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']
        predicted_soil_type_index = np.argmax(soil_type_probs)
        confidence_level = soil_type_probs[predicted_soil_type_index]

        if confidence_level < CONFIDENCE_THRESHOLD:
            return jsonify({'error': 'Low confidence in the prediction'}), 400

        predicted_soil_type = soil_classes[predicted_soil_type_index]
        
        # Store the predicted soil type
        predicted_soil_types[filename] = predicted_soil_type

        # Calculate soil quality based on NPK values
        soil_quality = calculate_soil_quality(npk_values[0], npk_values[1], npk_values[2])

        # Rule-based decision support system with detailed information
        decision_support = {
            'Black Soil': {
                'best_crops': 'Rice, Corn, Sugarcane, Lettuce, Kale, Spinach, Carrots, Potatoes, Sweet potatoes, Mangoes, Bananas, Citrus fruits, Coffee, Cacao, Tomatoes, Peppers',
                
                'season': '[WET SEASON] : Rice, Corn, Sugarcane, Bananas, Citrus fruits, Coffee, Cacao \n [DRY SEASON]: Lettuce, Kale, Spinach, Carrots, Potatoes, Sweet potatoes, Mangoes, Tomatoes, Peppers',
                'fertilizer': 'High Nitrogen',
                'optimal_ph': '6.5 - 7.5',
                'common_issues':
    "Common issues include: "
    "waterlogging due to poor drainage (solution: improve drainage and raised beds), "
    "compaction restricting root growth (solution: tilling and adding organic matter), "
    "cracking from dry spells (solution: mulching and irrigation), "
    "nitrogen deficiency from leaching after rains (solution: apply nitrogen fertilizers or legumes), "
    "phosphorus fixation making phosphorus unavailable (solution: use phosphorus fertilizers and conduct soil tests), "
    "pH imbalance causing nutrient lockout (solution: regular pH testing and amendments), "
    "erosion leading to topsoil loss (solution: implement erosion control measures), "
    "salinity from excess salts (solution: manage irrigation and use gypsum), "
    "and slow warming delaying germination (solution: use plastic mulches or cover crops)."
,
            'management_practices': [
    'Regular soil testing: Monitor nutrient levels and pH.',
    'Crop rotation: Enhance soil fertility and disrupt pest cycles.',
    'Cover cropping: Improve soil structure and organic matter.',
    'Conservation tillage: Reduce erosion and maintain moisture.',
    'Organic amendments: Use compost or manure to boost nutrients.',
    'Mulching: Retain moisture and suppress weeds.',
    'Proper irrigation: Manage water levels and avoid waterlogging.',
    'Erosion control: Implement terracing or contour farming.',
    'Integrated pest management: Reduce chemical inputs and promote beneficial organisms.'
]
,
                'additional_info': 'Ideal for nutrient-demanding crops.'
            },
            'Cinder Soil': {
                'best_crops': 'Sweet Potatoes (Camote), Moringa (Malunggay), Cactus (Opuntia), Cassava (Kamoteng Kahoy), Sorghum, Legumes (e.g., mung beans, pigeon peas), Pineapple, Taro (Gabi)',
            'season': '[WET SEASON]: Moringa (Malunggay), Pigeon peas \n [DRY SEASON]: Sweet Potatoes (Camote), Cactus (Opuntia), Cassava (Kamoteng Kahoy), Sorghum, Mung beans, Pineapple, Taro (Gabi)',
                'fertilizer': 'Balanced NPK',
                'optimal_ph': '5.5 - 6.5',
                'common_issues': 'Cinder soil in the Philippines presents several common issues that affect agricultural productivity, including poor nutrient content, low water retention, and susceptibility to erosion. Its coarse texture leads to rapid drainage, making it challenging to retain moisture during dry spells. Additionally, cinder soil can be acidic, limiting nutrient availability to plants, and it often encourages weed growth, which competes with crops for essential resources. The reduced organic matter can also hinder microbial activity, impacting nutrient cycling and overall soil health. Furthermore, many traditional crops may struggle to thrive in cinder soil conditions, restricting farmers\' options for cultivation. Lastly, the soil\'s tendency to heat up quickly during the day and cool down rapidly at night can stress sensitive crops.',
                'management_practices': 'to effectively manage the challenges associated with cinder soil in the Philippines, farmers can implement several practices. Firstly, incorporating soil amendments, such as organic matter like compost, can significantly enhance fertility and improve water retention. Utilizing mulching techniques can help retain soil moisture, suppress weed growth, and gradually enrich the soil quality over time. Implementing crop rotation can further promote soil health and minimize pest and disease buildup. Additionally, planting cover crops can prevent erosion, enhance soil structure, and contribute organic matter back into the soil. Lastly, adopting conservation practices like conservation tillage and contour farming can reduce soil erosion and improve water retention, ultimately fostering a more sustainable agricultural environment.',
                'additional_info': 'Suitable for cooler climates and drought resistance.'
            },
            'Laterite Soil': {
                'best_crops' : 'Coconut, Pineapple, Banana, Papaya, Legumes (e.g., mung beans, peanuts), Taro (Gabi), Sugarcane, Coffee',
                'season': '[WET SEASON]: Taro (Gabi), Coconut, Papaya, Banana, Coffee \n [DRY SEASON]: Pineapple, Sugarcane, Legumes (e.g., mung beans, peanuts)',
                'fertilizer': 'High Phosphorus',
                'optimal_ph': '4.5 - 6.0',
                'common_issues': 'Latterite soil in the Philippines presents several common issues that can impact agricultural productivity, including low nutrient availability, acidity, and poor water retention. The soil\'s high drainage capacity can lead to rapid moisture loss, making it challenging to maintain adequate water levels for crops, particularly during dry spells. Additionally, its natural acidity can limit the availability of essential nutrients, hindering plant growth and development. The low organic matter content also reduces microbial activity, which is crucial for nutrient cycling and overall soil health. Furthermore, the loose structure of laterite soil makes it prone to erosion, especially on slopes, leading to further nutrient loss and soil degradation over time. These challenges require careful management practices to enhance soil fertility and support sustainable agricultural practices.',
                'management_practices': 'To effectively manage the challenges associated with laterite soil in the Philippines, farmers can adopt several practices that enhance soil fertility and crop productivity. First, incorporating organic matter, such as compost or green manure, can improve soil structure, nutrient availability, and moisture retention. Liming the soil can help address acidity issues, enhancing nutrient availability for crops. Utilizing mulching techniques can help retain soil moisture, suppress weeds, and gradually increase organic matter in the soil. Crop rotation is another important practice, as it can improve soil health and reduce pest and disease pressures by alternating deep-rooted and shallow-rooted crops. Additionally, planting cover crops can prevent soil erosion, enhance soil structure, and add organic matter back into the soil when tilled under. Implementing conservation practices, such as contour farming and reduced tillage, can also mitigate soil erosion and improve water retention. Together, these management strategies can help optimize the productivity of laterite soil and promote sustainable agriculture.',
                'additional_info': 'Well-suited for tropical regions.'
            },
            'Yellow Soil': {
                'best_crops': 'Rice, Corn, Sweet Potatoes (Camote), Legumes (e.g., mung beans, peanuts), Sugarcane, Taro (Gabi), Bananas, Coconut, Coffee',
                'season': '[WET SEASON]: Rice, Corn, Legumes (e.g., mung beans, peanuts), Taro (Gabi), Coconut, Banana, Coffee \n [DRY SEASON]: Sweet Potatoes (Camote), Sugarcane',
                'fertilizer': 'Balanced NPK',
                'optimal_ph': '6.0 - 7.0',
                'common_issues': 'Yellow soil, often referred to as lateritic soil, is common in many parts of the Philippines and presents several challenges for agriculture. One major issue is nutrient deficiency, as yellow soils often have low fertility due to leaching of essential nutrients like nitrogen, phosphorus, and potassium, making it difficult for crops to thrive. Additionally, these soils tend to be acidic, hindering nutrient availability, which may necessitate liming to neutralize acidity. Poor drainage can also occur in yellow soil, leading to waterlogging during heavy rains, negatively impacting root development and crop health. Erosion is another concern, as the loose structure of yellow soils makes them susceptible to erosion, especially on slopes, resulting in topsoil loss and further nutrient depletion. Compaction can occur due to heavy machinery or livestock, reducing aeration and water infiltration. Furthermore, the coarse texture of yellow soils can lead to poor moisture retention, requiring more frequent irrigation. Crops grown in nutrient-poor soils may also be more vulnerable to pests and diseases, affecting overall yield. To address these challenges, farmers can employ strategies such as incorporating organic matter, practicing crop rotation, using cover crops, applying lime to reduce acidity, and implementing conservation practices like contour farming or terracing to manage erosion and improve water retention.',
                'management_practices': 'To effectively manage yellow soil in the Philippines, farmers can implement several practices aimed at enhancing soil fertility and structure. Incorporating organic matter, such as compost or green manure, can significantly improve nutrient availability and soil texture. Practicing crop rotation helps maintain soil health by preventing nutrient depletion and reducing pest and disease buildup. Additionally, planting cover crops during the off-season can help prevent erosion, enhance organic matter content, and improve soil moisture retention. Applying lime to the soil can neutralize acidity and enhance nutrient availability, promoting better crop growth. Conservation practices like contour farming and terracing can be effective in managing erosion and improving water retention in sloped areas. Lastly, reducing soil compaction through practices like minimal tillage and careful management of heavy machinery can enhance aeration and water infiltration, contributing to healthier crops and improved productivity.',
                'additional_info': 'Good for a variety of crops with moderate irrigation.'
            },
            'Peat Soil': {
                'best_crops': 'Rice, Coconut, Vegetables (Lettuce, Cabbage, Tomatoes), Fruits (Bananas, Mangoes, Pineapples), Taro (Gabi), Watercress, Cassava (Kamoteng Kahoy), Lotus (Nelumbo nucifera)',
                'season': '[WET SEASON]: Rice, Taro (Gabi), Cabbage, Watercress \n [DRY SEASON]: Coconut, Vegetables (Lettuce, Tomatoes), Fruits (Bananas, Mangoes, Pineapples), Cassava (Kamoteng Kahoy), Lotus (Nelumbo nucifera)',
                'fertilizer': 'High Potassium',
                'optimal_ph': '5.0 - 6.5',
                'common_issues': 'Peat soil in the Philippines faces several common issues, including low nutrient content, which limits plant growth and necessitates fertilizer use; high acidity, often hindering nutrient availability; and waterlogging, which can suffocate plant roots. Additionally, drained peat soils may decompose rapidly, leading to subsidence and affecting agricultural viability and infrastructure stability. Environmental concerns arise from the release of significant amounts of carbon dioxide (CO2) during drainage, contributing to climate change and impacting biodiversity. Furthermore, poor drainage can lead to flooding during heavy rainfall, while erosion can occur when peatlands are disturbed, resulting in valuable topsoil loss. Lastly, the suitability of crops is limited, necessitating careful selection to ensure successful cultivation.',
                'management_practices': 'Management practices for addressing issues related to peat soil in the Philippines include amending the soil with lime to neutralize acidity and incorporating organic matter to enhance nutrient content and improve soil structure. Effective water management techniques, such as implementing proper drainage systems, can help control water levels and prevent waterlogging. Sustainable agricultural practices like using cover crops, agroforestry, and crop rotation can further enhance soil health and productivity. Additionally, adopting controlled drainage techniques can reduce carbon dioxide emissions and support the conservation of peatlands, ensuring that agricultural practices remain viable while protecting the environment.',
                'additional_info': 'Suitable for moisture-loving plants.'
            }
        }

        decision_support_info = decision_support.get(predicted_soil_type, {})

        # Format the response with the prediction results and decision support
        result = {
            'top_prediction': predicted_soil_type,
            'confidence_level': confidence_level,
            'soil_quality': soil_quality,  # Include soil quality in the response
            'npk_values': {
                'nitrogen': npk_values[0],
                'phosphorus': npk_values[1],
                'potassium': npk_values[2]
            },
            'decision_support': decision_support_info
        }

        return jsonify(result), 200

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Route to handle real-time image capture and make predictions
@app.route('/capture-image', methods=['POST'])
def capture_image():
    image_file = request.files.get('image')
    
    if not image_file:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Load and preprocess the image
        image = Image.open(io.BytesIO(image_file.read()))
        img_preprocessed = preprocess_image(image)

        # Make predictions using the model
        predictions = model.predict(img_preprocessed)
        soil_type_probs = predictions[0][0].tolist()
        npk_values = predictions[1][0].tolist()

        # Get the index of the highest probability for soil type
        soil_classes = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']
        predicted_soil_type_index = np.argmax(soil_type_probs)
        confidence_level = soil_type_probs[predicted_soil_type_index]

        if confidence_level < CONFIDENCE_THRESHOLD:
            return jsonify({'error': 'Low confidence in the prediction'}), 400

        predicted_soil_type = soil_classes[predicted_soil_type_index]

        # Calculate soil quality based on NPK values
        soil_quality = calculate_soil_quality(npk_values[0], npk_values[1], npk_values[2])

        # Rule-based decision support system with detailed information
        decision_support = {
            'Black Soil': {
                'best_crops': 'Rice, Corn, Sugarcane, Lettuce, Kale, Spinach, Carrots, Potatoes, Sweet potatoes, Mangoes, Bananas, Citrus fruits, Coffee, Cacao, Tomatoes, Peppers',
                
                'season': '[WET SEASON] : Rice, Corn, Sugarcane, Bananas, Citrus fruits, Coffee, Cacao \n [DRY SEASON]: Lettuce, Kale, Spinach, Carrots, Potatoes, Sweet potatoes, Mangoes, Tomatoes, Peppers',
                'fertilizer': 'High Nitrogen',
                'optimal_ph': '6.5 - 7.5',
                'common_issues':
    "Common issues include: "
    "waterlogging due to poor drainage (solution: improve drainage and raised beds), "
    "compaction restricting root growth (solution: tilling and adding organic matter), "
    "cracking from dry spells (solution: mulching and irrigation), "
    "nitrogen deficiency from leaching after rains (solution: apply nitrogen fertilizers or legumes), "
    "phosphorus fixation making phosphorus unavailable (solution: use phosphorus fertilizers and conduct soil tests), "
    "pH imbalance causing nutrient lockout (solution: regular pH testing and amendments), "
    "erosion leading to topsoil loss (solution: implement erosion control measures), "
    "salinity from excess salts (solution: manage irrigation and use gypsum), "
    "and slow warming delaying germination (solution: use plastic mulches or cover crops)."
,
            'management_practices': [
    'Regular soil testing: Monitor nutrient levels and pH.',
    'Crop rotation: Enhance soil fertility and disrupt pest cycles.',
    'Cover cropping: Improve soil structure and organic matter.',
    'Conservation tillage: Reduce erosion and maintain moisture.',
    'Organic amendments: Use compost or manure to boost nutrients.',
    'Mulching: Retain moisture and suppress weeds.',
    'Proper irrigation: Manage water levels and avoid waterlogging.',
    'Erosion control: Implement terracing or contour farming.',
    'Integrated pest management: Reduce chemical inputs and promote beneficial organisms.'
]
,
                'additional_info': 'Ideal for nutrient-demanding crops.'
            },
            'Cinder Soil': {
                'best_crops': 'Sweet Potatoes (Camote), Moringa (Malunggay), Cactus (Opuntia), Cassava (Kamoteng Kahoy), Sorghum, Legumes (e.g., mung beans, pigeon peas), Pineapple, Taro (Gabi)',
            'season': '[WET SEASON]: Moringa (Malunggay), Pigeon peas \n [DRY SEASON]: Sweet Potatoes (Camote), Cactus (Opuntia), Cassava (Kamoteng Kahoy), Sorghum, Mung beans, Pineapple, Taro (Gabi)',
                'fertilizer': 'Balanced NPK',
                'optimal_ph': '5.5 - 6.5',
                'common_issues': 'Cinder soil in the Philippines presents several common issues that affect agricultural productivity, including poor nutrient content, low water retention, and susceptibility to erosion. Its coarse texture leads to rapid drainage, making it challenging to retain moisture during dry spells. Additionally, cinder soil can be acidic, limiting nutrient availability to plants, and it often encourages weed growth, which competes with crops for essential resources. The reduced organic matter can also hinder microbial activity, impacting nutrient cycling and overall soil health. Furthermore, many traditional crops may struggle to thrive in cinder soil conditions, restricting farmers\' options for cultivation. Lastly, the soil\'s tendency to heat up quickly during the day and cool down rapidly at night can stress sensitive crops.',
                'management_practices': 'to effectively manage the challenges associated with cinder soil in the Philippines, farmers can implement several practices. Firstly, incorporating soil amendments, such as organic matter like compost, can significantly enhance fertility and improve water retention. Utilizing mulching techniques can help retain soil moisture, suppress weed growth, and gradually enrich the soil quality over time. Implementing crop rotation can further promote soil health and minimize pest and disease buildup. Additionally, planting cover crops can prevent erosion, enhance soil structure, and contribute organic matter back into the soil. Lastly, adopting conservation practices like conservation tillage and contour farming can reduce soil erosion and improve water retention, ultimately fostering a more sustainable agricultural environment.',
                'additional_info': 'Suitable for cooler climates and drought resistance.'
            },
            'Laterite Soil': {
                'best_crops' : 'Coconut, Pineapple, Banana, Papaya, Legumes (e.g., mung beans, peanuts), Taro (Gabi), Sugarcane, Coffee',
                'season': '[WET SEASON]: Taro (Gabi), Coconut, Papaya, Banana, Coffee \n [DRY SEASON]: Pineapple, Sugarcane, Legumes (e.g., mung beans, peanuts)',
                'fertilizer': 'High Phosphorus',
                'optimal_ph': '4.5 - 6.0',
                'common_issues': 'Latterite soil in the Philippines presents several common issues that can impact agricultural productivity, including low nutrient availability, acidity, and poor water retention. The soil\'s high drainage capacity can lead to rapid moisture loss, making it challenging to maintain adequate water levels for crops, particularly during dry spells. Additionally, its natural acidity can limit the availability of essential nutrients, hindering plant growth and development. The low organic matter content also reduces microbial activity, which is crucial for nutrient cycling and overall soil health. Furthermore, the loose structure of laterite soil makes it prone to erosion, especially on slopes, leading to further nutrient loss and soil degradation over time. These challenges require careful management practices to enhance soil fertility and support sustainable agricultural practices.',
                'management_practices': 'To effectively manage the challenges associated with laterite soil in the Philippines, farmers can adopt several practices that enhance soil fertility and crop productivity. First, incorporating organic matter, such as compost or green manure, can improve soil structure, nutrient availability, and moisture retention. Liming the soil can help address acidity issues, enhancing nutrient availability for crops. Utilizing mulching techniques can help retain soil moisture, suppress weeds, and gradually increase organic matter in the soil. Crop rotation is another important practice, as it can improve soil health and reduce pest and disease pressures by alternating deep-rooted and shallow-rooted crops. Additionally, planting cover crops can prevent soil erosion, enhance soil structure, and add organic matter back into the soil when tilled under. Implementing conservation practices, such as contour farming and reduced tillage, can also mitigate soil erosion and improve water retention. Together, these management strategies can help optimize the productivity of laterite soil and promote sustainable agriculture.',
                'additional_info': 'Well-suited for tropical regions.'
            },
            'Yellow Soil': {
                'best_crops': 'Rice, Corn, Sweet Potatoes (Camote), Legumes (e.g., mung beans, peanuts), Sugarcane, Taro (Gabi), Bananas, Coconut, Coffee',
                'season': '[WET SEASON]: Rice, Corn, Legumes (e.g., mung beans, peanuts), Taro (Gabi), Coconut, Banana, Coffee \n [DRY SEASON]: Sweet Potatoes (Camote), Sugarcane',
                'fertilizer': 'Balanced NPK',
                'optimal_ph': '6.0 - 7.0',
                'common_issues': 'Yellow soil, often referred to as lateritic soil, is common in many parts of the Philippines and presents several challenges for agriculture. One major issue is nutrient deficiency, as yellow soils often have low fertility due to leaching of essential nutrients like nitrogen, phosphorus, and potassium, making it difficult for crops to thrive. Additionally, these soils tend to be acidic, hindering nutrient availability, which may necessitate liming to neutralize acidity. Poor drainage can also occur in yellow soil, leading to waterlogging during heavy rains, negatively impacting root development and crop health. Erosion is another concern, as the loose structure of yellow soils makes them susceptible to erosion, especially on slopes, resulting in topsoil loss and further nutrient depletion. Compaction can occur due to heavy machinery or livestock, reducing aeration and water infiltration. Furthermore, the coarse texture of yellow soils can lead to poor moisture retention, requiring more frequent irrigation. Crops grown in nutrient-poor soils may also be more vulnerable to pests and diseases, affecting overall yield. To address these challenges, farmers can employ strategies such as incorporating organic matter, practicing crop rotation, using cover crops, applying lime to reduce acidity, and implementing conservation practices like contour farming or terracing to manage erosion and improve water retention.',
                'management_practices': 'To effectively manage yellow soil in the Philippines, farmers can implement several practices aimed at enhancing soil fertility and structure. Incorporating organic matter, such as compost or green manure, can significantly improve nutrient availability and soil texture. Practicing crop rotation helps maintain soil health by preventing nutrient depletion and reducing pest and disease buildup. Additionally, planting cover crops during the off-season can help prevent erosion, enhance organic matter content, and improve soil moisture retention. Applying lime to the soil can neutralize acidity and enhance nutrient availability, promoting better crop growth. Conservation practices like contour farming and terracing can be effective in managing erosion and improving water retention in sloped areas. Lastly, reducing soil compaction through practices like minimal tillage and careful management of heavy machinery can enhance aeration and water infiltration, contributing to healthier crops and improved productivity.',
                'additional_info': 'Good for a variety of crops with moderate irrigation.'
            },
            'Peat Soil': {
                'best_crops': 'Rice, Coconut, Vegetables (Lettuce, Cabbage, Tomatoes), Fruits (Bananas, Mangoes, Pineapples), Taro (Gabi), Watercress, Cassava (Kamoteng Kahoy), Lotus (Nelumbo nucifera)',
                'season': '[WET SEASON]: Rice, Taro (Gabi), Cabbage, Watercress \n [DRY SEASON]: Coconut, Vegetables (Lettuce, Tomatoes), Fruits (Bananas, Mangoes, Pineapples), Cassava (Kamoteng Kahoy), Lotus (Nelumbo nucifera)',
                'fertilizer': 'High Potassium',
                'optimal_ph': '5.0 - 6.5',
                'common_issues': 'Peat soil in the Philippines faces several common issues, including low nutrient content, which limits plant growth and necessitates fertilizer use; high acidity, often hindering nutrient availability; and waterlogging, which can suffocate plant roots. Additionally, drained peat soils may decompose rapidly, leading to subsidence and affecting agricultural viability and infrastructure stability. Environmental concerns arise from the release of significant amounts of carbon dioxide (CO2) during drainage, contributing to climate change and impacting biodiversity. Furthermore, poor drainage can lead to flooding during heavy rainfall, while erosion can occur when peatlands are disturbed, resulting in valuable topsoil loss. Lastly, the suitability of crops is limited, necessitating careful selection to ensure successful cultivation.',
                'management_practices': 'Management practices for addressing issues related to peat soil in the Philippines include amending the soil with lime to neutralize acidity and incorporating organic matter to enhance nutrient content and improve soil structure. Effective water management techniques, such as implementing proper drainage systems, can help control water levels and prevent waterlogging. Sustainable agricultural practices like using cover crops, agroforestry, and crop rotation can further enhance soil health and productivity. Additionally, adopting controlled drainage techniques can reduce carbon dioxide emissions and support the conservation of peatlands, ensuring that agricultural practices remain viable while protecting the environment.',
                'additional_info': 'Suitable for moisture-loving plants.'
            }
        }

        decision_support_info = decision_support.get(predicted_soil_type, {})

        # Format the response with the prediction results and decision support
        result = {
            'top_prediction': predicted_soil_type,
            'confidence_level': confidence_level,
            'soil_quality': soil_quality,  # Include soil quality in the response
            'npk_values': {
                'nitrogen': npk_values[0],
                'phosphorus': npk_values[1],
                'potassium': npk_values[2]
            }
            
        }

        decision_support_info = decision_support.get(predicted_soil_type, {})

        # Format the response with the prediction results and decision support
        result = {
            'top_prediction': predicted_soil_type,
            'confidence_level': confidence_level,
            'soil_quality': soil_quality,  # Include soil quality in the response
            'npk_values': {
                'nitrogen': npk_values[0],
                'phosphorus': npk_values[1],
                'potassium': npk_values[2]
            },
            'decision_support': decision_support_info
        }

        return jsonify(result), 200

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
@app.route('/save-captured-image', methods=['POST'])
def save_captured_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file part"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Secure the filename and save the image
    filename = secure_filename(image_file.filename)
    save_path = os.path.join(upload_dir, filename)
    image_file.save(save_path)

    return jsonify({"message": "Image saved successfully!", "filename": filename}), 200
if __name__ == '__main__':
    #app.run(host='222.2.2.32', port=5000, debug=True)
    app.run(debug=False)
