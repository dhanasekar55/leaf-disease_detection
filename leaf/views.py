# views.py

from django.shortcuts import render
from .models import PlantDisease, PageView
from django.conf import settings
import os
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the disease prediction model (update the path accordingly)
model = load_model("C:/Users/Intel/Desktop/leaf/disease/Leaf Deases(96,88).h5")

# Define the list of disease classes (You should have the full list of diseases here)
DISEASE_CLASSES = [
    'Apple scab', 'Black rot', 'Cedar apple rust', 'Healthy', 
    'Powdery Mildew', 'Late Blight', 'Early Blight', 'Cedar Apple Rust', 
    'Anthracnose', 'Botrytis', 'Leaf Spot Diseases', 'Rusts', 'Bacterial Spot', 
    'Fire Blight', 'Bacterial Wilt', 'Crown Gall', 'Mosaic Virus', 
    'Tomato Spotted Wilt Virus', 'Cucumber Mosaic Virus', 'Clubroot', 
    'Fusarium Wilt', 'Verticillium Wilt', 'Blossom End Rot', 'Downy Mildew', 
    'Root Rot', 'Yellow Leaf Curl Virus', 'Alternaria Leaf Spot'
]

# Disease information (you can extend this as needed)
DISEASE_DETAILS = {
    'Apple scab': {
        'definition': 'A fungal disease that causes dark lesions on leaves, stems, and fruit.',
        'symptoms': 'Dark lesions on leaves and fruit, deformed fruit.',
        'remedy': 'Apply fungicides, remove infected plant material.',
        'link':'https://extension.umn.edu/plant-diseases/apple-scab'
    },
    'Black rot': {
        'definition': 'A bacterial disease caused by Xanthomonas campestris, it affects fruit, leaves, and stems.',
        'symptoms': 'Black lesions on leaves, fruit rotting, wilting of branches.',
        'remedy': 'Remove infected plant parts, use copper-based bactericides.',
        'link':'https://extension.umn.edu/disease-management/organic-management-black-rot#:~:text=Black%20rot%20is%20caused%20by,rot%20can%20persist%20in%20residues'
    },
    'Cedar apple rust': {
        'definition': 'A fungal disease caused by Gymnosporangium juniper-virginianae, affecting apple and cedar trees.',
        'symptoms': 'Orange spore-producing lesions on leaves, yellow spots on apple fruit.',
        'remedy': 'Prune cedar trees near apple orchards, use fungicides like myclobutanil.',
        'link':'https://extension.umn.edu/plant-diseases/cedar-apple-rust',
    },
    'Healthy': {
        'definition': 'No disease present.',
        'symptoms': 'No visible symptoms of disease.',
        'remedy': 'Maintain proper care, watering, and environment.'
    },
    'Powdery Mildew': {
        'definition': 'A fungal disease caused by several species like Erysiphe cichoracearum, affecting a wide range of plants.',
        'symptoms': 'White, powdery fungal growth on the leaves and stems.',
        'remedy': 'Use fungicides like sulfur or neem oil, remove infected parts, and improve air circulation.',
        'link':'https://extension.umn.edu/plant-diseases/powdery-mildew-flower-garden'
    },
    'Late Blight': {
        'definition': 'A fungal disease caused by Phytophthora infestans, affecting tomatoes and potatoes.',
        'symptoms': 'Dark, water-soaked lesions on leaves and stems, rotting fruit.',
        'remedy': 'Apply fungicides, remove infected plants, and avoid overhead watering.',
        'link':'https://extension.umn.edu/disease-management/late-blight'
    },
    'Early Blight': {
        'definition': 'A fungal disease caused by Alternaria solani, it affects tomatoes, potatoes, and other solanaceous plants.',
        'symptoms': 'Concentric dark lesions with yellow halos on lower leaves.',
        'remedy': 'Use copper fungicides, rotate crops, and remove infected plant material.',
        'link':'https://extension.umn.edu/disease-management/early-blight-tomato-and-potato'
    },
    'Anthracnose': {
        'definition': 'A fungal disease caused by various species such as Colletotrichum.',
        'symptoms': 'Dark sunken lesions on leaves, stems, and fruits.',
        'remedy': 'Remove infected leaves and fruit, apply fungicides like chlorothalonil.',
        'link':'https://extension.umn.edu/plant-diseases/anthracnose-trees-and-shrubs'
    },
    'Botrytis': {
        'definition': 'A fungal disease caused by Botrytis cinerea, affecting a wide range of plants.',
        'symptoms': 'Grayish mold growth on decaying plant tissues.',
        'remedy': 'Reduce humidity, improve air circulation, and use fungicides.',
        'link':'https://hort.extension.wisc.edu/articles/gray-mold-botrytis-blight/'
    },
    'Leaf Spot Diseases': {
        'definition': 'Caused by various fungi and bacteria, leaf spots affect the foliage of many plants.',
        'symptoms': 'Dark, circular lesions with yellow halos on leaves.',
        'remedy': 'Remove infected leaves, apply fungicides like copper-based products.',
        'link':'https://extension.umn.edu/plant-diseases/leaf-spot-diseases-trees-and-shrubs'
    },
    'Rusts': {
        'definition': 'Caused by several types of rust fungi such as Puccinia.',
        'symptoms': 'Orange, red, or brown spots with powdery spore masses on the underside of leaves.',
        'remedy': 'Remove infected leaves, apply fungicides, and avoid overcrowding.',
        'link': 'https://extension.umn.edu/plant-diseases/rust-flower-garden'
    },
    'Bacterial Spot': {
        'definition': 'Caused by Xanthomonas bacteria, it affects a wide variety of plants.',
        'symptoms': 'Water-soaked lesions on leaves, stems, and fruits, often with a yellow margin.',
        'remedy': 'Use copper-based bactericides, remove infected plant material.',
        'link': 'https://hort.extension.wisc.edu/articles/bacterial-spot-of-tomato/#:~:text=Bacterial%20spot%20can%20affect%20all,wet%2Dlooking)%20circular%20areas.'
    },
    'Fire Blight': {
        'definition': 'Caused by Erwinia amylovora, affecting apple and pear trees.',
        'symptoms': 'Dark, sunken lesions on branches, withered and blackened leaves, resembling fire damage.',
        'remedy': 'Prune infected parts, use streptomycin or copper sprays.',
        'link':'https://extension.umn.edu/plant-diseases/fire-blight#:~:text=Fire%20blight%20is%20a%20disease,may%20appear%20scorched%20by%20fire.'
    },
    'Bacterial Wilt': {
        'definition': 'Caused by Ralstonia solanacearum, it primarily affects solanaceous crops like tomatoes and potatoes.',
        'symptoms': 'Wilting and yellowing of leaves, brown staining of vascular tissue.',
        'remedy': 'Use resistant varieties, apply soil fumigants.',
        'link':'https://extension.umn.edu/disease-management/bacterial-wilt'
    },
    'Crown Gall': {
        'definition': 'Caused by Agrobacterium tumefaciens, it forms galls on roots and stems of many plants.',
        'symptoms': 'Swelling or galls on the crown of the plant near the soil line.',
        'remedy': 'Remove infected plants, use resistant rootstocks.',
        'link': 'https://hort.extension.wisc.edu/articles/crown-gall/'
    },
    'Mosaic Virus': {
        'definition': 'A viral disease caused by various viruses such as Tobacco Mosaic Virus (TMV).',
        'symptoms': 'Yellowing, mottling, and stunting of leaves, deformed fruits.',
        'remedy': 'Remove infected plants, use virus-resistant varieties, avoid tobacco products.',
        'link':'https://hort.extension.wisc.edu/articles/cucumber-mosaic/'
    },
    'Tomato Spotted Wilt Virus': {
        'definition': 'Caused by Tospovirus, it affects tomatoes and other crops.',
        'symptoms': 'Mottled and spotted leaves, stunted growth.',
        'remedy': 'Remove infected plants, use resistant varieties, and control thrips.',
        'link': 'https://hort.extension.wisc.edu/articles/tomato-spotted-wilt-of-potato/'
    },
    'Cucumber Mosaic Virus': {
        'definition': 'A viral disease affecting cucumbers, melons, and other cucurbits.',
        'symptoms': 'Yellowing, mottling, and curling of leaves.',
        'remedy': 'Remove infected plants, control aphid populations, use resistant varieties.',
        'link': 'https://hort.extension.wisc.edu/articles/cucumber-mosaic/'
    },
    'Clubroot': {
        'definition': 'Caused by the soil-borne fungus Plasmodiophora brassicae, it affects cruciferous plants.',
        'symptoms': 'Swelling of roots, stunted growth, yellowing leaves.',
        'remedy': 'Use resistant varieties, practice crop rotation, improve soil drainage.',
        'link':'https://extension.umn.edu/plant-diseases/clubroot#:~:text=Clubroot%20is%20a%20disease%20that,into%20thick%2C%20irregular%20club%20shapes.'
    },
    'Fusarium Wilt': {
        'definition': 'Caused by Fusarium oxysporum, it affects many crops including tomatoes and bananas.',
        'symptoms': 'Yellowing of leaves, wilting, and vascular discoloration in the stem.',
        'remedy': 'Use resistant varieties, practice crop rotation.',
        'link': 'https://extension.umn.edu/disease-management/fusarium-wilt'
    },
    'Verticillium Wilt': {
        'definition': 'Caused by Verticillium species, it affects a variety of plants, especially tomatoes and peppers.',
        'symptoms': 'Wilting of leaves, yellowing, and brown streaks in vascular tissue.',
        'remedy': 'Use resistant varieties, practice crop rotation, and avoid overhead irrigation.',
        'link': 'https://hort.extension.wisc.edu/articles/verticillium-wilt-of-trees-and-shrubs/'
    },
    'Blossom End Rot': {
        'definition': 'Caused by calcium deficiency, primarily affects tomatoes, peppers, and cucumbers.',
        'symptoms': 'Black, sunken lesions at the blossom end of the fruit.',
        'remedy': 'Ensure consistent watering and apply calcium supplements.',
        'link':'https://hort.extension.wisc.edu/articles/blossom-end-rot/#:~:text=What%20is%20blossom%20end%20rot,and%20rots%2C%20thus%20reducing%20yield.'
    },
    'Downy Mildew': {
        'definition': 'Caused by several species of Peronospora, it affects a wide range of plants, including cucumbers and grapes.',
        'symptoms': 'Yellowish spots on leaves, with fuzzy gray fungal growth on the underside.',
        'remedy': 'Use fungicides, improve air circulation, and avoid overhead watering.',
        'link':'https://extension.umn.edu/disease-management/downy-mildew-cucurbits'
    },
    'Root Rot': {
        'definition': 'Caused by soil fungi such as Pythium or Phytophthora, it leads to decay of plant roots.',
        'symptoms': 'Yellowing leaves, wilting, and brown, decayed roots.',
        'remedy': 'Avoid overwatering, improve soil drainage, use fungicides.',
        'link': 'https://hort.extension.wisc.edu/articles/root-rots-houseplants/'
    },
    'Yellow Leaf Curl Virus': {
        'definition': 'A viral disease caused by Begomovirus, it primarily affects tomatoes and peppers.',
        'symptoms': 'Curling and yellowing of leaves, stunted growth.',
        'remedy': 'Remove infected plants, control whitefly populations.',
        'link': 'https://agriculture.vic.gov.au/biosecurity/plant-diseases/vegetable-diseases/tomato-yellow-leaf-curl-virus'
    },
    'Alternaria Leaf Spot': {
        'definition': 'Caused by Alternaria species, it affects a variety of plants including tomatoes and cucumbers.',
        'symptoms': 'Dark, round lesions on leaves, often with concentric rings.',
        'remedy': 'Use fungicides, remove infected leaves, and improve air circulation.',
        'link': 'https://extension.umn.edu/disease-management/alternaria-leaf-blight'
    }
}


def classify_disease(image_path):
    """Classify the uploaded image and return the predicted disease"""
    # Resize the image to the correct shape (150, 150)
    img = image.load_img(image_path, target_size=(150, 150))  # Update the target_size here
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    # Perform prediction
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_disease = DISEASE_CLASSES[predicted_class_idx]

    return predicted_disease


def upload_image(request):
    """Upload an image for disease detection"""
    if request.method == "POST" and request.FILES["image"]:
        uploaded_image = request.FILES["image"]
        new_disease = PlantDisease(image=uploaded_image)
        new_disease.save()

        # Process image for disease classification
        disease_name = classify_disease(new_disease.image.path)
        new_disease.predicted_disease = disease_name
        new_disease.save()

        # Retrieve disease details
        disease_info = DISEASE_DETAILS.get(disease_name, {
            'definition': 'Information not available.',
            'symptoms': 'Symptoms not available.',
            'remedy': 'Remedy not available.'
            
        })
        

        return render(request, "upload_image.html", {
            "image": new_disease.image.url,
            "disease_name": disease_name,
            "definition": disease_info['definition'],
            "symptoms": disease_info['symptoms'],
            "remedy": disease_info['remedy'],
            'link':disease_info['link']
        })
        

    return render(request, "upload_image.html")
