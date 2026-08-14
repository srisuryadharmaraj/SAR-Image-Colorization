# import os
# import time
# import requests
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from PIL import Image
# import numpy as np

# # Importing machine learning libraries for image classification
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing.image import img_to_array
# import tensorflow as tf

# class ImageScraperCategorizer:
#     def __init__(self, max_images=25, save_dir="downloaded_images"):
#         """
#         Initialize the image scraper and categorizer.
        
#         :param max_images: Maximum number of images to download
#         :param save_dir: Base directory to save downloaded images
#         """
#         # Setup WebDriver (Chrome)
#         self.service = Service(r'D:\\chromedriver-win64\\chromedriver.exe')  # Replace with your ChromeDriver path
#         self.driver = webdriver.Chrome(service=self.service)
        
#         # Create base download directory
#         self.save_dir = save_dir
#         os.makedirs(self.save_dir, exist_ok=True)
        
#         # Maximum images to download
#         self.max_images = max_images
        
#         # Load pre-trained image classification model
#         self.model = ResNet50(weights='imagenet')
    
#     def download_images(self, url):
#         """
#         Download images from the given URL.
        
#         :param url: Website URL to scrape images from
#         :return: List of downloaded image paths
#         """
#         try:
#             self.driver.get(url)
#             time.sleep(3)  
            
#             # Find all image elements
#             img_elements = self.driver.find_elements(By.TAG_NAME, "img")
#             print(f"Found {len(img_elements)} image elements on the page.")
            
#             downloaded_images = []
#             count = 0
            
#             for img in img_elements:
#                 if count >= self.max_images:
#                     break
#                 try:
#                     img_url = img.get_attribute("src")
#                     if img_url and img_url.startswith(('http', 'https')):
#                         response = requests.get(img_url, stream=True)
#                         img_path = os.path.join(self.save_dir, f"image_{count + 1}.jpg")
                        
#                         # Save the image
#                         with open(img_path, 'wb') as img_file:
#                             img_file.write(response.content)
                        
#                         downloaded_images.append(img_path)
#                         count += 1
#                 except Exception as e:
#                     print(f"Failed to download image: {e}")
            
#             print(f"Successfully downloaded {count} images.")
#             return downloaded_images
        
#         finally:
#             self.driver.quit()
    
#     def categorize_images(self, image_paths):
#         """
#         Categorize downloaded images using image classification.
        
#         :param image_paths: List of paths to downloaded images
#         """
#         # Predefined categories based on ImageNet classes
#         categories = {
#             'vehicles': ['sports_car', 'bicycle', 'motorcycle', 'car', 'ambulance', 'fire_engine'],
#             'animals': ['tiger', 'lion', 'leopard', 'zebra', 'elephant', 'bear', 'dog', 'cat'],
#             'nature': ['mountain', 'forest', 'lake', 'beach'],
#             'objects': ['laptop', 'phone', 'camera', 'book'],
#             'people': ['baseball_player', 'golfer', 'swimmer', 'tennis_player']
#         }
        
#         # Create category directories
#         for category in categories:
#             os.makedirs(os.path.join(self.save_dir, category), exist_ok=True)
        
#         # Categorize each image
#         for image_path in image_paths:
#             try:
#                 # Load and preprocess the image
#                 img = Image.open(image_path)
                
#                 # Convert image to RGB if it has an alpha channel
#                 if img.mode == 'RGBA':
#                     img = img.convert('RGB')
#                 elif img.mode != 'RGB':
#                     img = img.convert('RGB')
                
#                 img = img.resize((224, 224))  # ResNet50 requires 224x224 input
#                 x = img_to_array(img)
#                 x = np.expand_dims(x, axis=0)
#                 x = preprocess_input(x)
                
#                 # Predict image category
#                 preds = self.model.predict(x)
#                 decoded_preds = decode_predictions(preds, top=3)[0]
                
#                 # Find the first matching category
#                 categorized = False
#                 for category, class_names in categories.items():
#                     for (number, label, score) in decoded_preds:
#                         print(label)
#                         if label in class_names and score > 0.3:  # Confidence threshold
#                             # Move image to category folder
#                             new_path = os.path.join(self.save_dir, category, os.path.basename(image_path))
#                             os.rename(image_path, new_path)
#                             print(f"Categorized {os.path.basename(image_path)} as {category} (matched {label})")
#                             categorized = True
#                             break
#                     if categorized:
#                         break
                
#                 # If no category found, keep in main folder
#                 if not categorized:
#                     print(f"Could not categorize {os.path.basename(image_path)}")
            
#             except Exception as e:
#                 print(f"Error categorizing {image_path}: {e}")
    
#     def scrape_and_categorize(self, url):
#         """
#         Combined method to scrape and categorize images.
        
#         :param url: Website URL to scrape images from
#         """
#         # Download images
#         downloaded_images = self.download_images(url)
        
#         # Categorize downloaded images
#         self.categorize_images(downloaded_images)

# # Example usage
# def main():
#     # Get URL from user input
#     target_url = input("Enter the URL to scrape images from: ").strip()
    
#     # Initialize and run scraper
#     scraper = ImageScraperCategorizer(max_images=50)  # Increased max images for more categories
#     scraper.scrape_and_categorize(target_url)

# if __name__ == "__main__":
#     main()

# """
# Additional Notes and Prerequisites:
# 1. Install required libraries:
#    pip install selenium requests pillow tensorflow

# 2. Download ChromeDriver and update the path in the script

# 3. The script uses ResNet50 pre-trained model from Keras/TensorFlow 
#    - This provides good general image classification
#    - Categories are based on ImageNet classes
#    - Confidence threshold can be adjusted

# 4. Limitations:
#    - Not 100% accurate in categorization
#    - Works best with clear, distinct images
#    - May struggle with complex or ambiguous images
# """


import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

class ImageScraper:
    def __init__(self, max_images=25, base_dir="downloads"):
        """
        Initialize the image scraper.

        :param max_images: Maximum number of images to download
        :param base_dir: Base directory to save downloaded images
        """
        # Setup WebDriver (Chrome)
        self.service = Service(r'D:\\chromedriver-win64\\chromedriver.exe')  # Replace with your ChromeDriver path
        self.driver = webdriver.Chrome(service=self.service)

        # Create base download directory
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        # Maximum images to download
        self.max_images = max_images

    def download_images(self, url, query):
        """
        Download images from the given URL.

        :param url: Website URL to scrape images from
        :param query: Query for which images are being scraped (used to create a subfolder)
        :return: List of downloaded image paths
        """
        try:
            # Create a subfolder for the query
            query_dir = os.path.join(self.base_dir, query.replace(" ", "_"))
            os.makedirs(query_dir, exist_ok=True)

            self.driver.get(url)
            time.sleep(3)  # Allow time for the page to load

            # Find all image elements
            img_elements = self.driver.find_elements(By.TAG_NAME, "img")
            print(f"Found {len(img_elements)} image elements on the page for query: {query}.")

            downloaded_images = []
            count = 0

            for img in img_elements:
                if count >= self.max_images:
                    break
                try:
                    img_url = img.get_attribute("src")
                    if img_url and img_url.startswith(('http', 'https')):
                        response = requests.get(img_url, stream=True)
                        img_path = os.path.join(query_dir, f"image_{count + 1}.jpg")

                        # Save the image
                        with open(img_path, 'wb') as img_file:
                            img_file.write(response.content)

                        downloaded_images.append(img_path)
                        count += 1
                except Exception as e:
                    print(f"Failed to download image: {e}")

            print(f"Successfully downloaded {count} images for query: {query}.")
            return downloaded_images

        finally:
            self.driver.quit()

# Example usage
def main():
    # List of queries
    query_list = ["animals", "plants"]

    # Initialize scraper
    scraper = ImageScraper(max_images=50)  # Adjust max_images as needed

    for query in query_list:
        target_url = f"https://unsplash.com/s/photos/{query}"
        retries = 3
        success = False

        while retries > 0 and not success:
            try:
                scraper.download_images(target_url, query)
                success = True
            except requests.ConnectionError as e:
                print(f"Connection error for query '{query}': {e}. Retrying...")
                retries -= 1
                time.sleep(5)  # Delay before retrying

        if not success:
            print(f"Failed to download images for query: {query} after multiple attempts.")

        time.sleep(5)  # Pause between queries

if __name__ == "__main__":
    main()
