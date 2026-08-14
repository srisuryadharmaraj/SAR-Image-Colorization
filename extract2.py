from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import os
import time
import requests

def setup_driver():
    """Sets up the Selenium WebDriver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")
    service = Service('D:\\chromedriver-win64\\chromedriver.exe')  # Replace with your ChromeDriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def search_and_download_images(query, download_folder, num_images=100):
    """
    Searches Google Images for the query and downloads images.

    :param query: Search query (e.g., "animals").
    :param download_folder: Folder to save downloaded images.
    :param num_images: Number of images to download.
    """
    driver = setup_driver()
    search_url = f"https://unsplash.com/s/photos/{query.replace(' ', '+')}"
    
    print(search_url)
    driver.get(search_url)

    # Scroll to load enough images
    for _ in range(5):  # Adjust scrolls if necessary
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        time.sleep(2)  # Allow time for images to load

    # Get image elements
    image_elements = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
    print(f"Found {len(image_elements)} images on the page.")

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    count = 0
    for img_elem in image_elements[:num_images]:
        try:
            img_url = img_elem.get_attribute("src") or img_elem.get_attribute("data-src")
            if img_url and img_url.startswith("http"):
                # Download the image
                response = requests.get(img_url, stream=True)
                if response.status_code == 200:
                    filename = os.path.join(download_folder, f"{query}_{count + 1}.jpg")
                    with open(filename, 'wb') as file:
                        for chunk in response.iter_content(1024):
                            file.write(chunk)
                    print(f"Downloaded: {filename}")
                    count += 1
        except Exception as e:
            print(f"Failed to download image. Error: {e}")
        
        if count >= num_images:
            break

    print(f"Downloaded {count}/{num_images} images.")
    driver.quit()

# Run the function
search_query = "tiger"
output_folder = "selenium_downloads"
search_and_download_images(search_query, output_folder, num_images=5)


''' https://unsplash.com/s/photos/tiger '''