import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options


def extract_data(url: str) -> str:
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure GUI is off
    chrome_options.add_argument("--no-sandbox")

    homedir = os.path.expanduser("~")
    webdriver_service = Service(f"{homedir}/chromedriver/stable/chromedriver")

    # Choose Chrome Browser
    driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

    content = ""
    try:
        # Direct the browser to the target URL
        driver.get(url)

        # Give the JavaScript time to render (if necessary)
        time.sleep(1)

        xpath_expression = "//*[self::p or self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6]"
        # print("OK1")
        elements = driver.find_elements("xpath", xpath_expression)
        # print("OK2")
        for element in elements:
            if element.text.strip():
                content += "".join(["\n", element.text])

        # # Extract paragraphs
        # paragraphs = driver.find_elements(By.TAG_NAME, 'p')
        # print("\nParagraphs:")
        # for paragraph in paragraphs:
        #     print(paragraph.text)

        # # Extract headers (assuming h1 to h6)
        # print("\nHeaders:")
        # for i in range(1, 7):
        #     headers = driver.find_elements(By.TAG_NAME, f'h{i}')
        #     for header in headers:
        #         print(header.text)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # It's important to close the browser after scraping.
        driver.quit()

    return content


if __name__ == "__main__":
    url_to_scrape = "https://en.wikipedia.org/wiki/2022_Tour_Championship"
    content = extract_data(url=url_to_scrape)
    print(content)
