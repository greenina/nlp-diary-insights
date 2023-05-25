import pandas as pd
import os
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import undetected_chromedriver as uc
from dotenv import load_dotenv

load_dotenv()
url = os.environ.get("URL")


def init_driver():
    driver = uc.Chrome()
    # url = 'https://play.google.com/books/reader?id=kRE8xxCmUcMC&pg=GBS.PA3.w.2.0.0&hl=en_GB'
    driver.get(url)
    print("Before return driver")
    time.sleep(30)
    return driver


def do_login(driver):
    print("DO login")
    iframes = driver.find_elements(By.CSS_SELECTOR, 'iframe')
    driver.switch_to.frame(iframes[0])
    driver.find_element(
        By.XPATH, '/html/body/reader-app/reader-main/reader-app-bar/div/div[2]/reader-sign-in-button').click()
    try:
        WebDriverWait(driver, 90).until(
            EC.presence_of_element_located(
                (By.CLASS_NAME, 'indent')
            )
        )
    except TimeoutException:
        print("TimeoutException")


if __name__ == "__main__":
    driver = init_driver()
    do_login(driver)

    print("End Login")
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    current_path = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(current_path, f"data_{current_time}.csv")

    csv_raw_data = []
    cnt = 0
    while cnt < 10:
        current_url = driver.current_url
        print("현재 페이지 URL:", current_url)

        response = requests.get(current_url)
        iframes = driver.find_elements(By.CSS_SELECTOR, 'iframe')
        driver.switch_to.frame(iframes[0])
        try:
            WebDriverWait(driver, 30).until(EC.presence_of_element_located(
                (By.CLASS_NAME, 'chapter1')))
            date = driver.find_element(By.CLASS_NAME, 'chapter1').text.strip()
            content = ""
        except TimeoutException:
            print("TimeoutException for chapter1")

        try:
            contents_elem = driver.find_elements(By.CLASS_NAME, 'indent')
        except NoSuchElementException:
            print("NoSuchElementException for indent")
            break
        contents_text = [content.text.strip() for content in contents_elem]
        joint_contents = (' ').join(contents_text)

        csv_raw_data.append([current_url, date, joint_contents])

        # prev_button = driver.find_element(
        #     By.CSS_SELECTOR, '[aria-label="Previous page"]')
        try:
            driver.find_element(
                By.CSS_SELECTOR, '[aria-label="Next page"]').click()
        except NoSuchElementException:
            print("NoSuchElementException for next btn")
            break

        cnt += 1

    driver.quit()

    df = pd.DataFrame(csv_raw_data, columns=[
                      "URL", "Creation Time", "Content"])

    df.to_csv(file_path, index=True)
