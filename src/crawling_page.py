from selenium.webdriver.common.action_chains import ActionChains
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
import os


load_dotenv()
url = os.environ.get("URL2")

driver = uc.Chrome()
driver.get(url)

try:
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, '[aria-label="Open table of contents"]')
        )
    )
    dropdown = driver.find_element(
        By.CSS_SELECTOR, '[aria-label="Open table of contents"]')
    print("DROPDOWN: ", dropdown)
    dropdown.click()
except TimeoutException:
    print("TimeoutException for dropdown")
    quit()
except NoSuchElementException:
    print("NoSuchElementException for dropdown")
    quit()


for _ in range(20):  # Adjust the range as needed
    ActionChains(driver).move_to_element_with_offset(dropdown, 0, 10).perform()
    time.sleep(0.1)  # Adjust the sleep time as needed

# Extract the information from the items
try:
    items = driver.find_elements(
        By.CSS_SELECTOR, 'div>mat-tab-body>div>reader-table-of-contents>mat-nav-list>a')
except NoSuchElementException:
    print("NoSuchElementException for items")
for item in items:
    date = item.find_element(
        By.CLASS_NAME, 'mat-line chapter-label ng-star-inserted').text
    page = item.find_element(By.CLASS_NAME, 'mat-line page-number').text
    print(f"Date: {date}, Page: {page}")
