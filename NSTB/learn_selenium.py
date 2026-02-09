import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


from selenium.webdriver.support.ui import Select

brave_path = "/usr/bin/brave-browser"
download_dir = "/home/amit/Projects/CTEVT/rpl_files_27"

chrome_options = Options()
chrome_options.binary_location = brave_path
chrome_options.add_experimental_option("detach", True)

chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True,
})

# --- This line allows insecure downloads ---
chrome_options.add_argument("--allow-running-insecure-content")
chrome_options.add_argument("--disable-web-security")  # Optional: disables cross-origin and mixed content restrictions
chrome_options.add_argument("--ignore-certificate-errors")  # Optional: for self-signed certs


driver = webdriver.Chrome(options=chrome_options)


driver.get("http://202.45.147.85:1515")
# driver.get("http://google.com")
driver.maximize_window()

user_inp = input('Enter User and password in the Browser and the HIt enter:')

# driver.find_element(By.XPATH, "//p[text()=' Report ']")
driver.execute_script("window.open('http://202.45.147.85:1515/Candidate/RptCandidateProfile?modereport=1', '_blank');")

# Switch to the new tab (this is important if you want to interact with the new tab)
driver.switch_to.window(driver.window_handles[-1])

time.sleep(5)

for i in range(19, 21):
    
    year = driver.find_element(By.ID, 'Year')
    year_sel = Select(year)
    year_sel.select_by_visible_text('2081/2082')
    
    file_id = driver.find_element(By.ID, 'FielnOsearch')
    file_id.send_keys(f'RPLPGA-{i}')

    search_button = driver.find_element(By.ID, 'btnSearchFilenO')
    search_button.click()
    time.sleep(30)

    driver.switch_to.frame('WebForm')
    div_element = driver.find_element(By.ID, "ReportViewer1_ctl09_ctl04_ctl00")
    div_element.click()
    time.sleep(2)
    driver.execute_script("$find('ReportViewer1').exportReport('EXCELOPENXML');")
    driver.switch_to.default_content()
    time.sleep(10)
    
    driver.execute_script("window.open('http://202.45.147.85:1515/Candidate/RptCandidateProfile?modereport=1', '_self');")