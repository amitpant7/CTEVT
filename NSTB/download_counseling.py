from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from selenium.webdriver.common.action_chains import ActionChains
import time
import os
import glob
import shutil
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
COUNSELLING_FILE_ID = 1261  # Change this to your required ID


#all ids
file_names = [
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025,
    2025, 2025, 2025, 2025, 2025, 2025
]


download_dir = 'CounselingFiles'
os.makedirs(download_dir, exist_ok=True)
logger.info(f"Download directory: {download_dir}")

# Generate a temporary profile directory
temp_profile_dir = tempfile.mkdtemp(prefix="brave_profile_")
logger.info(f"Temporary profile directory: {temp_profile_dir}")

# Setup Brave Browser with Selenium
brave_path = "/usr/bin/brave-browser"  # Path to Brave on Fedora
chrome_options = Options()
chrome_options.binary_location = brave_path

# Use fresh temporary user profile
chrome_options.add_argument(f"--user-data-dir={temp_profile_dir}")

chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,  # Auto-download directory
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

# Initialize WebDriver
try:
    browser = webdriver.Chrome(options=chrome_options)
    logger.info("WebDriver initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize WebDriver: {e}")
    raise

try:
    # 1. Open the login page
    login_url = "http://202.45.147.85:1515/Account/Login"
    browser.get(login_url)
    logger.info(f"Opened login page: {login_url}")

    # 2. Wait for manual login
    print("Please log in manually in the opened browser window.")
    input("Press Enter after logging in manually...")
    logger.info("Manual login completed")


## Now make it function 
    # 3. Go to the download page
    url = f"http://202.45.147.85:1515/CounsellingFile/CounsellingCandidateList?CounsellingFileID={COUNSELLING_FILE_ID}&tabName=Verified"
    browser.get(url)
    logger.info(f"Navigated to candidate list page: {url}")

    wait = WebDriverWait(browser, 30)  # Increased wait time for better reliability

    # 4. Click "Candidate List"
    try:
        candidate_list_button = wait.until(EC.element_to_be_clickable((By.ID, "btnCounsellingReportForTab")))
        logger.info("Found 'Candidate List' button")
        candidate_list_button.click()
        logger.info("Clicked on 'Candidate List' button")
    except TimeoutException:
        logger.error("Candidate List button not found or not clickable")
        raise

    # 5. Switch to new tab with better waiting
    # Wait for the new tab to open
    start_time = time.time()
    timeout = 20  # seconds to wait for new tab
    
    while len(browser.window_handles) < 2 and (time.time() - start_time) < timeout:
        logger.info("Waiting for new tab to open...")
        time.sleep(1)
    
    if len(browser.window_handles) < 2:
        logger.error("New tab did not open within the timeout period")
        raise TimeoutException("New tab did not open")
    
    # Switch to the new tab
    browser.switch_to.window(browser.window_handles[-1])
    logger.info("Switched to the report viewer tab")
    
    # Wait for the report viewer to load
    try:
        wait.until(lambda driver: driver.title != "")
        logger.info(f"New tab loaded with title: {browser.title}")
    except TimeoutException:
        logger.warning("Timed out waiting for page title, will proceed anyway")
    
    # Wait for the report viewer iframe to be present and switch to it if exists
    try:
        # First check if there's an iframe containing the report viewer
        iframes = browser.find_elements(By.TAG_NAME, "iframe")
        if iframes:
            logger.info(f"Found {len(iframes)} iframes on the page")
            for idx, iframe in enumerate(iframes):
                try:
                    logger.info(f"Switching to iframe {idx}")
                    browser.switch_to.frame(iframe)
                    # Check if we can find report elements in this iframe
                    if browser.find_elements(By.ID, "ReportViewer1") or browser.find_elements(By.CLASS_NAME, "msrs-reportview"):
                        logger.info(f"Report viewer found in iframe {idx}")
                        break
                    # If not found, switch back to main content
                    browser.switch_to.default_content()
                except Exception as e:
                    logger.warning(f"Error switching to iframe {idx}: {e}")
                    browser.switch_to.default_content()
        
        # Wait for report viewer to be present either in iframe or main page
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#ReportViewer1, .msrs-reportview, .rvToolbar")))
        logger.info("Report viewer element found")
    except TimeoutException:
        logger.warning("Report viewer element not found within timeout, but will try to proceed")
    
    # Additional wait for report to render
    print("Waiting for report to fully load...")
    time.sleep(10)  # Give extra time for the report to render completely
    logger.info("Waited 10 seconds for report to render")

    # 6. Try multiple approaches to export the report
    export_success = False
    
    # Approach 1: Try using the toolbar buttons
    if not export_success:
        try:
            logger.info("Attempting to find export button in the toolbar")
            # Look for export button with various selectors
            export_selectors = [
                "#ReportViewer1_ctl09_ctl04_ctl00_ButtonLink",  # Original selector
                "#ReportViewer1_ctl08_ctl04_ctl00",
                "#ReportViewer1_ctl09_ctl04_ctl00",
                "#ReportViewer1_ctl10_ctl04_ctl00",
                ".rvToolbar .glyphui-save",  # Class-based selector
                "[title='Export']",  # Title-based selector
                "a[title*='Export']",  # Partial title match
                "div[title*='Export']"  # Div with export title
            ]
            
            for selector in export_selectors:
                try:
                    elements = browser.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        logger.info(f"Found {len(elements)} elements with selector {selector}")
                        for idx, element in enumerate(elements):
                            try:
                                logger.info(f"Attempting to click element {idx} with selector {selector}")
                                element.click()
                                logger.info(f"Successfully clicked export button with selector {selector}")
                                time.sleep(3)  # Wait for menu to appear
                                
                                # Look for Excel option
                                excel_selectors = [
                                    "a[title='Excel']",
                                    "a:contains('Excel')",
                                    "li[title='Excel']",
                                    ".rvMenuItem:contains('Excel')"
                                ]
                                
                                for excel_selector in excel_selectors:
                                    try:
                                        excel_elements = browser.find_elements(By.CSS_SELECTOR, excel_selector)
                                        if excel_elements:
                                            excel_elements[0].click()
                                            logger.info(f"Clicked Excel option with selector {excel_selector}")
                                            export_success = True
                                            break
                                    except Exception as e:
                                        logger.warning(f"Failed to click Excel with selector {excel_selector}: {e}")
                                
                                if export_success:
                                    break
                            except Exception as e:
                                logger.warning(f"Failed to click element {idx} with selector {selector}: {e}")
                        
                        if export_success:
                            break
                except Exception as e:
                    logger.warning(f"Error with selector {selector}: {e}")
        except Exception as e:
            logger.warning(f"Approach 1 failed: {e}")

    # Approach 2: Try right-click context menu approach
    if not export_success:
        try:
            logger.info("Attempting right-click approach")
            # Find the report content area
            report_area_selectors = [
                "#ReportViewer1_fixedTable",
                ".msrs-reportview",
                "#ReportViewer1"
            ]
            
            for selector in report_area_selectors:
                try:
                    elements = browser.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        # Create action chain for right click
                        action = ActionChains(browser)
                        action.context_click(elements[0]).perform()
                        logger.info(f"Right-clicked on report area with selector {selector}")
                        
                        # Wait for context menu
                        time.sleep(2)
                        
                        # Look for export option in context menu
                        context_menu_selectors = [
                            "a:contains('Export')",
                            "li:contains('Export')",
                            "div:contains('Export')"
                        ]
                        
                        for cm_selector in context_menu_selectors:
                            try:
                                cm_elements = browser.find_elements(By.CSS_SELECTOR, cm_selector)
                                if cm_elements:
                                    cm_elements[0].click()
                                    logger.info(f"Clicked Export in context menu with selector {cm_selector}")
                                    time.sleep(2)
                                    
                                    # Now look for Excel
                                    excel_selectors = [
                                        "a:contains('Excel')",
                                        "li:contains('Excel')",
                                        "div:contains('Excel')"
                                    ]
                                    
                                    for excel_selector in excel_selectors:
                                        try:
                                            excel_elements = browser.find_elements(By.CSS_SELECTOR, excel_selector)
                                            if excel_elements:
                                                excel_elements[0].click()
                                                logger.info(f"Clicked Excel option with selector {excel_selector}")
                                                export_success = True
                                                break
                                        except Exception as e:
                                            logger.warning(f"Failed with Excel selector {excel_selector}: {e}")
                                    
                                    if export_success:
                                        break
                            except Exception as e:
                                logger.warning(f"Failed with context menu selector {cm_selector}: {e}")
                        
                        if export_success:
                            break
                except Exception as e:
                    logger.warning(f"Error with report area selector {selector}: {e}")
        except Exception as e:
            logger.warning(f"Approach 2 failed: {e}")

    # Approach 3: Try to find export button by taking screenshot and analyzing HTML structure
    if not export_success:
        try:
            logger.info("Attempting to debug export button via screenshot approach")
            # Take screenshot for debugging
            screenshot_path = os.path.join(download_dir, "report_screenshot.png")
            browser.save_screenshot(screenshot_path)
            logger.info(f"Saved screenshot to {screenshot_path}")
            
            # Print out HTML structure around likely export buttons
            logger.info("HTML structure for debugging:")
            try:
                toolbar_elements = browser.find_elements(By.CSS_SELECTOR, ".rvToolbar, .ToolBar")
                if toolbar_elements:
                    logger.info(f"Found {len(toolbar_elements)} toolbar elements")
                    for idx, toolbar in enumerate(toolbar_elements):
                        logger.info(f"Toolbar {idx} HTML: {toolbar.get_attribute('outerHTML')}")
                else:
                    logger.info("No toolbar elements found")
            except Exception as e:
                logger.warning(f"Error getting toolbar HTML: {e}")
            
            # Interactive mode for manual export
            print("\n*** IMPORTANT: Automatic export failed. ***")
            print("Please manually export the report to Excel through the UI.")
            print("1. Look for an export button in the toolbar (may look like a disk or have 'Export' text)")
            print("2. Click it and select 'Excel' from the dropdown")
            print("3. If that doesn't work, try right-clicking on the report and look for export options")
            input("Press Enter after manually exporting the report to Excel...")
            logger.info("User indicated manual export was completed")
            export_success = True
            
        except Exception as e:
            logger.warning(f"Approach 3 failed: {e}")

    if not export_success:
        logger.error("All export approaches failed")
        raise Exception("Failed to export report to Excel")

    # 8. Wait for the file to download
    logger.info("Waiting for download to complete...")
    print("Waiting for Excel file to download...")
    download_complete = False
    timeout = 120  # Extended timeout for download (seconds)
    start_time = time.time()

    while not download_complete and (time.time() - start_time) < timeout:
        time.sleep(2)
        downloading_files = glob.glob(os.path.join(download_dir, "*.crdownload")) + \
                           glob.glob(os.path.join(download_dir, "*.part")) + \
                           glob.glob(os.path.join(download_dir, "*.tmp"))
        
        # Check for various possible filenames the report might have
        excel_patterns = [
            "RptCandidateCounsellingFileRdlc.xlsx",
            "*.xlsx",  # Any Excel file
            "Counselling*.xlsx",
            "Report*.xlsx",
            "CounsellingCandidateList*.xlsx"
        ]
        
        downloaded_files = []
        for pattern in excel_patterns:
            downloaded_files.extend(glob.glob(os.path.join(download_dir, pattern)))
        
        if downloading_files:
            print("Download in progress...")
        
        if not downloading_files and downloaded_files:
            download_complete = True
            logger.info(f"Download completed successfully: {downloaded_files}")
            print("Download completed!")

    if not download_complete:
        logger.warning(f"Download timed out after {timeout} seconds")
        print(f"WARNING: Download timed out after {timeout} seconds")

    # 9. Rename the downloaded file
    all_xlsx_files = glob.glob(os.path.join(download_dir, "*.xlsx"))
    
    if all_xlsx_files:
        # Sort by modification time (newest first)
        all_xlsx_files.sort(key=os.path.getmtime, reverse=True)
        src_file = all_xlsx_files[0]  # Take the most recent Excel file
        new_filename = f"CFPGA_{COUNSELLING_FILE_ID}_2025.xlsx"
        dst_file = os.path.join(download_dir, new_filename)
        shutil.move(src_file, dst_file)
        logger.info(f"File renamed to: {new_filename}")
        print(f"SUCCESS: File downloaded and saved as {new_filename}")
    else:
        logger.error("Downloaded file not found!")
        print("ERROR: Download failed or file not found")
        
        # List all files in download directory for debugging
        all_files = os.listdir(download_dir)
        if all_files:
            logger.info(f"Files in download directory: {all_files}")
            print(f"Files found in download directory: {all_files}")
        else:
            logger.info("Download directory is empty")
            print("Download directory is empty")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
    print(f"ERROR: {e}")

finally:
    logger.info("Cleaning up...")
    time.sleep(5)
    
    # Ask user if browser should be closed (in case they want to continue manually)
    close_browser = input("Close browser window? (y/n): ").strip().lower() == 'y'
    if close_browser:
        browser.quit()
        logger.info("Browser closed")
    else:
        logger.info("Browser left open at user request")

    # Cleanup temp Brave profile
    try:
        if close_browser:  # Only remove profile if browser was closed
            shutil.rmtree(temp_profile_dir)
            logger.info(f"Temporary profile at {temp_profile_dir} deleted")
    except Exception as e:
        logger.warning(f"Failed to delete temporary profile: {e}")