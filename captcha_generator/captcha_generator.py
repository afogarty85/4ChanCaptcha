from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time as t
import undetected_chromedriver as uc
import pandas as pd


storage_df = pd.DataFrame()

for i in range(500):
    print(f'Now on captcha generating attempt: {i}')
    try:
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument('--disable-notifications')
        options.add_argument("--window-size=1280,720")
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--allow-running-insecure-content')
        browser = uc.Chrome(options=options)

        # load browser but take 5 seconds to do so
        wait = WebDriverWait(browser, 7)

        # the board
        url = 'https://boards.4channel.org/o/'

        # open url
        browser.get(url)

        # start a thread
        wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, 'Start a New Thread'))).click()
        # wait 5 mins
        t.sleep(301)
        wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@id="t-load"]'))).click()

        # extract background and foreground
        b64img_background = wait.until(EC.element_to_be_clickable((By.XPATH, '//div[@id="t-bg"]'))).get_attribute('style').split('url("data:image/png;base64,')[1].split('");')[0]
        b64img_foreground = wait.until(EC.element_to_be_clickable((By.XPATH, '//div[@id="t-fg"]'))).get_attribute('style').split('url("data:image/png;base64,')[1].split('");')[0]

        # store the data into a dataframe
        temp_df = pd.DataFrame({'bg': [b64img_background],
                                'fg': [b64img_foreground]
                                })

        # add it to the main results
        storage_df = pd.concat([temp_df, storage_df], axis=0)

        # close chrome
        browser.close()
        browser.quit()

    # log errors
    except Exception as e:
        print(e)
        continue

# check shape
storage_df.shape

# results:
storage_df.to_csv('captcha.csv', index=False)
