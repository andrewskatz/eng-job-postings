# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 06:03:52 2021

@author: akatz4
"""


"""

Engineering Jobs data collection


"""





import requests
from bs4 import BeautifulSoup as bs
import os
#import re
import pandas as pd
#import pickle
#import itertools
#import numpy as np
from time import sleep
#from collections import OrderedDict
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common import action_chains
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.select import Select

import shutil




"""

Setup working directory

"""

os.getcwd()
proj_path = "G:/My Drive/AK Faculty/Research/Projects/project political economy of engineering education/project engineering jobs"
os.chdir(proj_path)
os.getcwd()
os.listdir()




"""

Setup selenium and chromedriver

"""



driver_path = "C:\\Users\\akatz4\\AppData\\Local\\Continuum\\anaconda3\\Lib\\site-packages\\selenium\\webdriver\\chrome\\chromedriver_win32\\chromedriver.exe"
# with options
#driver = webdriver.Chrome(driver_path, options = options)
# without options
driver = webdriver.Chrome(driver_path)

# https://au.engineerjobs.com/jobs/search?ak=mechanical+engineering&l=anywhere&job=TtqvwoXFwKYJKNgpgo2EMp9Sie6YlsQBOux1QJviK6bMRbUR7Bjyzw



## to handle popup and close
# //button[@id='modal-close']
def check_popup():
    try: 
        driver.find_element_by_xpath("//button[@id='modal-close']").click()
    except:
        return



## to handle cookies
# //button[@id='onetrust-accept-btn-handler']
def close_cookies():
    try:
        driver.find_element_by_xpath("//button[@id='onetrust-accept-btn-handler']").click()
    except:
        return

close_cookies()




def run_search(storage_list, search_term="civil engineering", search_location="anywhere", country="US"):

    # switch for different country websites
    if country == "US":
        stem_url = "https://www.engineerjobs.com/"
        landing_url = "https://www.engineerjobs.com/"
    elif country == "UK":
        stem_url = "https://uk.engineerjobs.com/"
        landing_url = "https://uk.engineerjobs.com/"
    elif country == "Australia":
        stem_url = "https://au.engineerjobs.com/"
        landing_url = "https://au.engineerjobs.com/"

    driver.get(landing_url)

    sleep(2)
    
    close_cookies()
    # start entering information for search fields
    
    # enter search term
    search_fld = driver.find_element_by_xpath("//input[@id='TextInput_ak']")
    search_fld.clear()
    search_fld.send_keys(search_term)
    
    # enter search location
    location_fld = driver.find_element_by_xpath("//input[@id='TextInput_l']")
    location_fld.clear()
    location_fld.send_keys(search_location)
    
    
    # click search button
    #search_btn = driver.find_element_by_xpath("//p[@class='SearchBox-searchButtonText']")
    search_btn = driver.find_element_by_xpath("//button[@type='submit']")
    search_btn.click()
    
    sleep(2)
    
    
    # total search numbers
    tot_jobs = driver.find_element_by_xpath("//span[@class='TwoPaneSerp-titleTotalNum']").text
    
    search_location = search_location.replace(" ", "+")
    search_term = search_term.replace(" ", "+")
    

    # empty list for storing each job
    # jobs_list = []

    
    # iteratre through each job card to collect detailed info
    
    for i in range(int(int(tot_jobs)/20+1)):
        page_num = str(i+1)
    
        url = f"{stem_url}jobs/search?ak={search_term}&l={search_location}&page={page_num}"
    
        print(url)
        driver.get(url)
        
        sleep(1)
        
        check_popup()
        
        job_cards = driver.find_elements_by_xpath("//article")
    
        for card in job_cards:
            card.click()
            sleep(0.5)
            
            # collect info in job header (title, company, location, and salary)    
            try:
                title = driver.find_element_by_xpath("//h3[@class='ViewJobHeader-title']").text
            except:
                title = "missing title"
            try:
                company = driver.find_element_by_xpath("//div[@class='ViewJobHeader-company']").text    
            except:
                company = "missing company"
            try:
                location = driver.find_element_by_xpath("//span[@class='ViewJobHeader-property']").text
            except:
                location = "missing location"
            try:
                salary = driver.find_element_by_xpath("//div[@class='ViewJobHeader-properties']/div").text
            except:
                salary = "missing salary"
            
        
            # collect info in job entities (education, skills)
        
            try:
                ed_list = [ed.text for ed in driver.find_elements_by_xpath("//span[text()='Education']/following-sibling::ul[1]/child::li")]
            except:
                ed_list = "missing education"
            try:
                skill_list = [skill.text for skill in driver.find_elements_by_xpath("//span[text()='Skills']/following-sibling::ul[1]/child::li")]
            except:
                skill_list = "missing skills"
            
            
            # collect full job description text
            try:
                description = driver.find_element_by_xpath("//div[@class='viewjob-description ViewJob-description']").text
            except:
                description = "missing description"
            
            #print(title, company, location, salary, ed_list, skill_list)
            
            job_dict = {
                'title': title,
                'company': company,
                'location': location,
                'salary': salary,
                'education': ed_list,
                'skills': skill_list,
                'description': description,
                'country': country
                }
            
            storage_list.append(job_dict)
    
        #old method for advancing pages
        #nxt_pg = driver.find_element_by_xpath("//a[@class='Pagination-link Pagination-link--next']")
        #driver.get(nxt_pg.get_attribute('href'))
        #sleep(1)
    
    # return storage_list








# save parameters to pass to run_search()
search_term = "computer engineering"
country = "US"
jobs_list = []

# run the search
run_search(jobs_list, search_term=search_term, country=country)

# convert list to dataframe
jobs_df = pd.DataFrame(jobs_list)

# save file
file_name = search_term + "_" + country + ".csv"
file_name = file_name.replace(" ", "_")
jobs_df.to_csv(file_name)









###
# Notes to self
###
"""
ran searches for:
civil engineering
biomedical engineering
chemical engineering
mechanical engineering
electrical engineering (stopped at 40931 even though 51035 were listed - something weird with pop up)



sustainability

"""









# to loop through pages
while True: 
    if not driver.find_element_by_xpath("//a[@class='Pagination-link Pagination-link--next']"):
        break
    nxt_pg = driver.find_element_by_xpath("//a[@class='Pagination-link Pagination-link--next']")
    driver.get(nxt_pg.get_attribute('href'))
    sleep(1)


# get each job card (the short cards on the left hand side) and click it
job_cards = driver.find_elements_by_xpath("//article")

# number of search results

job_title = "mechanical+engineering"
location = "anywhere"





for i in range(10):
    page_num = str(i+1)
    url = f"https://www.engineerjobs.com/jobs/search?ak={job_title}&l={location}&page={page_num}"

    print(url)
    driver.get(url)
    
    sleep(1)


###
# job title: //h3[@class='ViewJobHeader-title']
# job company: //div[@class='ViewJobHeader-company']
# location: //span[@class='ViewJobHeader-property']
# salary: //div[@class='JobCard-salary'] ** this is going to cause problems - need to specify under div //div[@class='ViewJobHeader-properties']


job_entities = driver.find_element_by_xpath("//div[@class='viewjob-entities']")
job_entities.text

# education: //div[@class='viewjob-entities']//span[text()='Education']
driver.find_element_by_xpath("//div[@class='viewjob-entities']//span[text()='Education']").text

for ed in driver.find_elements_by_xpath("//span[text()='Education']/following-sibling::ul[1]/child::li"):
    print(ed.text)


# skills
for skill in driver.find_elements_by_xpath("//span[text()='Skills']/following-sibling::ul[1]/child::li"):
    print(skill.text)


job_desc = driver.find_element_by_xpath("//div[@class='viewjob-description ViewJob-description']")
job_desc.text.replace("\n", "--")



## to handle popup and close
# //button[@id='modal-close']
def check_popup():
    if driver.find_element_by_xpath("//button[@id='modal-close']"):
        driver.find_element_by_xpath("//button[@id='modal-close']").click()



## to handle cookies
# //button[@id='onetrust-accept-btn-handler']
def close_cookies():
    if driver.find_element_by_xpath("//button[@id='onetrust-accept-btn-handler']"):
        driver.find_element_by_xpath("//button[@id='onetrust-accept-btn-handler']").click()



## next page

nxt_pg = driver.find_element_by_xpath("//a[@class='Pagination-link Pagination-link--next']")

driver.get(nxt_pg.get_attribute('href'))


# //div[contains(text(), 'matchtext')]
# //div[text() = 'matchtext']

# example of traversing to parent then sibling
#//a[text()='test2']//parent::td[@class='datalistrow']//preceding-sibling::td[@class='datalistrow']//input[@name='contact_id']
#forward-sibling
















