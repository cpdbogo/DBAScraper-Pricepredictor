
"""
Used for scraping Bang & Olufsen ads from www.DBA.dk.

This was made for a school project about machine learning.
"""

import json
import os
import sys

import pandas as pd
import requests
from fuzzywuzzy import process


class DBAScraper():
    """DBAScraper class for scraping DBA."""
    
    def __init__(self):
        """Initialize various files."""
        with open(os.path.join(sys.path[0], r'../data/allowed_models.json')) as json_file:
            allowed_models = json.load(json_file)  # Load allowed models.
        self.allowed_models = allowed_models
        self.final_df = pd.DataFrame()

        if os.path.exists(os.path.join(sys.path[0], r'../data/scraped_dataframe_records_dummified.json')) is not True:  # If no dummified records are available,
            dummy_col = pd.DataFrame()                                                       # create an empy dataframe with the dummy collumns.
            with open(os.path.join(sys.path[0], r'../data/dummy_columns.json')) as json_file:
                dummy_col = json.load(json_file)
                self.final_df = pd.DataFrame(columns=dummy_col)
        else:
            self.final_df = pd.read_json(os.path.join(sys.path[0], r'../data/scraped_dataframe_records_dummified.json'))

        self.temp_df = pd.DataFrame(columns=['price', 'watt', 'condition', 'model'])  # For loading the scraping results.

    def fuzzy_check(self, row):
        """
        Check that the model is within the allowed models list. If not, drop the row.
        
        Works by using the Levenshtein distance algorithm. (used in FuzzyWuzzy)
        """
        fuzz_check = process.extract(row['model'], self.allowed_models, limit=1)
        if fuzz_check[0][1] >= 95:
            self.printColor(f"## Fuzzy check ## found {fuzz_check[0][0]}, matched with {row['model']}", "green")
            row['model'] = fuzz_check[0][0]
            self.temp_df = self.temp_df.append(row)

    def printColor(self, string, color):
        """Make a print in color, supply string and choose a color. ("blue","green","red")."""
        if (color == "blue"):
            print('\033[94m' + string + '\033[0m')
        elif (color == "green"):
            print('\033[92m' + string + '\033[0m')
        elif (color == "red"):
            print('\033[93m' + string + '\033[0m')
        else:
            raise ValueError("Invalid color " + str(color) + "supported colors: blue, green, red")

    def make_api_call(self, search_query, page_num):
        """
        Make an API call to DBA using page numbers for pagination.
        
        There is no documentation for DBA's API, further development of the scraper has to be done by trial and error.
        """
        url = "https://api.dba.dk/api/v2/ads/search/intermingled?q=" + search_query + "&pn=" + str(page_num) + "&ps=100&format=json"
        self.printColor("### Sending API Request " + url + " ###", "blue")
        response = requests.get(url, headers={"DbaApiKey": "54c1a7af-42ea-47f3-dac9-08d886f5a4d6", "User-Agent": "dba/6.8.0 iPad 14.1 (iPad6,11)"})
        self.printColor("### Got answer ###", "green")
        return json.loads(response.text)

    def scrape_dba(self):
        """
        Use for scraping DBA for Bang & Olufsen ads.
        
        Will filter out any ads not posted by private people.
        """
        current_call = self.make_api_call('Bang og Olufsen', 1)

        search_query = current_call['info']['title']
        hits = current_call['info']['hits']
        self.printColor("Got " + str(hits)
                        + " pages, using query "
                        + str(search_query)
                        + ", some will be discarded for being categorized wrong.", "green")

        content_num = len(current_call['content'])
        append_list = []

        i = 2
        while (content_num != 0):  # Keep adding to the pagination untill there is no more ads.
            for ad in current_call['content']:
                append_ad_brand = False  # I Only want ads that have Bang & Olufsen as manufacturer
                append_ad_model = False  # I only want ads that have specified the concerned model
                append_ad_seller = False  # I want private sellers only
                buffer_dict = {}
                if 'price' in ad['body']:
                    buffer_dict['price'] = ad['body']['price']
                if 'ad-owner' in ad['body']:
                    if 'ad-ownertype' in ad['body']['ad-owner']:
                        if ad['body']['ad-owner']['ad-ownertype'] == 1:
                            append_ad_seller = True
                        else:
                            continue
                if 'matrixdata' in ad['body']:
                    for value in ad['body']['matrixdata']:
                        if value['label'] == 'Mærke':
                            if value['value'].lower() in ['bang & olufsen', 'bang og olufsen', 'b&o', 'b & o']:
                                append_ad_brand = True
                        if value['label'] == 'Model':
                            buffer_dict['model'] = value['value'].lower()
                            append_ad_model = True
                        if value['label'] == 'Watt':
                            buffer_dict['watt'] = value['value'].lower()
                        if value['label'] == 'Stand':
                            buffer_dict['condition'] = value['value'].lower()

                if append_ad_brand and append_ad_model and append_ad_seller:
                    append_list.append(buffer_dict)

            current_call = self.make_api_call('Bang og Olufsen', i)
            content_num = len(current_call['content'])
            i += 1

        temp_df = pd.DataFrame(columns=['price', 'watt', 'condition', 'model'])
        temp_df = temp_df.append(append_list)
        for index, row in temp_df.iterrows():
            self.fuzzy_check(row)
        self.temp_df.to_json(os.path.join(sys.path[0], r'../data/latest_scrape_results.json'), orient='records', lines=True)

        dummified = pd.get_dummies(self.temp_df, columns=['condition', 'model'])
        self.final_df = self.final_df.append(dummified)
        self.final_df = self.final_df.fillna(0)
        self.final_df = self.final_df.drop_duplicates()
        self.final_df.to_json(os.path.join(sys.path[0], r'../data/scraped_dataframe_records_dummified.json'), orient='records')
        self.printColor("#### Scraping done ####", "blue")
