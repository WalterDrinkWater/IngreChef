import requests
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
import time
import config

headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}

class webScraper():
    def __init__(self, url):
        self.url = url 
        self.soup = BeautifulSoup(requests.get(url, headers=headers).content, 'html.parser')
    
    def recipe_name(self):
        try:
            return self.soup.find('h1').text.strip()
        except: 
            return np.nan
        
    def serves(self):
        try:
            if(self.url.startswith("https://www.jamieoliver.com")):
                return self.soup.find('div', {'class': 'recipe-detail serves'}).text.split(' ',1)[1]
            else:
                test = self.soup.find_all('div', {'class': 'mntl-recipe-details__value'})
                serve = str(test[len(test)-1])
                sub1 = '">'
                sub2 = ' </div>'
                # getting index of substrings
                serve = serve.replace(sub1,"*")
                serve = serve.replace(sub2,'*')
                return serve.split("*")[1]
        except:
            return np.nan 

    def cooking_time(self):
        try:
            if(self.url.startswith("https://www.jamieoliver.com")):
                return self.soup.find('div', {'class': 'recipe-detail time'}).text.split('In')[1]
            else:
                test = self.soup.find_all('div', {'class': 'mntl-recipe-details__value'})
                serve = str(test[len(test)-2])
                sub1 = '">'
                sub2 = '</div>'
                # getting index of substrings
                serve = serve.replace(sub1,"*")
                serve = serve.replace(sub2,'*')               
                return serve.split("*")[1].strip()
        except:
            return np.nan

    def ingredients(self):
        try:
            ingredients = [] 
            if(self.url.startswith("https://www.jamieoliver.com")):
                for li in self.soup.select('.ingred-list li'):
                    ingred = ' '.join(li.text.split())
                    ingredients.append(ingred)
            else:
                for li in self.soup.select('.mntl-structured-ingredients__list li'):
                    ingred = ' '.join(li.text.split())
                    ingredients.append(ingred)
                    
            return ingredients
        except:
            return np.nan
    
    def steps(self):
        try:
            steps = [] 
            if(self.url.startswith("https://www.jamieoliver.com")):
                for li in self.soup.select('.recipeSteps li'):
                    step = ' '.join(li.text.split())
                    steps.append(step)
            else:
                for li in self.soup.select('#mntl-sc-block_2-0 li'):
                    step = ' '.join(li.text.split())
                    steps.append(step)
            return steps
        except:
            return np.nan
        
    def img(self):
        try:
            if(self.url.startswith("https://www.jamieoliver.com")):
                img_ele = self.soup.find('div', {'class': 'hero-wrapper'}).find("img")
            else:
                img_ele = self.soup.find('div', {'class': 'img-placeholder'}).find("img")
            return img_ele['src']

        except:
            return np.nan


if __name__ == "__main__":
    recipe_df = pd.read_csv(config.RECIPE_URLS)["recipe_urls"][:3]
    attribs = ['recipe_name', 'serves', 'cooking_time', 'ingredients','steps','img']

    temp = pd.DataFrame(columns=attribs)
    for i in range(0,len(recipe_df)):
        try:
            url = recipe_df[i]
            recipe_scraper = webScraper(recipe_df[i])
            temp.loc[i] = [getattr(recipe_scraper, attrib)() for attrib in attribs]
            print(f'Step {i} completed')
            time.sleep(np.random.randint(1,3))
        except:
                print(f'Step {i} failed')

    temp['recipe_urls'] = recipe_df
    columns = ['recipe_urls'] + attribs
    temp = temp[columns]

    df = temp
    df.to_csv(rf"{config.RECIPES_DETAILS}", index=False)