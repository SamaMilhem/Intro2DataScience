import json
import requests
from bs4 import BeautifulSoup

FRUITS_NAMES = [
    "Apple", "Banana", "Cherry", "Date", "Grape", "Orange_(fruit)", "Peach", "Pear", "Plum", "Watermelon",
    "Blueberry", "Strawberry", "Mango", "Kiwifruit", "Papaya", "Pineapple", "Lemon", "Lime_(fruit)",
    "Raspberry", "Blackberry"
]

FRUITS_LINKS = {fruit.split('_')[0]: f"https://en.wikipedia.org/wiki/{fruit}" for fruit in FRUITS_NAMES}


def fruitcrawl(fruit, link):
    """
       Crawl a Wikipedia page for a specific fruit, extract the main content text,
       and save the extracted information in a JSON file.

       Parameters:
       ----------
       fruit : str
           The name of the fruit being crawled (e.g., "Apple").

       link : str
           The URL of the Wikipedia page for the fruit."""


    request = requests.get(link)
    wikipedia_page = BeautifulSoup(request.text, "html.parser")
    text = ""
    for paragraph in wikipedia_page.find('main').find_all('p'):
        text += paragraph.get_text()

    with open('./Output_WikiPages/'+fruit+'.json', 'w', encoding='utf-8') as file:
       json.dump({fruit: text}, file, indent=4, ensure_ascii=False)


def linking_fruitscrawl():
    for key, value in FRUITS_LINKS.items():
        fruitcrawl(key, value)

if __name__ == '__main__':
    linking_fruitscrawl()