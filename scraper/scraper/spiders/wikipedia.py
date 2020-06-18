# -*- coding: utf-8 -*-
import csv
import os
import scrapy
from datetime import datetime, timedelta
from pathlib import Path


class WikipediaSpider(scrapy.Spider):
    name = 'wikipedia'
    allowed_domains = ['wikipedia.org']
    start_urls = ['https://es.wikipedia.org/wiki/Pandemia_de_enfermedad_por_coronavirus_de_2020_en_Bolivia']

    month_mapping = {
        'enero': 1,
        'febrero': 2,
        'marzo': 3,
        'abril': 4,
        'mayo': 5,
        'junio': 6,
        'julio': 7,
        'agosto': 8,
        'septiembre': 9,
        'octubre': 10,
        'noviembre': 11,
        'diciembre': 12,
    }

    def parse(self, response):
        """Parse response."""
        data = {}
        for table in response.xpath('//table[@class="wikitable"]'):
            th = table.xpath('.//th')[0]
            title = th.xpath('text()').get().strip()
            if title == 'Fecha':
                for row in table.xpath('tbody/tr'):
                    cells = row.xpath('td')
                    if cells:
                        date = cells[0].xpath('.//text()').get().strip()
                        cases = cells[1].xpath('text()').get().strip()
                        deaths = cells[4].xpath('text()').get().strip()
                        recovers = cells[7].xpath('text()').get().strip()
                        date = self.parse_date(date)
                        cases = self.parse_numbers(cases)
                        deaths = self.parse_numbers(deaths)
                        recovers = self.parse_numbers(recovers)
                        data = self.update_data(data, date, cases, deaths, recovers)
        data = self.save_csv(data)

    def parse_date(self, date):
        """Parse date:
        10 de marzo -> 2020.05.10
        """
        year = datetime.now().year
        day, month = date.split(' de ')
        month = self.month_mapping[month.lower()]
        date = datetime.strptime(f'{year}-{month}-{day}', '%Y-%m-%d').date()
        return date

    def parse_numbers(self, numbers):
        """Parse numbers:
        (NUM1)CITY1, (NUM2)CITY2 -> {'CITY1': NUM1, 'CITY2: NUM2}
        """
        cities_dict = {}
        if numbers != '-':
            cities = numbers.split(',')
            for city in cities:
                if city.strip():
                    city = city.replace('(3LPZ', '(3)LPZ')  # temp fix for crappy data
                    num, city = city.strip().split(')')
                    num = num.strip()[1:]
                    if num == '!':
                        num = 1
                    num = int(num)
                    cities_dict[city.strip()] = num
        return cities_dict

    def update_data(self, data, date, cases, deaths, recovers):
        """Update data and return a dictionary:
        cities = {
            'LPZ': {
                'cases': [(2020.03.10, 1), (2020.03.10, 2)],
                'deaths': [(2020.03.10, 1), (2020.03.10, 2)],
                'recovers': [(2020.03.10, 1), (2020.03.10, 2)],
            },
            'SCZ': {
                ...
            },
            ...
        }
        """
        data = self.append_data(data, date, 'cases', cases)
        data = self.append_data(data, date, 'deaths', deaths)
        data = self.append_data(data, date, 'recovers', recovers)
        return data

    def append_data(self, data, date, category, d):
        """Append the individual number for the specified date.
        """
        for city, number in d.items():
            city = city.replace('.', '').strip()
            if city not in data:
                data[city] = {'cases': [], 'deaths': [],'recovers': []}
            data[city][category].append((date, number))
        return data

    def save_csv(self, data):
        """Save the csv files."""
        path = Path(__file__).parents[3]
        for city, categories in data.items():
            for category, _ in categories.items():
                last_date = None
                with open(os.path.join(path, f'csv/{city.lower()}-{category}.csv'), 'w') as file:
                    writer = csv.writer(file)
                    for date, num in data[city][category]:
                        if last_date is not None:
                            # in case there's a lapse between the last recorded date, fill the lapse with zeros
                            while (last_date + timedelta(days=1)) != date:
                                last_date = last_date + timedelta(days=1)
                                writer.writerow([last_date.strftime('%Y-%m-%d'), num])
                        writer.writerow([date.strftime('%Y-%m-%d'), num])
                        last_date = date
