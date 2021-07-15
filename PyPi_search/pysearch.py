#!/usr/bin/env python

import sys
import requests
from bs4 import BeautifulSoup

class Query:
    '''
    Models a query to pypi website
    '''

    # the base URL of pypi
    URL = "https://pypi.org"

    def __init__(self, query):
        '''
        query must be a 
        tokenized list of queries
        '''
        
        # save the query
        self.pypi_query = self._querify(query)
        # retrieve the website
        try:
            self.website  = requests.get(self.pypi_query)
            # parse the webpage
            self.soup = BeautifulSoup(self.website.content, 'html.parser')
        except requests.ConnectTimeout:
            print("Connection attempt timed out. Try later")
            sys.exit(1)
        except requests.ConnectionError:
            print("Page unreachable. Check you Internet connection")
            sys.exit(1)

    def _querify(self, query):
        '''
        transforms a query
        into a valid format
        for pypi
        '''
        joined_query = '%20'.join(query)
        pypi_query = self.URL + "/search/?q=" + joined_query

        return pypi_query

    def parser(self):
        '''
        parse the elements
        of the webpage
        '''

        # parse package name, version and description
        package_names = self.soup.find_all(class_ = "package-snippet__name")
        package_versions = self.soup.find_all(class_ = "package-snippet__version")
        package_released = self.soup.find_all(class_ = "package-snippet__released")
        package_descriptions = self.soup.find_all(class_ = "package-snippet__description")

        # build the elements container
        # if there are results
        if package_names:
            self.container = self._gather_elements(package_names, package_versions, package_released, package_descriptions)
        else:
            self.container = []

    def packages(self, max_packages=None):
        '''
        print the results of the query
        show up to max_packages
        '''

        if not max_packages:
            max_packages = len(self.container)
        try:
            # if there are results
            if self.container:
                print("")
                for count, package in enumerate(self.container):
                    # break if reached count
                    if count == max_packages:
                        break

                    name = package["name"]
                    version = package["version"]
                    released = package["released"]
                    description = package["description"]
                    self.prettify(name, version, released, description)
                print("")
            else:
                print("\n\tNothing! ", "¯\_( ͡° ͜ʖ ͡°)_/¯", "\n")

        except NameError:
            print("You need to parse the webpage first!")

    @staticmethod
    def _gather_elements(package_names, package_versions, package_released, package_descriptions):
        '''
        Helper method for parser
        Creates a list
        of dictionaries that contain the elements
        of each package
        '''

        collection = []
        for package in range(len(package_names)):
            name = package_names[package].text
            # shorten name if too long
            name = Query._cut_name(name, 25)
            version = package_versions[package].text
            released = package_released[package].text.strip()
            description = package_descriptions[package].text
            # shorten description if too long
            description = Query._cut_description(description, 70)
            
            package = {"name":name,
                    "version":version,
                    "released": released,
                    "description":description,
                    }

            collection.append(package)

        return collection

    @staticmethod
    def _cut_name(name, length):
        '''
        cut name
        '''
        if len(name) > length:
            name = name[:length] + "..."
        return name

    @staticmethod
    def _cut_description(description, length):
        '''
        cut description
        '''
        if len(description) > length:
            description = description[:length] + "..."
        return description

    @staticmethod
    def prettify(name, version, released, description):
        '''
        print package details
        in nice format
        '''

        name_ver = name+" "+version
        print(f"{name_ver:50}\t({released})\t{description}")


if __name__ == "__main__":
    import argparse
    description = "Search a python package using PyPi search engine"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("query", help="search term to submit to PyPi search engine", nargs="*")
    parser.add_argument("-n", help="show the first n packages", type=int)
    args = parser.parse_args()
    query = args.query

    # create class instance
    results = Query(query)
    # parse the results
    results.parser()
    # show the packages
    if args.n:
        results.packages(args.n)
    else:
        results.packages()
