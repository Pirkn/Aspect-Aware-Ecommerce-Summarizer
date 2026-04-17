from amazon_scrape import AmazonScraper

scraper = AmazonScraper()

input = input("Enter the ASIN: ") 
scraper.scrape_product(input)