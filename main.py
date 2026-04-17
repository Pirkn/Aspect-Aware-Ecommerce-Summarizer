from amazon_scrape import AmazonScraper

scraper = AmazonScraper()

input = input("Enter the ASIN or URL: ") 
df = scraper.scrape(input)
df.to_csv("output.csv", index=False)