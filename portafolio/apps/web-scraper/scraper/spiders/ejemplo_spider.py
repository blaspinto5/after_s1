import scrapy

class EjemploSpider(scrapy.Spider):
    name = "ejemplo"
    start_urls = ["https://quotes.toscrape.com/js"]

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                meta={"playwright": True},  # <--- activar Playwright
                callback=self.parse
            )

    def parse(self, response):
        for quote in response.css("div.quote"):
            yield {
                "texto": quote.css("span.text::text").get(),
                "autor": quote.css("small.author::text").get(),
            }
