import scrapy
from idna import unicode
from scrapy import Item
from scrapy.loader import ItemLoader
from scrapy.loader.processors import MapCompose


class ExampleSpider(scrapy.Spider):
    name = 'example'
    allowed_domains = ['example.com']
    start_urls = ['http://example.com/']

    def parse(self, response):
        l = ItemLoader(item=Item(), response=response)

        # Load fields using XPath expressions
        l.add_xpath('title', '//*[@itemprop="name"][1]/text()',
                    MapCompose(unicode.strip, unicode.title))
        l.add_xpath('price', './/*[@itemprop="price"][1]/text()',MapCompose(lambda i: i.replace(',', ''),float),re = '[,.0-9]+')
        l.add_xpath('description', '//*[@itemprop="description"]'
        　　　　　　　　　　'[1]/text()',
        　　　　　　　　　　MapCompose(unicode.strip), Join())
        pass
