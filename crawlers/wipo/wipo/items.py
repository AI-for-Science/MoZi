# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class WipoItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    rule = scrapy.Field()
    link_text = scrapy.Field()
    depth = scrapy.Field()
    retry_times = scrapy.Field()
    download_timeout = scrapy.Field()
    download_slot = scrapy.Field()
    download_latency = scrapy.Field()

