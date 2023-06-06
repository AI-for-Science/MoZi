import os
import scrapy
from wipo.items import WipoItem
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

DATA_DIR = 'data'

class WipoIntSpider(CrawlSpider):
    name = "wipo_int"
    allowed_domains = ["www.wipo.int"]

    main_site = "https://www.wipo.int/treaties"
    start_urls = [
        main_site,
    ]
    rules = (
        # Extract links matching 'category.php' (but not matching 'subsection.php')
        # and follow links from them (since no callback means follow=True by default).
        # Rule(LinkExtractor(allow=('category\.php',), deny=('subsection\.php',))),

        # # Extract links matching 'item.php' and parse them with the spider's method parse_item
        # Rule(LinkExtractor(allow=('chinese\/*\/*\.htm',)), callback='parse_item', follow=True),
        Rule(LinkExtractor(allow=('/treaties/.*',)), callback='parse_item', follow=True),
        Rule(LinkExtractor(allow=('\.pdf',)), callback='save_pdf', follow=True),
        # Rule(LinkExtractor(allow=('\d+\.index\.htm',)), callback='parse_item'),
    )

    def save_pdf(self, response):
        # path = response.url.split('/')[-1]
        self.logger.info('Hi, this is an pdf page! %s', response.url)
        pdf_file = os.path.join(DATA_DIR, response.url.replace(self.main_site, ''))
        self.logger.info('Saving PDF %s', pdf_file)
        os.makedirs(os.path.dirname(pdf_file), exist_ok=True)
        with open(pdf_file, 'wb') as f:
            f.write(response.body)

    def parse_item(self, response):
        self.logger.info('Hi, this is an item page! %s', response.url)
        html_file = DATA_DIR + response.url.replace(self.main_site, '')
        if not html_file.endswith('.html'):
            html_file = html_file + ".html"

        self.logger.info('Saving to dir %s', DATA_DIR)
        self.logger.info('Saving to html %s', html_file)
        content = response.body.decode()
        # try:
        #     content = response.body.decode("gb2312").replace('gb2312', 'utf8')
        # except UnicodeDecodeError:
        #     self.logger.info('Unicode error for {}'.format(response.url))
        #     content = response.body.decode('gb18030').replace('gb2312', 'utf8')

        os.makedirs(os.path.dirname(html_file), exist_ok=True)
        with open(html_file, 'w', encoding='utf8') as f:
            f.write(content)
        item = WipoItem()
        for k in response.meta:
            item[k] = response.meta[k]
        # item['id'] = response.xpath('//td[@id="item_id"]/text()').re(r'ID: (\d+)')
        # item['name'] = response.xpath('//td[@id="item_name"]/text()').get()
        # item['description'] = response.xpath('//td[@id="item_description"]/text()').get()
        # item['link_text'] = response.meta['link_text']

        return item

