from pathlib import Path
from lxml import etree

XML_DIR = Path("datasets/eparchos/xml")

total_lines = 0
total_words = 0

for xml_file in XML_DIR.glob("*.xml"):
    tree = etree.parse(xml_file)

    lines = tree.xpath('//*[local-name()="TextLine"]')
    total_lines += len(lines)

    words = tree.xpath('//*[local-name()="Word"]')
    total_words += len(words)

print("Pages:", len(list(XML_DIR.glob("*.xml"))))
print("Lines:", total_lines)
print("Words:", total_words)