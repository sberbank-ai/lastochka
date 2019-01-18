# -*- coding: utf-8 -*-
# this is hack to import from pre-parent directory
import unittest
import docutils.nodes
import docutils.parsers.rst
import docutils.utils
import numpy as np


def parse_rst(text: str) -> docutils.nodes.document:
    parser = docutils.parsers.rst.Parser()
    components = (docutils.parsers.rst.Parser,)
    settings = docutils.frontend.OptionParser(components=components).get_default_values()
    document = docutils.utils.new_document('<rst-doc>', settings=settings)
    parser.parse(text, document)
    return document


class BaseTest(unittest.TestCase):
    """
    Basic functionality tests
    """

    def testReadMe(self):
        with open("./README.rst", "r", encoding="utf-8") as fhandler:
            _text = fhandler.read()

        parsed_doc = parse_rst(_text)
        good_part = parsed_doc[0][-1][-1].astext()
        exec(good_part)


if __name__ == "__main__":
    unittest.main()

