import unittest

from .test_object_decoder import CapsuleLayerTestCase, CapsuleLikelihoodTestCase, CapsuleObjectDecoderTestCase
from .test_part_decoder import TemplateBasedImageDecoderTestCase, TemplateGeneratorTestCase
from .test_part_encoder import CapsuleImageEncoderTestCase
from .test_scae import SCAETestCase
from .test_set_transformer import SetTransformerTestCase


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(CapsuleImageEncoderTestCase))
    test_suite.addTest(unittest.makeSuite(TemplateGeneratorTestCase))
    test_suite.addTest(unittest.makeSuite(TemplateBasedImageDecoderTestCase))
    test_suite.addTest(unittest.makeSuite(SetTransformerTestCase))
    test_suite.addTest(unittest.makeSuite(CapsuleLayerTestCase))
    test_suite.addTest(unittest.makeSuite(CapsuleLikelihoodTestCase))
    test_suite.addTest(unittest.makeSuite(CapsuleObjectDecoderTestCase))
    test_suite.addTest(unittest.makeSuite(SCAETestCase))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
