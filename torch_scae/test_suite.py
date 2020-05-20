import unittest  # second test

from torch_scae.test_object_decoder import CapsuleLayerTestCase, CapsuleLikelihoodTestCase, CapsuleObjectDecoderTestCase
from torch_scae.test_part_decoder import TemplateBasedImageDecoderTestCase, TemplateGeneratorTestCase
from torch_scae.test_part_encoder import CapsuleImageEncoderTestCase
from torch_scae.test_scae import SCAETestCase
from torch_scae.test_set_transformer import SetTransformerTestCase


def suite():
    """
        Gather all the tests from this module in a test suite.
    """
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
    mySuit = suite()

    runner = unittest.TextTestRunner()
    runner.run(mySuit)
