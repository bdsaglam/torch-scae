import unittest

import torch

from torch_scae.set_transformer import qkv_attention, MultiHeadQKVAttention,\
    MAB, SAB, ISAB, PMA, SetTransformer


class SetTransformerTestCase(unittest.TestCase):
    def test_qkv_attention(self):
        B = 32
        d_k = 16
        d_v = 32
        N = 10
        M = 10

        q = torch.rand(B, N, d_k)
        k = torch.rand(B, M, d_k)
        v = torch.rand(B, M, d_v)
        presence = torch.rand(B, M)
        out = qkv_attention(q, k, v, presence)

        self.assertTrue(out.shape == (B, N, d_v))

    def test_multi_head_qkv_attention(self):
        B = 32
        d_k = 16
        d_v = 32
        N = 10
        M = 10
        n_heads = 3

        q = torch.rand(B, N, d_k)
        k = torch.rand(B, M, d_k)
        v = torch.rand(B, M, d_v)
        presence = torch.rand(B, M)

        with torch.no_grad():
            mhqkv = MultiHeadQKVAttention(d_k=d_k, d_v=d_v, n_heads=n_heads)
            out = mhqkv(q, k, v, presence)

        self.assertTrue(out.shape == (B, N, d_v))

    def test_mab(self):
        B = 32
        d = 16
        N = 10
        M = 10
        n_heads = 3

        q = torch.rand(B, N, d)
        k = torch.rand(B, M, d)
        presence = torch.rand(B, M)

        with torch.no_grad():
            mab = MAB(d=d, n_heads=n_heads, layer_norm=False)
            out = mab(q, k, presence)

        self.assertTrue(out.shape == (B, N, d))

    def test_mab_with_layer_normalization(self):
        B = 32
        d = 16
        N = 10
        M = 10
        n_heads = 3

        q = torch.rand(B, N, d)
        k = torch.rand(B, M, d)
        presence = torch.rand(B, M)

        with torch.no_grad():
            mab = MAB(d=d, n_heads=n_heads, layer_norm=True)
            out = mab(q, k, presence)

        self.assertTrue(out.shape == (B, N, d))

    def test_sab(self):
        B = 32
        d = 16
        N = 10
        n_heads = 3

        x = torch.rand(B, N, d)
        presence = torch.rand(B, N)

        with torch.no_grad():
            sab = SAB(d=d, n_heads=n_heads, layer_norm=True)
            out = sab(x, presence)

        self.assertTrue(out.shape == (B, N, d))

    def test_isab(self):
        B = 32
        d = 16
        N = 10
        n_heads = 3
        n_inducing_points = 5

        x = torch.rand(B, N, d)
        presence = None

        with torch.no_grad():
            isab = ISAB(d=d, n_heads=n_heads, n_inducing_points=n_inducing_points,
                        layer_norm=True)
            out = isab(x, presence)

        self.assertTrue(out.shape == (B, N, d))

    def test_pma(self):
        B = 32
        d = 16
        N = 10
        n_heads = 3
        n_seeds = 10

        x = torch.rand(B, N, d)
        presence = torch.rand(B, N)

        with torch.no_grad():
            pma = PMA(d=d, n_heads=n_heads, n_seeds=n_seeds, layer_norm=True)
            out = pma(x, presence)

        self.assertTrue(out.shape == (B, N, d))

    def test_set_transformer_with_sab(self):
        B = 32
        N = 40

        n_heads = 1
        dim_in = 16
        dim_hidden = 16
        dim_out = 256
        n_outputs = 10
        n_layers = 3

        x = torch.rand(B, N, dim_in)
        presence = None

        with torch.no_grad():
            st = SetTransformer(
                dim_in,
                dim_hidden,
                dim_out,
                n_outputs,
                n_layers,
                n_heads,
                layer_norm=True,
            )
            out = st(x, presence)

        self.assertTrue(out.shape == (B, n_outputs, dim_out))

    def test_set_transformer_with_isab(self):
        B = 32
        N = 40

        n_heads = 1
        dim_in = 16
        dim_hidden = 16
        dim_out = 256
        n_outputs = 10
        n_layers = 3

        x = torch.rand(B, N, dim_in)
        presence = None

        with torch.no_grad():
            st = SetTransformer(
                dim_in,
                dim_hidden,
                dim_out,
                n_outputs,
                n_layers,
                n_heads,
                layer_norm=True,
                n_inducing_points=20
            )
            out = st(x, presence)

        self.assertTrue(out.shape == (B, n_outputs, dim_out))


if __name__ == '__main__':
    unittest.main()
