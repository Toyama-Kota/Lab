import torch
import torch.nn as nn

# ◆ ---------- オリジナル ---------- ◆
from embedding import Embedding
from linear_projection import LinearProjection
from mlp_head import MLPHead
from patching import Patching
from transformer_encoder import TransformerEncoder

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, class_num, vec_length, encoder_layer_num, head_num, channels = 3, mlp_vec_length = 256):
        """ [input]
            - image_size (int) : 画像の縦の長さ（= 横の長さ）
            - patch_size (int) : パッチの縦の長さ（= 横の長さ）
            - class_num (int) : 分類するクラスの数
            - vec_length (int) : 各パッチのベクトルが変換されたベクトルの長さ（参考[1] (1)式 D）
            - encoder_layer_num (int) : Transformer Encoder の層の深さ（参考[1] (2)式 L）
            - head_num (int) : Multi-Head Attention の head の数
            - chahnnels (int) : 入力のチャネル数（RGBの画像なら3）
            - mlp_vec_length (int) : MLP の隠れ層のノード数
        """

        super().__init__()
        
        # Params
        patche_num = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size
        self.encoder_layer_num = encoder_layer_num

        # Layers
        self.patching = Patching(patch_size = patch_size)
        self.linear_projection_of_flattened_patches = LinearProjection(patch_dim = patch_dim, vec_length = vec_length)
        self.embedding = Embedding(vec_length = vec_length, patche_num = patche_num)
        self.transformer_encoder = TransformerEncoder(vec_length = vec_length, n_heads = head_num, mlp_dim = mlp_vec_length, depth = encoder_layer_num)
        self.mlp_head = MLPHead(vec_length = vec_length, out_dim = class_num)


    def forward(self, img):
        """ [input]
            - img (torch.Tensor) : 画像データ
                - img.shape = torch.Size([batch_size, channels, image_height, image_width])
        """

        # 入力画像
        x = img

        # パッチに分割する
        # x.shape : [batch_size, channels, image_height, image_width] -> [batch_size, patche_num, channels * (patch_size ** 2)]
        x = self.patching(x)

        # 各パッチをベクトルに変換
        # x.shape : [batch_size, patche_num, channels * (patch_size ** 2)] -> [batch_size, patche_num, vec_length]
        x = self.linear_projection_of_flattened_patches(x)

        # [class] トークン付加 + 位置エンコーディング 
        # x.shape : [batch_size, patche_num, vec_length] -> [batch_size, patche_num + 1, vec_length]
        x = self.embedding(x)

        # Encoder Block
        # x.shape : [batch_size, patche_num + 1, vec_length] -> [batch_size, patche_num + 1, vec_length]
        x = self.transformer_encoder(x)

        # 出力の0番目のベクトルを MLP Head で処理
        # x.shape : [batch_size, patche_num + 1, vec_length] -> [batch_size, vec_length] -> [batch_size, class_num]
        x = x[:, 0]
        x = self.mlp_head(x)

        return x