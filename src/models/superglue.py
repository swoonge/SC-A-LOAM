# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn


# ex) channels = [16, 64, 32, 10] -> MLP의 nn.Sequentail로 return, do_bn==True라면 BatchNorm 진행
def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1): # 마지막 layer가 아니면서 do_bn이 True라면 BatchNorm layer추가.
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

# 적당한 값으로 scaling factor를 정하고, centor로 중심을 0으로 보냄
# 여기서는 size.max(1, keepdim=True).values * 0.7 >> 이미지의 제일 큰 방향의 0.7 정도를 scaling factor로 함
# 따라서 큰 축이 640, 480 짜리 이미지라면, 큰축의 픽셀 위치를 640/2, 작은축의 픽셀들 위치를 480/2 만큼 빼주어 [-320~320, -240~240]의 범위로 바꾼다.
# 그 뒤, 모든 픽셀에 대해 640 * 0.7 = 448의 값으로 나누어 큰 축은 [-0.714~0.714], 작은 축은 [-0.536~0.536]의 범위로 바꾼다.
def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    height, width = image_shape
    # _,_,height, width = image_shape
    one = kpts.new_tensor(1) # kpts와 같은 데이터 타입으로 1값을 가진 scala value tensor생성
    size = torch.stack([one*width, one*height])[None] ## [None]: 차원 추가
    center = size / 2 # 중앙 픽셀값
    scaling = size.max(1, keepdim=True).values * 0.7 # >> scaling == 640 * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :] 


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        # inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(0).unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5 # 쿼리와 키 사이의 dot product(q_i^T * k_j), 쿼리의 차원의 제곱근으로 나누어져 스케일링
    prob = torch.nn.functional.softmax(scores, dim=-1) # prob == attentional score alpha.
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob # \alpha_{i,j} * v_j 연산을 의미.

# AttentionalPropagation 모듈의 attention부분. 이 모듈의 output이 MLP를 통과하면 Attention 모듈 하나 완료
class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0 # d_model(== 256)은 num_heads(== 4)로 나누어 떨어져야 하므로, 이를 확인하는 assert 문이 포함
        self.dim = d_model // num_heads # Multihead div: 각 head의 차원 수를 계산하여 self.dim에 저장 (self.dim = 64)
        self.num_heads = num_heads   
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1) # self.merge -> 어텐션의 K, Q, V 의 weight 역할       
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)]) # self.merge, 즉 Weight를 K, Q, V에 대해 3개씩 세트를 지어 만듦.(각각 W_k, W_q, W_v)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0) # batch_dim을 가져옴. demo에서는 항상 1이었지만, train에서는 batchsize가 됨.
        # self.proj에 정의된 각 Conv1d 레이어를 query, key, value에 적용하고, 그 결과를 적절한 형태로 변형합니다. 이 과정을 통해 각 입력은 (배치 크기, head 당 차원 수, head 수, descriptor수)의 형태를 가지게 됩니다.
        # for l, x in zip(self.proj, (query, key, value)) -> (W_k, W_q, W_v)세트의 l, (query, key, value)의 x에 대해 (l(x))의 연산을 각각 수행
        # l = nn.ModuleList의 각각의 nn.Conv1d이므로, l(x)하면 각각 q_i = W_q * x^Q + b_q 식을 통과하게 됨
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1) for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value) # x -> attention
        # contiguous() 함수는 이러한 텐서를 새로운 텐서에 복사하여 요소가 메모리에서 연속적으로 배치되도록 합니다.
        # view(batch_dim, self.dim*self.num_heads, -1) -> muiltihead로 나뉘었던 head를 다시 묶음.
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1)) # \alpha_{i,j} * v_j 연산에서 나온 attention에 1dConv를 한번 더 해줌

# 이거 하나당 attention 모듈 하나를 의미. AttentionalGNN은 AttentionalPropagation 18개가 쌓임.
class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        # self.attn의 input은 query, key, value이다.
        #  x, source, cource의 경우, cross, self attention에 대한 차이.
        message = self.attn(x, source, source) 
        return self.mlp(torch.cat([x, message], dim=1)) # MLP([x_i^A || m_{\epsilon -> i}])에 해당하는 연산 수행

# 'superglue': {'weights': 'indoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2, }
# default_config = {'descriptor_dim': 256, 'weights': 'indoor', 'keypoint_encoder': [32, 64, 128, 256], 
#                   'GNN_layers': ['self', 'cross'] * 9, 'sinkhorn_iterations': 100, 'match_threshold': 0.2,}
class AttentionalGNN(nn.Module): # feature_dim == 256, layer_name == ['self', 'cross'] * 9
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))]) # 총 18개의 AttentionalPropagation을 ModuleList로 묶음. AttentionalPropagation(dim, num_heads)
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            # desc1, desc0 는 노드로 보면 된다. 여기서는 edge가 self또는 cross로 모든 노드에 대해 연결되어 있으니 edge에 대한 정보는 따로 더 필요가 없다.
            # 즉, 아래 if문이 엣지의 모든 경우의 수를 나눈 것이라 볼 수 있다.
            # 이름이 'cross'인 경우 입력 텐서를 교차시키고, 그렇지 않은 경우 ('self') 입력 텐서를 그대로 사용
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1) # layer의 input으로 desc0와 src0(cross, self인지에 따라 달라짐), output인 delta는 gnn의 MLP통과 단 까지를 의미.
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1) # 따라서 l+1번째 x를 만들기 위해 l번째 x와 MLP결과를 더함.
        return desc0, desc1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        # KeypointENC >> input: 3ch(x, y, c), output: 256ch descriptor
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        self.load_state_dict(torch.load(str(path)))
        print('Loaded SuperGlue model (\"{}\" weights)'.format(
            self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                # A tensor of the same shape as kpts0 and kpts1, filled with -1. The -1 value is often used to indicate a lack of match or an invalid index in PyTorch.
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                # A tensor of the same shape as kpts0 and kpts1, filled with 0. This indicates that there are no matching scores because there are no keypoints.
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        # input boundary -> if image size is [480, 640], output is [[-0.536~0.536],[-0.714~0.714]]
        kpts0 = normalize_keypoints(kpts0, data['shape0'])
        kpts1 = normalize_keypoints(kpts1, data['shape1'])

        # Keypoint MLP encoder( = positional encoder ). formula: x_i = d_i + MLP_enc(p_i) // kpts0, data['scores0'] == x,y,c
        # input: x_i = d_i(desc0) + MLP_enc(p_i(kpts0, data['scores0']))
        # print(self.kenc(kpts0, data['scores0']).shape, desc0.shape)
        desc0 = desc0 + self.kenc(kpts0, data['scores0']) # desc0.shape >> [1, 256, The number of keypoints in image 0] -> transformer의 특징 중 하나이 positional encoding을 한 것과 비슷한 효과
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        # output: 식 3번을 L번까지 완료한 (L)x_i^A, B들
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection. mdesc == matching descriptors f_i^A, ^B
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        # 여기서 'bdn'과 'bdm'은 각각 mdesc0와 mdesc1의 차원을, 'bnm'은 결과 텐서의 차원을 나타냄. 이 연산은 mdesc0의 마지막 차원과 mdesc1의 두 번째 차원을 곱하고, 나머지 차원에 대해 합계를 계산
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5 # norm

        # Run the optimal transport. -> output == partial assignment
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # scores^2을 하면 대충 partial assignment가 되는 듯
        # Get the matches with score above "match_threshold".
        # 각 키포인트에 대해 가장 높은 점수를 가진 매칭을 찾습니다. 
        # max0은 첫 번째 이미지의 각 키포인트에 대해 가장 높은 점수를 가진 매칭을 찾고, 
        # max1은 두 번째 이미지의 각 키포인트에 대해 가장 높은 점수를 가진 매칭을 찾습니다.
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'scores' : scores,
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }
