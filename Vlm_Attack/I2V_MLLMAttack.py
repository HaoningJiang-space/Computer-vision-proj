from base_attacks import Attack
import torch
import torch.nn as nn
import numpy as np

class I2V_MLLMAttack(Attack):
    '''
    Algorithm 1: I2V-MLLM Attack
    Inputs:
        - V: Original video tensor (B, C, T, H, W)
        - T: Caption set (target text)
    Parameters:
        - Step size (alpha)
        - Iteration number (I)
        - Perturbation budget (epsilon)
        - Key-frame ratio (beta)
        - Loss function weights (lambda1, lambda2)
    Output:
        - V_adv: Adversarial video tensor (B, C, T, H, W)
    '''
    def __init__(self, model, processor, projector, epsilon=16/255, steps=10, alpha=0.01, beta=0.1, lambda1=1.0, lambda2=1.0):
        super(I2V_MLLMAttack, self).__init__("I2V_MLLMAttack", model)
        self.processor = processor
        self.projector = projector.to(self.device)  # 添加 Projector 模型
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.beta = beta  # key-frame ratio
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = next(model.parameters()).device

    def to(self, device):
        self.model.to(device)
        self.projector.to(device)
        return self

    def _select_key_frames(self, video):
        '''
        Split video into K clips based on key-frame ratio beta and extract first frame of each clip.
        video: Tensor (B, C, T, H, W)
        Returns: Tensor of key frames (B, C, K, H, W)
        '''
        B, C, T, H, W = video.shape
        K = max(1, int(T * self.beta))
        clip_length = T // K
        key_frames = []
        for k in range(K):
            start = k * clip_length
            key_frame = video[:, :, start, :, :]  # (B, C, H, W)
            key_frames.append(key_frame)
        key_frames = torch.stack(key_frames, dim=2)  # (B, C, K, H, W)
        return key_frames

    def _initialize_perturbations(self, key_frames):
        '''
        Initialize perturbations δ0 ∈ U(-epsilon, epsilon)
        key_frames: Tensor (B, C, K, H, W)
        Returns: perturbations δ (B, C, K, H, W)
        '''
        delta = torch.empty_like(key_frames).uniform_(-self.epsilon, self.epsilon)
        return delta

    def _get_visual_features(self, inputs):
        '''
        使用 BLIP2 模型的视觉编码器提取视觉特征。
        '''
        with torch.no_grad():
            # 使用 LAVIS 的 extract_features 方法
            visual_features = self.model.extract_features(inputs).multimodal_embeds  # 根据实际模型调整
        return visual_features  # (B, N_total, D1)

    def _extract_temporal_spatial_features(self, visual_features, N, K):
        '''
        提取时间和空间特征并进行平均池化
        visual_features: Tensor (B, N_total, D1)
        N: Number of patches per frame
        K: Number of key frames
        Returns:
            FtsV: Tensor (B, 2*D1)
        '''
        B, N_total, D1 = visual_features.shape
        assert N_total >= N + K, "N_total should be at least N + K"
        
        FtV = visual_features[:, :N, :]  # (B, N, D1)
        FsV = visual_features[:, N:N+K, :]  # (B, K, D1)
        
        # 平均池化沿时间维度
        FtV_pooled = FtV.mean(dim=1)  # (B, D1)
        
        # 平均池化沿空间维度
        FsV_pooled = FsV.mean(dim=1)  # (B, D1)
        
        # 拼接时间和空间特征
        FtsV = torch.cat([FtV_pooled, FsV_pooled], dim=1)  # (B, 2*D1)
        return FtsV

    def _compute_loss(self, original_features, adversarial_features, projector_original_features, projector_adversarial_features, textual_features=None):
        '''
        Compute total loss.
        Includes Vision Model loss and Projector loss.
        '''
        # Vision Model Loss (LV)
        loss_cos_vision = nn.CosineSimilarity(dim=1)
        loss_vision = 1 - loss_cos_vision(original_features, adversarial_features).mean()

        # Projector Loss (LP = LPv + LPv2t)
        loss_cos_projector_vision = nn.CosineSimilarity(dim=1)
        loss_LPv = 1 - loss_cos_projector_vision(projector_original_features, projector_adversarial_features).mean()

        if textual_features is not None:
            loss_cos_projector_multimodal = nn.CosineSimilarity(dim=2)
            # projector_adversarial_features: (B, N1, D2)
            # textual_features: (B, N2, D2)
            # 计算每个视觉特征与每个文本特征的余弦相似性
            cosine_sim_matrix = loss_cos_projector_multimodal(projector_adversarial_features.unsqueeze(2), textual_features.unsqueeze(1))  # (B, N1, N2)
            loss_LPv2t = 1 - cosine_sim_matrix.mean(dim=(1,2))  # (B)
            loss_LPv2t = loss_LPv2t.mean()  # Scalar
        else:
            loss_LPv2t = 0.0

        loss_projector = loss_LPv + loss_LPv2t

        # Total Loss (Ltotal = λ1 * LV + λ2 * LP)
        total_loss = self.lambda1 * loss_vision + self.lambda2 * loss_projector
        return total_loss

    def forward(self, video, target_text=None, text_features=None):
        '''
        Generate adversarial video
        video: Tensor (B, C, T, H, W)
        target_text: str (optional) - Not used in this simplified version
        text_features: Tensor (B, M, D2) (optional) - Precomputed text features
        Returns: adversarial_video (B, C, T, H, W)
        '''
        B, C, T, H, W = video.shape

        # 1. Key-frame Selection
        key_frames = self._select_key_frames(video)  # (B, C, K, H, W)
        B, C, K, H, W = key_frames.shape  # Update K based on selection
        delta = self._initialize_perturbations(key_frames).to(self.device).requires_grad_(True)  # (B, C, K, H, W)

        # 2. 获取原始视频特征
        # 对关键帧进行处理
        key_frames_flat = key_frames.flatten(start_dim=2).permute(0, 2, 1, 3, 4)  # (B, K, C, H, W)
        key_frames_flat = key_frames_flat.view(B * K, C, H, W)  # (B*K, C, H, W)
        inputs_video = {"image": key_frames_flat, "text_input": ["Describe the video."] * (B * K)}  # 简化示例
        original_visual_features = self._get_visual_features(inputs_video)  # (B*K, N_total, D1)
        N = original_visual_features.shape[1]
        FtsV_original = self._extract_temporal_spatial_features(original_visual_features, N, K)  # (B*K, 2*D1)
        FtsV_original = FtsV_original.view(B, K, -1).mean(dim=1)  # (B, 2*D1)

        # Projector Attack: Extract Projector Original Features
        projector_original_features = self.projector(FtsV_original)  # (B, N1, D2)

        # 3. Perturbation Optimization
        for step in range(self.steps):
            self.model.zero_grad()
            self.projector.zero_grad()
            
            # Apply perturbations to key frames
            adversarial_key_frames = torch.clamp(key_frames + delta, 0, 1)  # (B, C, K, H, W)

            # Prepare adversarial key frames for feature extraction
            adversarial_key_frames_flat = adversarial_key_frames.flatten(start_dim=2).permute(0, 2, 1, 3, 4)  # (B, K, C, H, W)
            adversarial_key_frames_flat = adversarial_key_frames_flat.view(B * K, C, H, W)  # (B*K, C, H, W)
            inputs_adv = {"image": adversarial_key_frames_flat, "text_input": ["Describe the video."] * (B * K)}  # 简化示例
            adversarial_visual_features = self._get_visual_features(inputs_adv)  # (B*K, N_total, D1)
            FtsV_adversarial = self._extract_temporal_spatial_features(adversarial_visual_features, N, K)  # (B*K, 2*D1)
            FtsV_adversarial = FtsV_adversarial.view(B, K, -1).mean(dim=1)  # (B, 2*D1)

            # Projector Attack: Extract Projector Adversarial Features
            projector_adversarial_features = self.projector(FtsV_adversarial)  # (B, N1, D2)

            # Compute loss
            loss = self._compute_loss(FtsV_original, FtsV_adversarial, projector_original_features, projector_adversarial_features, text_features)
            
            # Backward
            loss.backward()

            # Update perturbations
            with torch.no_grad():
                delta += self.alpha * delta.grad.sign()
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                # Ensure adversarial_key_frames within [0,1]
                adversarial_key_frames = torch.clamp(key_frames + delta, 0, 1)
                delta = (adversarial_key_frames - key_frames).detach().requires_grad_(True)

            print(f"步骤 {step+1}/{self.steps}，损失: {loss.item()}")

        # 4. Perturbation Propagation
        adversarial_video = self._propagate_perturbations(video, torch.clamp(key_frames + delta, 0, 1))

        # 5. Construct the adversarial video
        return adversarial_video

    def _propagate_perturbations(self, original_video, adversarial_key_frames):
        '''
        Propagate perturbations from key frames back to video
        original_video: Tensor (B, C, T, H, W)
        adversarial_key_frames: Tensor (B, C, K, H, W)
        Returns: adversarial_video (B, C, T, H, W)
        '''
        B, C, T, H, W = original_video.shape
        _, _, K, _, _ = adversarial_key_frames.shape
        clip_length = T // K
        adversarial_video = original_video.clone()

        for k in range(K):
            start = k * clip_length
            end = start + clip_length
            if k == K - 1:
                # 处理最后一个片段，可能长度不同
                end = T
            adversarial_video[:, :, start:end, :, :] = adversarial_key_frames[:, :, k, :, :].unsqueeze(2).repeat(1, 1, end - start, 1, 1)

        return adversarial_video