"""
TRANSFORMER ARCHITECTURE - Building Blocks for Modern Sequence Processing.
This module defines a manual implementation of the Transformer encoder architecture.
It includes customized Multi-Head Attention mechanisms, Position-wise Feed-Forward networks, 
and the top-level Transformer Classifier for topic-based article categorization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelMultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention implementation using scaled dot-product.
    Allows the model to jointly attend to information from different 
    representation subspaces at different positions.
    """
    def __init__(self, ProjectionDimensionality, CountOfAttnHeads, ProbabilityOfDropout=0.1):
        super().__init__()
        assert ProjectionDimensionality % CountOfAttnHeads == 0, "Dimensions must be divisible by head count"
        
        self.D_Model = ProjectionDimensionality
        self.H_Heads = CountOfAttnHeads
        self.D_Head_Inner = ProjectionDimensionality // CountOfAttnHeads
        
        # Dense layers to project inputs into Query, Key, and Value spaces
        self.QueryProjectionLayer = nn.Linear(ProjectionDimensionality, ProjectionDimensionality)
        self.KeyProjectionLayer = nn.Linear(ProjectionDimensionality, ProjectionDimensionality)
        self.ValueProjectionLayer = nn.Linear(ProjectionDimensionality, ProjectionDimensionality)
        
        # Final output projection
        self.OutputUnifiedProjection = nn.Linear(ProjectionDimensionality, ProjectionDimensionality)
        
        self.AttentionDropoutLayer = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.InternalDropoutInstance = nn.Dropout(ProbabilityOfDropout)

    def CalculateAttentionScores(self, QueriesBatch, KeysBatch, ValuesBatch, SequenceMaskBatch=None):
        """Computes the Softmax(QK^T / sqrt(d)) * V attention output."""
        DotProductScores = torch.matmul(QueriesBatch, KeysBatch.transpose(-2, -1)) / math.sqrt(self.D_Head_Inner)
        
        if SequenceMaskBatch is not None:
            # Mask out padding positions using a very small negative value
            DotProductScores = DotProductScores.masked_fill(SequenceMaskBatch == 0, -1e10)
            
        SoftmaxProbabilities = F.softmax(DotProductScores, dim=-1)
        SoftmaxProbabilities = self.InternalDropoutInstance(SoftmaxProbabilities)
        
        WeightedSumOutputs = torch.matmul(SoftmaxProbabilities, ValuesBatch)
        return WeightedSumOutputs, SoftmaxProbabilities

    def forward(self, QueryInput, KeyInput, ValueInput, AttentionMask=None):
        BatchSizeVal = QueryInput.size(0)
        
        # 1. Project and Reshape into (Batch, Heads, SequenceLength, HeadDim)
        QueryStates = self.QueryProjectionLayer(QueryInput).view(BatchSizeVal, -1, self.H_Heads, self.D_Head_Inner).transpose(1, 2)
        KeyStates = self.KeyProjectionLayer(KeyInput).view(BatchSizeVal, -1, self.H_Heads, self.D_Head_Inner).transpose(1, 2)
        ValueStates = self.ValueProjectionLayer(ValueInput).view(BatchSizeVal, -1, self.H_Heads, self.D_Head_Inner).transpose(1, 2)
        
        # 2. Execute Scaled Dot-Product Attention
        if AttentionMask is not None:
            AttentionMask = AttentionMask.unsqueeze(1).unsqueeze(2) # Broadcast to heads and queries
            
        TransformedFeatures, _ = self.CalculateAttentionScores(QueryStates, KeyStates, ValueStates, AttentionMask)
        
        # 3. Concatenate heads back together and apply linear output projection
        ConcatenatedHeads = TransformedFeatures.transpose(1, 2).contiguous().view(BatchSizeVal, -1, self.D_Model)
        
        return self.OutputUnifiedProjection(ConcatenatedHeads)


class PointwiseFeedForwardNetwork(nn.Module):
    """
    Two-layer fully connected network applied to each position independently and identically.
    Features a ReLU non-linearity in the intermediate hidden layer.
    """
    def __init__(self, InputFeatureDim, IntermediateHiddenDim, DropoutRateVal=0.1):
        super().__init__()
        self.ExpandingLinearLayer = nn.Linear(InputFeatureDim, IntermediateHiddenDim)
        self.DropoutLayerAtCenter = nn.Dropout(DropoutRateVal)
        self.ContractingLinearLayer = nn.Linear(IntermediateHiddenDim, InputFeatureDim)

    def forward(self, InputFeatureBatch):
        IntermediateActivations = F.relu(self.ExpandingLinearLayer(InputFeatureBatch))
        return self.ContractingLinearLayer(self.DropoutLayerAtCenter(IntermediateActivations))


class TransformerEncoderBrick(nn.Module):
    """
    A single brick of the Transformer Encoder stack.
    Consists of: Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm.
    """
    def __init__(self, FeatureDimensionality, HeadQuantity, ExpansionRatio, DropoutRatio):
        super().__init__()
        self.ContextualAttentionHead = ParallelMultiHeadAttention(FeatureDimensionality, HeadQuantity, DropoutRatio)
        self.LayerNormAfterAttention = nn.LayerNorm(FeatureDimensionality)
        
        self.NonLinearProjector = PointwiseFeedForwardNetwork(FeatureDimensionality, FeatureDimensionality * ExpansionRatio, DropoutRatio)
        self.LayerNormAfterProjector = nn.LayerNorm(FeatureDimensionality)
        
        self.StochasticDropout = nn.Dropout(DropoutRatio)

    def forward(self, SequenceLayerInput, OptionalAttnMask=None):
        # Sub-layer 1: Attention and Residual Connection
        AttentionSubOutput = self.ContextualAttentionHead(SequenceLayerInput, SequenceLayerInput, SequenceLayerInput, OptionalAttnMask)
        NormalizedPostAttention = self.LayerNormAfterAttention(SequenceLayerInput + self.StochasticDropout(AttentionSubOutput))
        
        # Sub-layer 2: MLP and Residual Connection
        FeedForwardSubOutput = self.NonLinearProjector(NormalizedPostAttention)
        FinalBracketOutput = self.LayerNormAfterProjector(NormalizedPostAttention + self.StochasticDropout(FeedForwardSubOutput))
        
        return FinalBracketOutput


class NeuralTransformerCategorizationModel(nn.Module):
    """
    Full Transformer-based Classifier for text categorization.
    - Embeddings + Positional encodings (simple sinusoidal alternative).
    - Stack of N Encoder blocks.
    - Pooling through Global Average or CLS-style aggregation.
    - Linear projection to class logits.
    """
    def __init__(self, VocabularySpan, LatentEmbeddingDim, LayerStackDepth, HeadCountPerLayer, ExpansionFactor, MaxSeqLengthCapacity, TargetClassesCount, DropoutProp):
        super().__init__()
        self.TokenEmbeddingLayer = nn.Embedding(VocabularySpan, LatentEmbeddingDim)
        
        # Fixed Sinusoidal Positional Encoding (Self-calculated)
        self.register_buffer("AbsolutePositionalIndexMap", self._BuildSimpleSinusoidEncodings(MaxSeqLengthCapacity, LatentEmbeddingDim))
        
        # Construct the Encoder stack
        self.EncoderStackList = nn.ModuleList([
            TransformerEncoderBrick(LatentEmbeddingDim, HeadCountPerLayer, ExpansionFactor, DropoutProp) 
            for _ in range(LayerStackDepth)
        ])
        
        self.FinalDropoutFilter = nn.Dropout(DropoutProp)
        self.LogitsOutputPlane = nn.Linear(LatentEmbeddingDim, TargetClassesCount)

    def _BuildSimpleSinusoidEncodings(self, MaxLength, EmbedDim):
        """Generates the static sinusoidal positional bias."""
        PosEncodingBuffer = torch.zeros(MaxLength, EmbedDim)
        PosColumnIndices = torch.arange(0, MaxLength, dtype=torch.float).unsqueeze(1)
        DimScaleFactors = torch.exp(torch.arange(0, EmbedDim, 2).float() * (-math.log(10000.0) / EmbedDim))
        
        PosEncodingBuffer[:, 0::2] = torch.sin(PosColumnIndices * DimScaleFactors)
        PosEncodingBuffer[:, 1::2] = torch.cos(PosColumnIndices * DimScaleFactors)
        
        return PosEncodingBuffer.unsqueeze(0)

    def forward(self, TensorOfTokenIDs, TensorOfValidMasks):
        BatchSize, SeqLength = TensorOfTokenIDs.shape
        
        # 1. Coordinate Tokens and add Positional Induction
        DenseTokenRepresentations = self.TokenEmbeddingLayer(TensorOfTokenIDs)
        DenseTokenRepresentations = DenseTokenRepresentations + self.AbsolutePositionalIndexMap[:, :SeqLength, :]
        DenseTokenRepresentations = self.FinalDropoutFilter(DenseTokenRepresentations)
        
        # 2. Propagate through the Transformer blocks
        BlockActivations = DenseTokenRepresentations
        for EncoderBlockComponent in self.EncoderStackList:
            BlockActivations = EncoderBlockComponent(BlockActivations, TensorOfValidMasks)
            
        # 3. Aggregation Strategy: Average Pooling over non-masked tokens
        BatchFloatMask = TensorOfValidMasks.unsqueeze(-1).float()
        SummedTokenFeatures = (BlockActivations * BatchFloatMask).sum(dim=1)
        ValidLengthDenominator = BatchFloatMask.sum(dim=1).clamp(min=1e-12)
        PooledGlobalRepresentation = SummedTokenFeatures / ValidLengthDenominator
        
        # 4. Final classification decision
        FinalCategoryLogits = self.LogitsOutputPlane(self.FinalDropoutFilter(PooledGlobalRepresentation))
        
        return FinalCategoryLogits
