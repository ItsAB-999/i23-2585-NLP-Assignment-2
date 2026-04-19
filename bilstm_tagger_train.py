"""
BILSTM TAGGER TRAINER - Training Sequence Labelers for POS and NER.
This module implements a Bi-directional Long Short-Term Memory (BiLSTM) network 
for Part-of-Speech tagging and a BiLSTM+CRF (Conditional Random Field) architecture 
for Named Entity Recognition. It supports both frozen and finetuned word embeddings.
"""

import os
import sys
import json
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


def EstablishStandardOutputEncoding():
    """Forces standard output to UTF-8 to ensure Urdu text prints correctly."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def LoadSentencesFromConllFormat(FilePathToLoad):
    """
    Parses a CoNLL-formatted file and extracts word/tag sequences.
    Sentences are separated by blank lines.
    """
    CollectionOfSentencesData = []
    ActiveWordBuffer = []
    ActiveTagBuffer = []
    
    try:
        with open(FilePathToLoad, mode="r", encoding="utf-8") as InputDataStream:
            for CurrentRawLine in InputDataStream:
                CleanedLineContent = CurrentRawLine.rstrip("\n")
                if not CleanedLineContent.strip():
                    if ActiveWordBuffer:
                        CollectionOfSentencesData.append((ActiveWordBuffer, ActiveTagBuffer))
                        ActiveWordBuffer = []
                        ActiveTagBuffer = []
                    continue
                TabSeparatedParts = CleanedLineContent.split("\t")
                if len(TabSeparatedParts) >= 2:
                    ActiveWordBuffer.append(TabSeparatedParts[0])
                    ActiveTagBuffer.append(TabSeparatedParts[1])
                    
        if ActiveWordBuffer:
            CollectionOfSentencesData.append((ActiveWordBuffer, ActiveTagBuffer))
    except Exception as IOErrorWhileLoading:
        print(f"FAILED TO LOAD CoNLL FILE: {IOErrorWhileLoading}")
        
    return CollectionOfSentencesData


def AdaptWordToIndexForPadding(OriginalWordToIndexMap):
    """
    Adjusts the word-to-index mapping to reserve index 0 for padding.
    Offsets all original indices by +1.
    """
    ShiftedWordToIndexMap = {}
    for SurfaceLexeme, OldIndexID in OriginalWordToIndexMap.items():
        ShiftedWordToIndexMap[SurfaceLexeme] = int(OldIndexID) + 1
        
    ShiftedUnknownWordIndex = int(OriginalWordToIndexMap["<UNK>"]) + 1
    TotalAdjustedVocabSize = len(OriginalWordToIndexMap) + 1
    
    return ShiftedWordToIndexMap, ShiftedUnknownWordIndex, TotalAdjustedVocabSize


def PrepareEmbeddingWeightsWithPadRow(RawW2vMatrixData, TargetVocabPlusOne, DimensionalityValue):
    """
    Converts the NumPy embedding matrix into a PyTorch tensor.
    Prepends a zero-vector at index 0 for the <PAD> token.
    """
    MasterWeightsBuffer = np.zeros((TargetVocabPlusOne, DimensionalityValue), dtype=np.float32)
    MasterWeightsBuffer[1:] = RawW2vMatrixData.astype(np.float32)
    return torch.from_numpy(MasterWeightsBuffer)


def CreateTagIndexDictionaries(SentencesCollection):
    """Generates mapping for the output tag alphabet found in the training data."""
    UniqueSetOfTags = sorted({SingleTag for _, TagSequence in SentencesCollection for SingleTag in TagSequence})
    TagToNumericIDMap = {StringTag: NumericIdx for NumericIdx, StringTag in enumerate(UniqueSetOfTags)}
    return TagToNumericIDMap, UniqueSetOfTags


def PartitionTrainingAndValidationSets(DataSentences, ValidationBucketFraction, RandomStateSeed):
    """Randomly splits the dataset into Training and Validation partitions."""
    InternalRng = random.Random(RandomStateSeed)
    MasterIndicesList = list(range(len(DataSentences)))
    InternalRng.shuffle(MasterIndicesList)
    
    ValidationQuantity = max(1, int(round(len(MasterIndicesList) * ValidationBucketFraction)))
    ValidationIndicesSet = set(MasterIndicesList[:ValidationQuantity])
    
    TrainingSplit = [DataSentences[i] for i in range(len(DataSentences)) if i not in ValidationIndicesSet]
    ValidationSplit = [DataSentences[i] for i in range(len(DataSentences)) if i in ValidationIndicesSet]
    
    return TrainingSplit, ValidationSplit


class NeuralSentenceLabelingDataset(Dataset):
    """PyTorch Dataset wrapper for sequence labeling data."""
    def __init__(self, ListOfSents, WordToIDMap, UnknownWordID, TagToIDMap):
        self.DatasetStorage = ListOfSents
        self.W2ID = WordToIDMap
        self.UNK_ID = int(UnknownWordID)
        self.T2ID = TagToIDMap

    def __len__(self):
        return len(self.DatasetStorage)

    def __getitem__(self, IndexPointer):
        WordSequence, TagSequence = self.DatasetStorage[IndexPointer]
        
        IntegerWordIDs = [self.W2ID.get(W, self.UNK_ID) for W in WordSequence]
        IntegerTagIDs = [self.T2ID[T] for T in TagSequence]
        
        return torch.tensor(IntegerWordIDs, dtype=torch.long), torch.tensor(IntegerTagIDs, dtype=torch.long)


def CustomCollateFunctionForPaddedBatches(MiniBatchSamples):
    """
    Pads sequences within a batch to a uniform length.
    Generates masks to identify non-padding positions for valid loss calculation.
    """
    MaximumBatchLengthVal = max(int(Sent[0].shape[0]) for Sent in MiniBatchSamples)
    MiniBatchSizeValue = len(MiniBatchSamples)
    
    PaddedWordsTensor = torch.zeros((MiniBatchSizeValue, MaximumBatchLengthVal), dtype=torch.long)
    PaddedTagsTensor = torch.full((MiniBatchSizeValue, MaximumBatchLengthVal), -100, dtype=torch.long)
    ObservationMaskTensor = torch.zeros((MiniBatchSizeValue, MaximumBatchLengthVal), dtype=torch.bool)
    
    for i, (Words, Tags) in enumerate(MiniBatchSamples):
        SequenceLength = int(Words.shape[0])
        PaddedWordsTensor[i, :SequenceLength] = Words
        PaddedTagsTensor[i, :SequenceLength] = Tags
        ObservationMaskTensor[i, :SequenceLength] = True
        
    return PaddedWordsTensor, PaddedTagsTensor, ObservationMaskTensor


class LinearChainCrfDependency(nn.Module):
    """
    CRF Layer for sequence labeling (NER).
    Models dependencies between adjacent tags using a transition matrix.
    """
    def __init__(self, CountOfUniqueTags):
        super().__init__()
        self.K_Tags = CountOfUniqueTags
        self.LogStartFires = nn.Parameter(torch.randn(CountOfUniqueTags) * 0.01)
        self.LogEndFires = nn.Parameter(torch.randn(CountOfUniqueTags) * 0.01)
        self.LogTransitionWeights = nn.Parameter(torch.randn(CountOfUniqueTags, CountOfUniqueTags) * 0.01)

    def ComputeLogPartitionFunctionZ(self, EmissionLogitsBatch, SequenceMaskBatch):
        """Standard Forward algorithm for CRF Log-Partition."""
        BSize, TimeSteps, KTags = EmissionLogitsBatch.shape
        ForwardAlphaScores = self.LogStartFires.unsqueeze(0) + EmissionLogitsBatch[:, 0, :]
        
        for t in range(1, TimeSteps):
            EmissionsAtT = EmissionLogitsBatch[:, t, :]
            ActiveMaskAtT = SequenceMaskBatch[:, t].unsqueeze(1).float()
            
            ExpandedAlpha = ForwardAlphaScores.unsqueeze(2) + self.LogTransitionWeights.unsqueeze(0)
            NewAlphaScores = torch.logsumexp(ExpandedAlpha, dim=1) + EmissionsAtT
            
            ForwardAlphaScores = (NewAlphaScores * ActiveMaskAtT) + (ForwardAlphaScores * (1.0 - ActiveMaskAtT))
            
        TerminationScores = ForwardAlphaScores + self.LogEndFires.unsqueeze(0)
        return torch.logsumexp(TerminationScores, dim=1)

    def ComputeGoldPathScores(self, EmissionsBatch, GoldTagsBatch, MaskBatch):
        """Calculates the log-potential scores for the actual ground truth sequence."""
        BSize, TSteps, KTags = EmissionsBatch.shape
        BatchIndexIter = torch.arange(BSize, device=EmissionsBatch.device)
        
        RunningPathScore = (self.LogStartFires[GoldTagsBatch[:, 0]] + EmissionsBatch[BatchIndexIter, 0, GoldTagsBatch[:, 0]]) * MaskBatch[:, 0].float()
        
        for t in range(1, TSteps):
            CombinedMask = (MaskBatch[:, t] & MaskBatch[:, t - 1]).float()
            PrevTagVal = GoldTagsBatch[:, t - 1].clamp(min=0)
            CurrTagVal = GoldTagsBatch[:, t].clamp(min=0)
            
            TransitionScoreTerm = self.LogTransitionWeights[PrevTagVal, CurrTagVal] + EmissionsBatch[BatchIndexIter, t, CurrTagVal]
            RunningPathScore = RunningPathScore + (TransitionScoreTerm * CombinedMask)
            
        LastValidIndices = MaskBatch.long().sum(dim=1) - 1
        ActualLastTags = GoldTagsBatch[BatchIndexIter, LastValidIndices]
        RunningPathScore = RunningPathScore + self.LogEndFires[ActualLastTags]
        
        return RunningPathScore

    def NegLogLikelihoodScore(self, EmissionsLogits, TrueTags, AttentionMask):
        """Negative log-likelihood: LogZ - Score(Gold)"""
        LogPartitionTotal = self.ComputeLogPartitionFunctionZ(EmissionsLogits, AttentionMask)
        ObservedPathScore = self.ComputeGoldPathScores(EmissionsLogits, TrueTags, AttentionMask)
        return (LogPartitionTotal - ObservedPathScore).mean()

    def PerformViterbiDecoding(self, EmissionsLogits, AttentionMask):
        """Inference using the Viterbi algorithm to find the highest-scoring sequence."""
        BSz, TSz, KSz = EmissionsLogits.shape
        BatchIter = torch.arange(BSz, device=EmissionsLogits.device)
        
        CumulativeScore = self.LogStartFires.unsqueeze(0) + EmissionsLogits[:, 0, :]
        BacktrackPointers = torch.zeros((BSz, TSz, KSz), dtype=torch.long, device=EmissionsLogits.device)
        
        for t in range(1, TSz):
            ExpPkg = CumulativeScore.unsqueeze(2) + self.LogTransitionWeights.unsqueeze(0)
            BestPrevScore, BestPrevIdx = ExpPkg.max(dim=1)
            
            NewScoreAccum = BestPrevScore + EmissionsLogits[:, t, :]
            Msk = AttentionMask[:, t].unsqueeze(1).float()
            
            CumulativeScore = (NewScoreAccum * Msk) + (CumulativeScore * (1.0 - Msk))
            BacktrackPointers[:, t, :] = BestPrevIdx
            
        FinalScoresTable = CumulativeScore + self.LogEndFires.unsqueeze(0)
        BestPathHead = FinalScoresTable.argmax(dim=1)
        
        DecodedOutputMatrix = torch.zeros((BSz, TSz), dtype=torch.long, device=EmissionsLogits.device)
        DecodedOutputMatrix[BatchIter, TSz - 1] = BestPathHead
        
        for t in range(TSz - 2, -1, -1):
            NextStepTags = DecodedOutputMatrix[:, t + 1].unsqueeze(1)
            DecodedOutputMatrix[:, t] = torch.gather(BacktrackPointers[:, t + 1, :], 1, NextStepTags).squeeze(1)
            
        return DecodedOutputMatrix


class BiLSTMTaggerCore(nn.Module):
    """Core LSTM logic applied for Part-of-Speech categorization."""
    def __init__(self, PretrainedWeights, ShouldFreezeEmbeds, LabelCountSize, HiddenSizeUnits, LayerDepthCount, PaddingIdxVal, DropoutRatioVal):
        super().__init__()
        VocabCountVal, DimValue = PretrainedWeights.shape
        self.LexicalEmbeddingLayer = nn.Embedding(VocabCountVal, DimValue, padding_idx=PaddingIdxVal)
        self.LexicalEmbeddingLayer.weight.data.copy_(PretrainedWeights)
        self.LexicalEmbeddingLayer.weight.requires_grad = not ShouldFreezeEmbeds
        
        self.RecurrentNetwork = nn.LSTM(
            DimValue,
            HiddenSizeUnits,
            num_layers=LayerDepthCount,
            bidirectional=True,
            batch_first=True,
            dropout=DropoutRatioVal if LayerDepthCount > 1 else 0.0,
        )
        self.ClassificationHead = nn.Linear(2 * HiddenSizeUnits, LabelCountSize)

    def forward(self, TensorOfWordIDs, TensorOfMasks):
        EmbeddedTokens = self.LexicalEmbeddingLayer(TensorOfWordIDs)
        SequenceLengthsForPack = TensorOfMasks.long().sum(dim=1).clamp(min=1).cpu()
        PackedInputBuffer = nn.utils.rnn.pack_padded_sequence(EmbeddedTokens, SequenceLengthsForPack, batch_first=True, enforce_sorted=False)
        
        LstmHiddensOutput, _ = self.RecurrentNetwork(PackedInputBuffer)
        PaddedHiddensOutput, _ = nn.utils.rnn.pad_packed_sequence(LstmHiddensOutput, batch_first=True)
        
        return self.ClassificationHead(PaddedHiddensOutput)


class BiLSTMNerCrfModel(nn.Module):
    """Integrated BiLSTM+CRF model for Named Entity Recognition."""
    def __init__(self, PretrainedWeights, ShouldFreezeEmbeds, LabelCountSize, HiddenSizeUnits, LayerDepthCount, PaddingIdxVal, DropoutRatioVal):
        super().__init__()
        VocabCountVal, DimValue = PretrainedWeights.shape
        self.LexicalEmbeddingLayer = nn.Embedding(VocabCountVal, DimValue, padding_idx=PaddingIdxVal)
        self.LexicalEmbeddingLayer.weight.data.copy_(PretrainedWeights)
        self.LexicalEmbeddingLayer.weight.requires_grad = not ShouldFreezeEmbeds
        
        self.RecurrentNetwork = nn.LSTM(
            DimValue,
            HiddenSizeUnits,
            num_layers=LayerDepthCount,
            bidirectional=True,
            batch_first=True,
            dropout=DropoutRatioVal if LayerDepthCount > 1 else 0.0,
        )
        self.EmissionProjector = nn.Linear(2 * HiddenSizeUnits, LabelCountSize)
        self.CrfDependencyLayer = LinearChainCrfDependency(LabelCountSize)

    def forward(self, TensorOfWordIDs, TensorOfMasks):
        EmbeddedTokens = self.LexicalEmbeddingLayer(TensorOfWordIDs)
        SequenceLengthsForPack = TensorOfMasks.long().sum(dim=1).clamp(min=1).cpu()
        PackedInputBuffer = nn.utils.rnn.pack_padded_sequence(EmbeddedTokens, SequenceLengthsForPack, batch_first=True, enforce_sorted=False)
        
        LstmHiddensOutput, _ = self.RecurrentNetwork(PackedInputBuffer)
        PaddedHiddensOutput, _ = nn.utils.rnn.pad_packed_sequence(LstmHiddensOutput, batch_first=True)
        
        return self.EmissionProjector(PaddedHiddensOutput)


def FlattenPredictionsAndGroundTruths(LogitTensors, GoldTagTensors, MaskTensors):
    """Converts batched tokens into flat lists of predictions and true labels for metric scoring."""
    ArgmaxIndices = LogitTensors.argmax(dim=-1)
    FlattenedGoldValues = []
    FlattenedPredictedValues = []
    
    for b in range(GoldTagTensors.shape[0]):
        for t in range(GoldTagTensors.shape[1]):
            if not MaskTensors[b, t]:
                continue
            TagIDValue = int(GoldTagTensors[b, t].item())
            if TagIDValue == -100:
                continue
            FlattenedGoldValues.append(TagIDValue)
            FlattenedPredictedValues.append(int(ArgmaxIndices[b, t].item()))
            
    return FlattenedGoldValues, FlattenedPredictedValues


def ComputeMacroF1StatisticFromLists(TrueIDsList, PredictedIDsList, DistinctLabelCount):
    """Calculates the Macro-averaged F1 score across all classes."""
    if not TrueIDsList:
        return 0.0
        
    InternalConfusionMatrix = np.zeros((DistinctLabelCount, DistinctLabelCount), dtype=np.float64)
    for GroundTruth, Prediction in zip(TrueIDsList, PredictedIDsList):
        if 0 <= GroundTruth < DistinctLabelCount and 0 <= Prediction < DistinctLabelCount:
            InternalConfusionMatrix[GroundTruth, Prediction] += 1.0
            
    IndividualClassF1Buffer = []
    for ClassIdx in range(DistinctLabelCount):
        TruePositives = InternalConfusionMatrix[ClassIdx, ClassIdx]
        FalsePositives = InternalConfusionMatrix[:, ClassIdx].sum() - TruePositives
        FalseNegatives = InternalConfusionMatrix[ClassIdx, :].sum() - TruePositives
        
        if TruePositives + FalsePositives + FalseNegatives <= 0:
            continue
            
        PrecisionVal = TruePositives / (TruePositives + FalsePositives + 1e-12)
        RecallVal = TruePositives / (TruePositives + FalseNegatives + 1e-12)
        ClassF1 = 2 * PrecisionVal * RecallVal / (PrecisionVal + RecallVal + 1e-12)
        IndividualClassF1Buffer.append(ClassF1)
        
    return float(np.mean(IndividualClassF1Buffer)) if IndividualClassF1Buffer else 0.0


def EvaluateNerDecodingPerformance(ActiveModel, DataIterator, HardwareDevice, LabelQty):
    """Inference loop for NER using Viterbi decoding to compute validation Macro-F1."""
    ActiveModel.eval()
    AggregateTrueLabels = []
    AggregatePredictedLabels = []
    
    with torch.no_grad():
        for WordIDs, TargetTags, MaskingBits in DataIterator:
            WordIDs, TargetTags, MaskingBits = WordIDs.to(HardwareDevice), TargetTags.to(HardwareDevice), MaskingBits.to(HardwareDevice)
            
            GeneratedEmissions = ActiveModel(WordIDs, MaskingBits)
            MaskedEmissions = GeneratedEmissions * MaskingBits.unsqueeze(-1).float()
            
            TopScoringPath = ActiveModel.CrfDependencyLayer.PerformViterbiDecoding(MaskedEmissions, MaskingBits)
            
            for b in range(WordIDs.shape[0]):
                for t in range(WordIDs.shape[1]):
                    if not MaskingBits[b, t]:
                        continue
                    AggregateTrueLabels.append(int(TargetTags[b, t].item()))
                    AggregatePredictedLabels.append(int(TopScoringPath[b, t].item()))
                    
    return ComputeMacroF1StatisticFromLists(AggregateTrueLabels, AggregatePredictedLabels, LabelQty)


def ExecutePosTrainingRegime(FreezeWeightsBit, TrainDatasetObj, ValDatasetObj, MainEmbeddingTensors, DeviceHandle, HidSize, LayerDepth, DropPct, LearningRateVal, WeightDecayCoeff, PatienceThreshold, MaxEpochLimit, VisualizationPath):
    """Orchestrates the BiLSTM training flow for POS tagging."""
    TrainLoaderPtr = DataLoader(TrainDatasetObj, batch_size=16, shuffle=True, collate_fn=CustomCollateFunctionForPaddedBatches)
    ValLoaderPtr = DataLoader(ValDatasetObj, batch_size=16, shuffle=False, collate_fn=CustomCollateFunctionForPaddedBatches)
    
    TotalTagsQuantity = len(TrainDatasetObj.T2ID)
    TaggerInstance = BiLSTMTaggerCore(MainEmbeddingTensors, FreezeWeightsBit, TotalTagsQuantity, HidSize, LayerDepth, 0, DropPct).to(DeviceHandle)
    
    ParameterOptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, TaggerInstance.parameters()), lr=LearningRateVal, weight_decay=WeightDecayCoeff)
    CategoricalLossCriterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    TrainingLossTrendHistory = []
    ValidationLossTrendHistory = []
    BestObservedMacroF1 = -1.0
    SnapshotOfBestStateDict = None
    EpochsWithoutImprovementCounter = 0
    
    for EpIdx in range(MaxEpochLimit):
        TaggerInstance.train()
        RunningLossAccumulator = 0.0
        TotalStepsInEpoch = 0
        
        for Words, Targets, Masks in TrainLoaderPtr:
            Words, Targets, Masks = Words.to(DeviceHandle), Targets.to(DeviceHandle), Masks.to(DeviceHandle)
            ParameterOptimizer.zero_grad()
            
            PredictionsLogits = TaggerInstance(Words, Masks)
            BatchScalarLoss = CategoricalLossCriterion(PredictionsLogits.view(-1, TotalTagsQuantity), Targets.view(-1))
            BatchScalarLoss.backward()
            ParameterOptimizer.step()
            
            RunningLossAccumulator += float(BatchScalarLoss.item())
            TotalStepsInEpoch += 1
            
        TrainingLossTrendHistory.append(RunningLossAccumulator / max(1, TotalStepsInEpoch))
        
        # Validation Phase
        TaggerInstance.eval()
        ValAccumulatedLoss = 0.0
        ValStepsCount = 0
        ObservedGoldList = []
        ObservedPredList = []
        
        with torch.no_grad():
            for WordsV, TargetsV, MasksV in ValLoaderPtr:
                WordsV, TargetsV, MasksV = WordsV.to(DeviceHandle), TargetsV.to(DeviceHandle), MasksV.to(DeviceHandle)
                
                ValOuts = TaggerInstance(WordsV, MasksV)
                ValAccumulatedLoss += float(CategoricalLossCriterion(ValOuts.view(-1, TotalTagsQuantity), TargetsV.view(-1)).item())
                ValStepsCount += 1
                
                SubsetGold, SubsetPred = FlattenPredictionsAndGroundTruths(ValOuts, TargetsV, MasksV)
                ObservedGoldList.extend(SubsetGold)
                ObservedPredList.extend(SubsetPred)
                
        MeanValLossForEpoch = ValAccumulatedLoss / max(1, ValStepsCount)
        ValidationLossTrendHistory.append(MeanValLossForEpoch)
        
        CurrentEpochMacroF1 = ComputeMacroF1StatisticFromLists(ObservedGoldList, ObservedPredList, TotalTagsQuantity)
        
        LogicLabelStr = "STATIONARY_EMB" if FreezeWeightsBit else "ADAPTIVE_EMB"
        print(f"POS | {LogicLabelStr} | EP {EpIdx + 1} | TR_LOSS: {round(TrainingLossTrendHistory[-1], 5)} | VA_LOSS: {round(MeanValLossForEpoch, 5)} | VA_F1: {round(CurrentEpochMacroF1, 5)}", flush=True)
        
        if CurrentEpochMacroF1 > BestObservedMacroF1 + 1e-6:
            BestObservedMacroF1 = CurrentEpochMacroF1
            SnapshotOfBestStateDict = copy.deepcopy(TaggerInstance.state_dict())
            EpochsWithoutImprovementCounter = 0
        else:
            EpochsWithoutImprovementCounter += 1
            if EpochsWithoutImprovementCounter >= PatienceThreshold:
                print(f"POS {LogicLabelStr}: Early convergence stopping at epoch {EpIdx + 1}")
                break
                
    # Persist the loss plot
    FigObj, AxObj = plt.subplots(figsize=(8, 5))
    AxObj.plot(range(1, len(TrainingLossTrendHistory) + 1), TrainingLossTrendHistory, label="Training Loss")
    AxObj.plot(range(1, len(ValidationLossTrendHistory) + 1), ValidationLossTrendHistory, label="Validation Loss")
    AxObj.set_xlabel("Epoch Number")
    AxObj.set_ylabel("Loss Magnitude")
    AxObj.set_title(f"POS Tagger Learning Progress ({'Frozen' if FreezeWeightsBit else 'Tuned'} Embeddings)")
    AxObj.legend()
    plt.tight_layout()
    plt.savefig(VisualizationPath, dpi=150)
    plt.close(FigObj)
    
    return SnapshotOfBestStateDict, BestObservedMacroF1


def ExecuteNerTrainingRegime(FreezeWeightsBit, TrainDatasetObj, ValDatasetObj, MainEmbeddingTensors, DeviceHandle, HidSize, LayerDepth, DropPct, LearningRateVal, WeightDecayCoeff, PatienceThreshold, MaxEpochLimit, VisualizationPath):
    """Orchestrates the BiLSTM+CRF training flow for NER."""
    TrainLoaderPtr = DataLoader(TrainDatasetObj, batch_size=8, shuffle=True, collate_fn=CustomCollateFunctionForPaddedBatches)
    ValLoaderPtr = DataLoader(ValDatasetObj, batch_size=8, shuffle=False, collate_fn=CustomCollateFunctionForPaddedBatches)
    
    TotalTagsQuantity = len(TrainDatasetObj.T2ID)
    NerNetworkInstance = BiLSTMNerCrfModel(MainEmbeddingTensors, FreezeWeightsBit, TotalTagsQuantity, HidSize, LayerDepth, 0, DropPct).to(DeviceHandle)
    
    ParameterOptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, NerNetworkInstance.parameters()), lr=LearningRateVal, weight_decay=WeightDecayCoeff)
    
    TrainingLossTrendHistory = []
    ValidationLossTrendHistory = []
    BestObservedMacroF1 = -1.0
    SnapshotOfBestStateDict = None
    EpochsWithoutImprovementCounter = 0
    
    for EpIdx in range(MaxEpochLimit):
        NerNetworkInstance.train()
        RunningLossAccumulator = 0.0
        TotalStepsInEpoch = 0
        
        for Words, Targets, Masks in TrainLoaderPtr:
            Words, Targets, Masks = Words.to(DeviceHandle), Targets.to(DeviceHandle), Masks.to(DeviceHandle)
            ParameterOptimizer.zero_grad()
            
            EmissionLogits = NerNetworkInstance(Words, Masks)
            MaskedEmissions = EmissionLogits * Masks.unsqueeze(-1).float()
            
            BatchScalarLoss = NerNetworkInstance.CrfDependencyLayer.NegLogLikelihoodScore(MaskedEmissions, Targets, Masks)
            BatchScalarLoss.backward()
            ParameterOptimizer.step()
            
            RunningLossAccumulator += float(BatchScalarLoss.item())
            TotalStepsInEpoch += 1
            
        TrainingLossTrendHistory.append(RunningLossAccumulator / max(1, TotalStepsInEpoch))
        
        NerNetworkInstance.eval()
        ValAccumulatedLoss = 0.0
        ValStepsCount = 0
        
        with torch.no_grad():
            for WordsV, TargetsV, MasksV in ValLoaderPtr:
                WordsV, TargetsV, MasksV = WordsV.to(DeviceHandle), TargetsV.to(DeviceHandle), MasksV.to(DeviceHandle)
                
                ValEmits = NerNetworkInstance(WordsV, MasksV)
                ValMaskedEmits = ValEmits * MasksV.unsqueeze(-1).float()
                
                ValAccumulatedLoss += float(NerNetworkInstance.CrfDependencyLayer.NegLogLikelihoodScore(ValMaskedEmits, TargetsV, MasksV).item())
                ValStepsCount += 1
                
        MeanValLossForEpoch = ValAccumulatedLoss / max(1, ValStepsCount)
        ValidationLossTrendHistory.append(MeanValLossForEpoch)
        
        # Specific Viterbi-based valuation for NER
        CurrentEpochMacroF1 = EvaluateNerDecodingPerformance(NerNetworkInstance, ValLoaderPtr, DeviceHandle, TotalTagsQuantity)
        
        LogicLabelStr = "STATIONARY_EMB" if FreezeWeightsBit else "ADAPTIVE_EMB"
        print(f"NER | {LogicLabelStr} | EP {EpIdx + 1} | TR_LOSS: {round(TrainingLossTrendHistory[-1], 5)} | VA_LOSS: {round(MeanValLossForEpoch, 5)} | VA_F1: {round(CurrentEpochMacroF1, 5)}", flush=True)
        
        if CurrentEpochMacroF1 > BestObservedMacroF1 + 1e-6:
            BestObservedMacroF1 = CurrentEpochMacroF1
            SnapshotOfBestStateDict = copy.deepcopy(NerNetworkInstance.state_dict())
            EpochsWithoutImprovementCounter = 0
        else:
            EpochsWithoutImprovementCounter += 1
            if EpochsWithoutImprovementCounter >= PatienceThreshold:
                print(f"NER {LogicLabelStr}: Early convergence stopping at epoch {EpIdx + 1}")
                break
                
    # Persist the loss plot
    FigObj, AxObj = plt.subplots(figsize=(8, 5))
    AxObj.plot(range(1, len(TrainingLossTrendHistory) + 1), TrainingLossTrendHistory, label="Training Loss")
    AxObj.plot(range(1, len(ValidationLossTrendHistory) + 1), ValidationLossTrendHistory, label="Validation Loss")
    AxObj.set_xlabel("Epoch Number")
    AxObj.set_ylabel("Loss Magnitude")
    AxObj.set_title(f"NER Tagger Learning Progress ({'Frozen' if FreezeWeightsBit else 'Tuned'} Embeddings)")
    AxObj.legend()
    plt.tight_layout()
    plt.savefig(VisualizationPath, dpi=150)
    plt.close(FigObj)
    
    return SnapshotOfBestStateDict, BestObservedMacroF1


def RunMainTrainingPipeline():
    """Primary driver for the training script. Configures parameters and initiates runs for both tasks."""
    EstablishStandardOutputEncoding()
    
    # Establish consistent randomness
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    HardwareDeviceTarget = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Word2Vec prerequisites
    try:
        WordToIndexSource = json.load(open("embeddings/word2idx.json", mode="r", encoding="utf-8"))
        PretrainedNumpyVectors = np.load("embeddings/embeddings_w2v.npy")
    except FileNotFoundError:
        print("ERROR: Prerequisite embedding files missing in 'embeddings/' directory.")
        return

    WordToIDMapping, UnknownID, TotalVocabWithPad = AdaptWordToIndexForPadding(WordToIndexSource)
    FinalEmbeddingWeightsTable = PrepareEmbeddingWeightsWithPadRow(PretrainedNumpyVectors, TotalVocabWithPad, PretrainedNumpyVectors.shape[1])

    # Load sequence datasets
    FullPosSentences = LoadSentencesFromConllFormat("data/pos_train.conll")
    FullNerSentences = LoadSentencesFromConllFormat("data/ner_train.conll")
    
    if len(FullPosSentences) != len(FullNerSentences):
        print("CRITICAL: Sentence count mismatch between POS and NER training sets.")
        return

    # Create stratified splits for validation monitoring
    PosTrainPart, PosValPart = PartitionTrainingAndValidationSets(FullPosSentences, 0.15, 42)
    NerTrainPart, NerValPart = PartitionTrainingAndValidationSets(FullNerSentences, 0.15, 42)

    PosTagIDMap, OrderedPosTags = CreateTagIndexDictionaries(PosTrainPart + PosValPart)
    NerTagIDMap, OrderedNerTags = CreateTagIndexDictionaries(NerTrainPart + NerValPart)

    # Wrap in Dataset objects
    DatasetPosTrain = NeuralSentenceLabelingDataset(PosTrainPart, WordToIDMapping, UnknownID, PosTagIDMap)
    DatasetPosVal = NeuralSentenceLabelingDataset(PosValPart, WordToIDMapping, UnknownID, PosTagIDMap)
    DatasetNerTrain = NeuralSentenceLabelingDataset(NerTrainPart, WordToIDMapping, UnknownID, NerTagIDMap)
    DatasetNerVal = NeuralSentenceLabelingDataset(NerValPart, WordToIDMapping, UnknownID, NerTagIDMap)

    # Hyper-parameters
    HiddenStateDims = 100
    LayerQuantity = 2
    DropoutCoefficient = 0.5
    LearningCoeff = 1e-3
    L2RegularizationWD = 1e-4
    EarlyStoppingPatienceVal = 5
    UpperLimitOnEpochs = 60

    os.makedirs("models", exist_ok=True)

    # --- TASK 1: PART-OF-SPEECH TRAINING ---
    print("\n" + "#"*40)
    print(" INITIATING POS TRAINING (FROZEN EMBEDDINGS) ")
    print("#"*40)
    BestPosStateFrozen, F1PosFrozen = ExecutePosTrainingRegime(
        True, DatasetPosTrain, DatasetPosVal, FinalEmbeddingWeightsTable, HardwareDeviceTarget, 
        HiddenStateDims, LayerQuantity, DropoutCoefficient, LearningCoeff, L2RegularizationWD, 
        EarlyStoppingPatienceVal, UpperLimitOnEpochs, "models/pos_loss_frozen.png"
    )

    print("\n" + "#"*40)
    print(" INITIATING POS TRAINING (FINETUNED EMBEDDINGS) ")
    print("#"*40)
    BestPosStateTuned, F1PosTuned = ExecutePosTrainingRegime(
        False, DatasetPosTrain, DatasetPosVal, FinalEmbeddingWeightsTable, HardwareDeviceTarget, 
        HiddenStateDims, LayerQuantity, DropoutCoefficient, LearningCoeff, L2RegularizationWD, 
        EarlyStoppingPatienceVal, UpperLimitOnEpochs, "models/pos_loss_finetune.png"
    )

    # Save the optimized POS model (Adaptive version usually performs better)
    torch.save({
        "state_dict": BestPosStateTuned,
        "pos_tag_to_idx": PosTagIDMap,
        "pos_idx_to_tag": OrderedPosTags,
        "hid": HiddenStateDims,
        "layers": LayerQuantity,
        "dropout": DropoutCoefficient,
        "vocab_plus_one": TotalVocabWithPad,
        "emb_dim": int(PretrainedNumpyVectors.shape[1]),
        "val_f1_frozen_best": F1PosFrozen,
        "val_f1_finetune_best": F1PosTuned,
    }, "models/bilstm_pos.pt")

    # --- TASK 2: NAMED ENTITY RECOGNITION TRAINING ---
    print("\n" + "#"*40)
    print(" INITIATING NER TRAINING (FROZEN EMBEDDINGS) ")
    print("#"*40)
    BestNerStateFrozen, F1NerFrozen = ExecuteNerTrainingRegime(
        True, DatasetNerTrain, DatasetNerVal, FinalEmbeddingWeightsTable, HardwareDeviceTarget, 
        HiddenStateDims, LayerQuantity, DropoutCoefficient, LearningCoeff, L2RegularizationWD, 
        EarlyStoppingPatienceVal + 3, UpperLimitOnEpochs, "models/ner_loss_frozen.png"
    )

    print("\n" + "#"*40)
    print(" INITIATING NER TRAINING (FINETUNED EMBEDDINGS) ")
    print("#"*40)
    BestNerStateTuned, F1NerTuned = ExecuteNerTrainingRegime(
        False, DatasetNerTrain, DatasetNerVal, FinalEmbeddingWeightsTable, HardwareDeviceTarget, 
        HiddenStateDims, LayerQuantity, DropoutCoefficient, LearningCoeff, L2RegularizationWD, 
        EarlyStoppingPatienceVal + 3, UpperLimitOnEpochs, "models/ner_loss_finetune.png"
    )

    # Save the optimized NER model
    torch.save({
        "state_dict": BestNerStateTuned,
        "ner_tag_to_idx": NerTagIDMap,
        "ner_idx_to_tag": OrderedNerTags,
        "hid": HiddenStateDims,
        "layers": LayerQuantity,
        "dropout": DropoutCoefficient,
        "vocab_plus_one": TotalVocabWithPad,
        "emb_dim": int(PretrainedNumpyVectors.shape[1]),
        "val_f1_frozen_best": F1NerFrozen,
        "val_f1_finetune_best": F1NerTuned,
    }, "models/bilstm_ner.pt")

    print("\nTRAINING CYCLE CONCLUDED. CHECKPOINTS SAVED IN 'models/'.")


if __name__ == "__main__":
    RunMainTrainingPipeline()
