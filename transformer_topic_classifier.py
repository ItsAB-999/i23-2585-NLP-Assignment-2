"""
TRANSFORMER TOPIC CLASSIFIER - Specialized Neural Model for Article Categorization.
This module handles the training and evaluation cycle for the self-implemented 
Transformer architecture for Urdu topic classification. It integrates the 
architecture from 'transformer_architecture.py' and processed data from 'data/'.
"""

import os
import sys
import json
import time
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as torchOptim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Updated import to reflect the new architecture filename
import transformer_architecture as ModernArchitectureEngine


def InitializeSystemOutputForUrduDisplay():
    """Stabilizes the console output for Urdu character rendering."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


class ArticleClassificationDataset(Dataset):
    """Simple wrapper for pre-processed integer-mapped article sequences."""
    def __init__(self, ListOfEncodedArticles):
        self.DatasetBuffer = ListOfEncodedArticles

    def __len__(self):
        return len(self.DatasetBuffer)

    def __getitem__(self, IndexPointer):
        RecordItem = self.DatasetBuffer[IndexPointer]
        X_InputTensor = torch.tensor(RecordItem["input"], dtype=torch.long)
        Y_LabelTensor = torch.tensor(RecordItem["label"], dtype=torch.long)
        
        # Identify padding positions (0) to create the attention mask
        # Mask is 1 for real tokens, 0 for padding
        AttentionMaskBits = (X_InputTensor != 0).long()
        
        return X_InputTensor, Y_LabelTensor, AttentionMaskBits


def CalculateAccuracyMetric(LogitMatrix, TargetLabelsVector):
    """Computes the percentage of correct predictions in a batch."""
    ArgmaxDecisions = LogitMatrix.argmax(dim=1)
    CorrectMatches = (ArgmaxDecisions == TargetLabelsVector).float().sum()
    return float(CorrectMatches.item())


def ExecuteComprehensiveTrainingLoop(
    NetworkModelInstance,
    TrainingStream,
    ValidationStream,
    DeviceTarget,
    OptimInstance,
    LossCriterionFunc,
    EpochLimitVal,
    PatienceLimit,
    VisualizationPath
):
    """
    Standard supervised training loop for the Transformer model.
    Includes validation monitoring and early stopping based on loss.
    """
    TrainingLossTrends = []
    ValidationLossTrends = []
    ValidationAccuracyTrends = []
    
    MinimumValidationLossObserved = float("inf")
    BestModelStateDictCopy = None
    StagnantEpochsCounter = 0
    
    print("\n" + "#"*50)
    print(" INITIATING TRANSFORMER TRAINING CYCLE ")
    print("#"*50)

    for EpochIndex in range(EpochLimitVal):
        NetworkModelInstance.train()
        AccumulatedTrainLoss = 0.0
        TotalBatchesProcessed = 0
        
        for BatchX, BatchY, BatchMask in TrainingStream:
            BatchX, BatchY, BatchMask = BatchX.to(DeviceTarget), BatchY.to(DeviceTarget), BatchMask.to(DeviceTarget)
            
            OptimInstance.zero_grad()
            OutputLogits = NetworkModelInstance(BatchX, BatchMask)
            CalculatedLoss = LossCriterionFunc(OutputLogits, BatchY)
            
            CalculatedLoss.backward()
            OptimInstance.step()
            
            AccumulatedTrainLoss += float(CalculatedLoss.item())
            TotalBatchesProcessed += 1
            
        MeanTrainLoss = AccumulatedTrainLoss / max(1, TotalBatchesProcessed)
        TrainingLossTrends.append(MeanTrainLoss)
        
        # -- Validation Phase --
        NetworkModelInstance.eval()
        AccumulatedValLoss = 0.0
        AccumulatedValAccuracy = 0.0
        TotalValSamplesCount = 0
        ValBatchesCount = 0
        
        with torch.no_grad():
            for ValX, ValY, ValMask in ValidationStream:
                ValX, ValY, ValMask = ValX.to(DeviceTarget), ValY.to(DeviceTarget), ValMask.to(DeviceTarget)
                
                ValOuts = NetworkModelInstance(ValX, ValMask)
                ValBatchLoss = LossCriterionFunc(ValOuts, ValY)
                
                AccumulatedValLoss += float(ValBatchLoss.item())
                AccumulatedValAccuracy += CalculateAccuracyMetric(ValOuts, ValY)
                TotalValSamplesCount += ValX.size(0)
                ValBatchesCount += 1
                
        MeanValLoss = AccumulatedValLoss / max(1, ValBatchesCount)
        FinalValAccuracyMetric = AccumulatedValAccuracy / max(1, TotalValSamplesCount)
        
        ValidationLossTrends.append(MeanValLoss)
        ValidationAccuracyTrends.append(FinalValAccuracyMetric)
        
        print(f"EPOCH {EpochIndex + 1:02d} | LOSS: {MeanTrainLoss:.5f} | VAL_LOSS: {MeanValLoss:.5f} | VAL_ACC: {FinalValAccuracyMetric:.4f}")
        
        # Early Stopping Logic based on Validation Loss
        if MeanValLoss < MinimumValidationLossObserved - 1e-4:
            MinimumValidationLossObserved = MeanValLoss
            BestModelStateDictCopy = copy.deepcopy(NetworkModelInstance.state_dict())
            StagnantEpochsCounter = 0
        else:
            StagnantEpochsCounter += 1
            if StagnantEpochsCounter >= PatienceLimit:
                print(f"--- Early stopping triggered at epoch {EpochIndex + 1} ---")
                break
                
    # Persist the training curves
    Fig, Ax = plt.subplots(1, 2, figsize=(12, 5))
    Ax[0].plot(range(1, len(TrainingLossTrends)+1), TrainingLossTrends, label="Train")
    Ax[0].plot(range(1, len(ValidationLossTrends)+1), ValidationLossTrends, label="Val")
    Ax[0].set_title("Evolution of Cross-Entropy Loss")
    Ax[0].set_xlabel("Epoch")
    Ax[0].legend()
    
    Ax[1].plot(range(1, len(ValidationAccuracyTrends)+1), ValidationAccuracyTrends, color="green")
    Ax[1].set_title("Validation Accuracy Performance")
    Ax[1].set_xlabel("Epoch")
    
    plt.tight_layout()
    plt.savefig(VisualizationPath, dpi=150)
    plt.close(Fig)
    
    return BestModelStateDictCopy, MinimumValidationLossObserved


def RunTransformerTrainingPipeline():
    """Main execution function for the Transformer training script."""
    InitializeSystemOutputForUrduDisplay()
    
    # Secure deterministic behavior
    random.seed(666)
    torch.manual_seed(666)
    np.random.seed(666)
    HardwareDeviceId = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load context configuration and datasets
    DataRoot = "data"
    try:
        ConfigDictionary = json.load(open(os.path.join(DataRoot, "topic_config.json"), mode="r", encoding="utf-8"))
        RawTrainPayload = json.load(open(os.path.join(DataRoot, "topic_train.json"), mode="r", encoding="utf-8"))
        RawTestPayload = json.load(open(os.path.join(DataRoot, "topic_test.json"), mode="r", encoding="utf-8"))
    except FileNotFoundError as NF_Err:
        print(f"CRITICAL ERROR: Data files missing. Logic: {NF_Err}")
        return

    # 2. Prepare DataLoaders
    DatasetTrain = ArticleClassificationDataset(RawTrainPayload)
    DatasetTest = ArticleClassificationDataset(RawTestPayload)
    
    BatchVolumeSize = 32
    LoadingStreamTrain = DataLoader(DatasetTrain, batch_size=BatchVolumeSize, shuffle=True)
    LoadingStreamTest = DataLoader(DatasetTest, batch_size=BatchVolumeSize, shuffle=False)

    # 3. Instantiate the Architecture
    ClassifierModel = ModernArchitectureEngine.NeuralTransformerCategorizationModel(
        VocabularySpan=ConfigDictionary["vocab_size"],
        LatentEmbeddingDim=128,
        LayerStackDepth=3,
        HeadCountPerLayer=4,
        ExpansionFactor=4,
        MaxSeqLengthCapacity=ConfigDictionary["max_len"],
        TargetClassesCount=ConfigDictionary["class_count"],
        DropoutProp=0.2
    ).to(HardwareDeviceId)

    # 4. Configure Optimization Components
    TotalParametersCount = sum(p.numel() for p in ClassifierModel.parameters() if p.requires_grad)
    print(f"--- MODEL INITIALIZED | TOTAL TRAINABLE PARAMS: {TotalParametersCount} ---")

    WeightOptimizer = torchOptim.Adam(ClassifierModel.parameters(), lr=5e-4, weight_decay=1e-5)
    LossEvaluationCriterion = nn.CrossEntropyLoss()

    # 5. Run Training Process
    BestSavedState, LowestValLoss = ExecuteComprehensiveTrainingLoop(
        ClassifierModel,
        LoadingStreamTrain,
        LoadingStreamTest,
        HardwareDeviceId,
        WeightOptimizer,
        LossEvaluationCriterion,
        EpochLimitVal=80,
        PatienceLimit=8,
        VisualizationPath="models/transformer_loss.png"
    )

    # 6. Final Evaluation on Test Set using the best weights
    if BestSavedState is not None:
        ClassifierModel.load_state_dict(BestSavedState)
        
    ClassifierModel.eval()
    AbsoluteCorrectCount = 0
    TotalSamplesCounter = 0
    
    with torch.no_grad():
        for TX, TY, TM in LoadingStreamTest:
            TX, TY, TM = TX.to(HardwareDeviceId), TY.to(HardwareDeviceId), TM.to(HardwareDeviceId)
            BatchOutputs = ClassifierModel(TX, TM)
            AbsoluteCorrectCount += CalculateAccuracyMetric(BatchOutputs, TY)
            TotalSamplesCounter += TX.size(0)

    FinalBenchmarkedAccuracy = (AbsoluteCorrectCount / max(1, TotalSamplesCounter)) * 100
    print("\n" + "="*50)
    print(f" FINAL BENCHMARKED TOPIC ACCURACY: {FinalBenchmarkedAccuracy:.2f}% ")
    print("="*50)

    # 7. Persist the model checkpoint
    os.makedirs("models", exist_ok=True)
    CheckpointMetadata = {
        "state_dict": BestSavedState,
        "config": ConfigDictionary,
        "test_acc": FinalBenchmarkedAccuracy
    }
    torch.save(CheckpointMetadata, "models/transformer_topic.pt")
    print("\nSUCCESS: Saved transformer checkpoint to 'models/transformer_topic.pt'.")


if __name__ == "__main__":
    RunTransformerTrainingPipeline()
