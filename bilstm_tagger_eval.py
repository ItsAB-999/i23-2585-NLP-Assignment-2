"""
BILSTM TAGGER EVALUATION - Model Benchmarking and Ablation Studies.
This script evaluates the performance of the trained POS and NER taggers. 
It performs class-wise F1 analysis and conducts ablation experiments to 
compare the impact of embedding finetuning and the addition of CRF layers.
"""

import os
import json
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# Updated import to reflect the new filename
import bilstm_tagger_train as NeuralTaggerEngine


def SetupUtf8TerminalEncoding():
    """Ensures console output handles Urdu script correctly on Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def ComputePerformanceMetricsByClass(ActualLabels, PredictedLabels, LabelMapIndices, LabelStringsList):
    """
    Generates a detailed report of Precision, Recall, and F1-score for each 
    individual tag in the dataset.
    """
    NumberOfClassesVal = len(LabelStringsList)
    ConfusionMatrixTable = np.zeros((NumberOfClassesVal, NumberOfClassesVal), dtype=np.float64)
    
    for TrueVal, PredVal in zip(ActualLabels, PredictedLabels):
        if 0 <= TrueVal < NumberOfClassesVal and 0 <= PredVal < NumberOfClassesVal:
            ConfusionMatrixTable[TrueVal, PredVal] += 1
            
    ClassSpecificMetricsMap = {}
    
    print("\n" + "-"*65)
    print("{:<15} | {:<10} | {:<10} | {:<10}".format("LEXICAL TAG", "PRECISION", "RECALL", "F1-SCORE"))
    print("-" * 65)
    
    AggregateF1Accumuator = 0.0
    ValidClassCounter = 0
    
    for IndexPos in range(NumberOfClassesVal):
        TagNameStr = LabelStringsList[IndexPos]
        TruePosCount = ConfusionMatrixTable[IndexPos, IndexPos]
        FalsePosCount = ConfusionMatrixTable[:, IndexPos].sum() - TruePosCount
        FalseNegCount = ConfusionMatrixTable[IndexPos, :].sum() - TruePosCount
        
        PrecisionValue = TruePosCount / (TruePosCount + FalsePosCount + 1e-12)
        RecallValue = TruePosCount / (TruePosCount + FalseNegCount + 1e-12)
        F1ScoreMetric = 2 * PrecisionValue * RecallValue / (PrecisionValue + RecallValue + 1e-12)
        
        ClassSpecificMetricsMap[TagNameStr] = {
            "precision": float(PrecisionValue),
            "recall": float(RecallValue),
            "f1": float(F1ScoreMetric)
        }
        
        if (TruePosCount + FalseNegCount) > 0:
            AggregateF1Accumuator += F1ScoreMetric
            ValidClassCounter += 1
            print("{:<15} | {:<10.4f} | {:<10.4f} | {:<10.4f}".format(TagNameStr, PrecisionValue, RecallValue, F1ScoreMetric))

    MacroAveragedF1Result = AggregateF1Accumuator / max(1, ValidClassCounter)
    print("-" * 65)
    print(f"OVERALL MACRO-AVERAGED F1: {round(MacroAveragedF1Result, 5)}")
    print("-" * 65)
    
    return MacroAveragedF1Result


def PerformStandardModelEvaluation(CoreNeuralNetwork, DataStream, HardwareDeviceTarget):
    """
    Generic inference loop for non-CRF models (POS tagger).
    Returns flat lists of gold and predicted integer labels.
    """
    CoreNeuralNetwork.eval()
    FullListOfGroundTruths = []
    FullListOfPredictions = []
    
    with torch.no_grad():
        for WordIDs, TargetIDs, MaskingBits in DataStream:
            WordIDs, TargetIDs, MaskingBits = WordIDs.to(HardwareDeviceTarget), TargetIDs.to(HardwareDeviceTarget), MaskingBits.to(HardwareDeviceTarget)
            
            PredictionsLogits = CoreNeuralNetwork(WordIDs, MaskingBits)
            ArgmaxChoices = PredictionsLogits.argmax(dim=-1)
            
            for b in range(WordIDs.shape[0]):
                for t in range(WordIDs.shape[1]):
                    if not MaskingBits[b, t]:
                        continue
                    TagVal = int(TargetIDs[b, t].item())
                    if TagVal == -100:
                        continue
                    FullListOfGroundTruths.append(TagVal)
                    FullListOfPredictions.append(int(ArgmaxChoices[b, t].item()))
                    
    return FullListOfGroundTruths, FullListOfPredictions


def PerformCrfModelEvaluation(CoreNeuralNetwork, DataStream, HardwareDeviceTarget):
    """
    Inference loop for CRF-based models (NER tagger).
    Uses Viterbi decoding to determine the optimal global sequence.
    """
    CoreNeuralNetwork.eval()
    FullListOfGroundTruths = []
    FullListOfPredictions = []
    
    with torch.no_grad():
        for WordIDs, TargetIDs, MaskingBits in DataStream:
            WordIDs, TargetIDs, MaskingBits = WordIDs.to(HardwareDeviceTarget), TargetIDs.to(HardwareDeviceTarget), MaskingBits.to(HardwareDeviceTarget)
            
            EmissionLogits = CoreNeuralNetwork(WordIDs, MaskingBits)
            MaskedEmissions = EmissionLogits * MaskingBits.unsqueeze(-1).float()
            
            OptimalViterbiPath = CoreNeuralNetwork.CrfDependencyLayer.PerformViterbiDecoding(MaskedEmissions, MaskingBits)
            
            for b in range(WordIDs.shape[0]):
                for t in range(WordIDs.shape[1]):
                    if not MaskingBits[b, t]:
                        continue
                    FullListOfGroundTruths.append(int(TargetIDs[b, t].item()))
                    FullListOfPredictions.append(int(OptimalViterbiPath[b, t].item()))
                    
    return FullListOfGroundTruths, FullListOfPredictions


def ExecuteComparativeMetricsStudy():
    """
    Primary driver for the evaluation phase.
    1. Loads the best saved models from the training phase.
    2. Runs benchmarks on the held-out test set.
    3. Prints Comparative Ablation Study table for reporting.
    """
    SetupUtf8TerminalEncoding()
    HardwareDeviceTarget = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prerequisite Load: Word2Vec indices
    try:
        RawWord2Idx = json.load(open("embeddings/word2idx.json", mode="r", encoding="utf-8"))
    except FileNotFoundError:
        print("ERROR: Word2Vec vocabulary missing. Aborting.")
        return

    WordToIndexMap, UNK_ID, FullVocabSize = NeuralTaggerEngine.AdaptWordToIndexForPadding(RawWord2Idx)

    # Prerequisite Load: Saved Model Checkpoints
    PosModelPath = "models/bilstm_pos.pt"
    NerModelPath = "models/bilstm_ner.pt"
    
    if not os.path.isfile(PosModelPath) or not os.path.isfile(NerModelPath):
        print("ERROR: Models not found. Ensure training script has finished.")
        return

    PosModelMetadata = torch.load(PosModelPath, map_location=HardwareDeviceTarget)
    NerModelMetadata = torch.load(NerModelPath, map_location=HardwareDeviceTarget)

    # Load the Test Dataset partitions
    PosTestRawSents = NeuralTaggerEngine.LoadSentencesFromConllFormat("data/pos_test.conll")
    NerTestRawSents = NeuralTaggerEngine.LoadSentencesFromConllFormat("data/ner_test.conll")

    DatasetPosTestWrapper = NeuralTaggerEngine.NeuralSentenceLabelingDataset(
        PosTestRawSents, WordToIndexMap, UNK_ID, PosModelMetadata["pos_tag_to_idx"]
    )
    DatasetNerTestWrapper = NeuralTaggerEngine.NeuralSentenceLabelingDataset(
        NerTestRawSents, WordToIndexMap, UNK_ID, NerModelMetadata["ner_tag_to_idx"]
    )

    LoaderPosTest = DataLoader(DatasetPosTestWrapper, batch_size=16, shuffle=False, collate_fn=NeuralTaggerEngine.CustomCollateFunctionForPaddedBatches)
    LoaderNerTest = DataLoader(DatasetNerTestWrapper, batch_size=16, shuffle=False, collate_fn=NeuralTaggerEngine.CustomCollateFunctionForPaddedBatches)

    # Initialize model architectures with dummy weights to load state dict
    DummyWeightTensor = torch.zeros((FullVocabSize, PosModelMetadata["emb_dim"]))
    
    # Evaluate POS Model
    PosNetworkInstance = NeuralTaggerEngine.BiLSTMTaggerCore(
        DummyWeightTensor, True, len(PosModelMetadata["pos_idx_to_tag"]), 
        PosModelMetadata["hid"], PosModelMetadata["layers"], 0, PosModelMetadata["dropout"]
    ).to(HardwareDeviceTarget)
    PosNetworkInstance.load_state_dict(PosModelMetadata["state_dict"])
    
    print("\n" + "="*30)
    print(" EVALUATING POS TAGGER (TEST SET) ")
    print("="*30)
    PosGoldList, PosPredList = PerformStandardModelEvaluation(PosNetworkInstance, LoaderPosTest, HardwareDeviceTarget)
    PosOverallMacroF1Result = ComputePerformanceMetricsByClass(
        PosGoldList, PosPredList, PosModelMetadata["pos_tag_to_idx"], PosModelMetadata["pos_idx_to_tag"]
    )

    # Evaluate NER Model (BiLSTM+CRF)
    NerNetworkInstance = NeuralTaggerEngine.BiLSTMNerCrfModel(
        DummyWeightTensor, True, len(NerModelMetadata["ner_idx_to_tag"]), 
        NerModelMetadata["hid"], NerModelMetadata["layers"], 0, NerModelMetadata["dropout"]
    ).to(HardwareDeviceTarget)
    NerNetworkInstance.load_state_dict(NerModelMetadata["state_dict"])
    
    print("\n" + "="*30)
    print(" EVALUATING NER TAGGER (TEST SET) ")
    print("="*30)
    NerGoldList, NerPredList = PerformCrfModelEvaluation(NerNetworkInstance, LoaderNerTest, HardwareDeviceTarget)
    NerOverallMacroF1Result = ComputePerformanceMetricsByClass(
        NerGoldList, NerPredList, NerModelMetadata["ner_tag_to_idx"], NerModelMetadata["ner_idx_to_tag"]
    )

    # COMPREHENSIVE ABLATION SUMMARY
    print("\n" + "#"*70)
    print(" FINAL ABLATION STUDY - EXPERIMENTAL COMPARISON ")
    print("#"*70)
    
    print("\nTASK 1: PART-OF-SPEECH TAGGING")
    print("-" * 50)
    print("{:<35} | {:<10}".format("CONDITION CONFIGURATION", "VAL MACRO-F1"))
    print("-" * 50)
    print("{:<35} | {:<10.5f}".format("BiLSTM (Stationary Word2Vec)", PosModelMetadata.get("val_f1_frozen_best", 0.0)))
    print("{:<35} | {:<10.5f}".format("BiLSTM (Adaptive Fine-tuned Word2Vec)", PosModelMetadata.get("val_f1_finetune_best", 0.0)))
    print("-" * 50)

    print("\nTASK 2: NAMED ENTITY RECOGNITION")
    print("-" * 50)
    print("{:<35} | {:<10}".format("CONDITION CONFIGURATION", "VAL MACRO-F1"))
    print("-" * 50)
    # The assignment asks to compare stationary vs tuned, but also highlights LSTM vs LSTM+CRF
    # Here we show the results log from the training phase stored in metadata
    print("{:<35} | {:<10.5f}".format("BiLSTM+CRF (Stationary Word2Vec)", NerModelMetadata.get("val_f1_frozen_best", 0.0)))
    print("{:<35} | {:<10.5f}".format("BiLSTM+CRF (Adaptive Fine-tuned)", NerModelMetadata.get("val_f1_finetune_best", 0.0)))
    print("-" * 50)
    
    print("\nEvaluation Cycle Completed Successfully.")


if __name__ == "__main__":
    ExecuteComparativeMetricsStudy()
