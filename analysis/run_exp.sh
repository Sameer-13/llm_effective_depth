#!/bin/bash

export NDIF_TOKEN=""

# Define the models and dataset sizes to iterate over
MODELS=("allam_7b_instruct" "qwen3_8b" "llama_3_1_8b_instruct")
SIZES=("1" "16" "40")
BASE_DATA_DIR="/home/ubuntu/llm_effective_depth/analysis/data"

# Exit immediately if a command exits with a non-zero status
set -e

# Iterate over dataset sizes first (1 first for testing, then 16, then 40)
for SIZE in "${SIZES[@]}"; do
    echo "============================================================"
    echo "Starting processing for dataset size: MIXED${SIZE}"
    echo "============================================================"
    
    # Define Data Paths dynamically (these only change based on the SIZE)
    DATA_AR="${BASE_DATA_DIR}/MIXED${SIZE}_arabic_analysis_cases.json"
    DATA_EN="${BASE_DATA_DIR}/MIXED${SIZE}_english_analysis_cases.json"
    
    # Iterate over each model
    for MODEL in "${MODELS[@]}"; do
        echo "  -> Processing model: $MODEL for MIXED${SIZE}"
        
        # Define Output Directories dynamically
        OUT_LAYER_AR="/home/ubuntu/llm_effective_depth/output/${MODEL}/arabic/random${SIZE}PM/layer_effect/"
        OUT_LAYER_EN="/home/ubuntu/llm_effective_depth/output/${MODEL}/english/random${SIZE}PM/layer_effect/"
        OUT_KL_AR="/home/ubuntu/llm_effective_depth/output/${MODEL}/arabic/random${SIZE}PM/KL/"
        OUT_KL_EN="/home/ubuntu/llm_effective_depth/output/${MODEL}/english/random${SIZE}PM/KL/"

        # Ensure output directories exist
        mkdir -p "$OUT_LAYER_AR" "$OUT_LAYER_EN" "$OUT_KL_AR" "$OUT_KL_EN"

        # 1. analyze_future_effects.py - Arabic
        echo "    [1/4] Running analyze_future_effects.py (Arabic)..."
        python analyze_future_effects.py "$MODEL" \
            --parts layer \
            --no-completions \
            --no-accuracy \
            --language arabic \
            --data-path "$DATA_AR" \
            --output-dir "$OUT_LAYER_AR"
        
        # 2. analyze_future_effects.py - English
        echo "    [2/4] Running analyze_future_effects.py (English)..."
        python analyze_future_effects.py "$MODEL" \
            --parts layer \
            --no-completions \
            --no-accuracy \
            --language english \
            --data-path "$DATA_EN" \
            --output-dir "$OUT_LAYER_EN"
        
        # 3. logitlens.py - Arabic
        echo "    [3/4] Running logitlens.py (Arabic)..."
        python logitlens.py "$MODEL" \
            --language arabic \
            --data-path "$DATA_AR" \
            --output-dir "$OUT_KL_AR"
        
        # 4. logitlens.py - English
        echo "    [4/4] Running logitlens.py (English)..."
        python logitlens.py "$MODEL" \
            --language english \
            --data-path "$DATA_EN" \
            --output-dir "$OUT_KL_EN"
        
        echo "  <- Finished model: $MODEL for dataset size: MIXED${SIZE}"
        echo "------------------------------------------------------------"
    done
done

echo "All runs completed successfully!"