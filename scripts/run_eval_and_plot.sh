#!/bin/bash
source /u/home/kulp/miniconda3/bin/activate mlqec-env
export MGHD_SAMPLER=stim

CHECKPOINT="/u/home/kulp/data/results_validation_d3_5/20251120-175741_surface_d5_iqm_garnet_example/best.pt"
OUTPUT_JSON="data/results_validation_d3_5/evaluation_results.json"
PLOT_DIR="data/results_validation_d3_5/plots"

echo "Starting evaluation..."
python /u/home/kulp/MGHD/scripts/evaluate_model.py \
  --checkpoint "$CHECKPOINT" \
  --distances "3,5" \
  --p-values "0.001,0.003,0.005,0.008,0.01,0.012" \
  --shots 20000 \
  --batch-size 1024 \
  --output "$OUTPUT_JSON" \
  --node-feat-dim 9 \
  --cuda

echo "Evaluation done. Generating plots..."
python /u/home/kulp/MGHD/scripts/plot_validation.py "$OUTPUT_JSON" --output-dir "$PLOT_DIR"

echo "All done! Results in $PLOT_DIR"
