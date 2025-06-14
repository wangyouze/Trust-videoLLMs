# #!/bin/bash

# # Check if model_id parameter was provided
# if [ -z "$1" ]; then
#   echo "Usage: $0 <model_id>"
#   exit 1
# fi

# MODEL_ID=$1
# BASE_DIR="scripts/score/truthfulness"

# # Iterate over each subdirectory in BASE_DIR
# for DIR in "$BASE_DIR"/*; do
#     echo "Processing directory: $DIR"
#   if [ -d "$DIR" ]; then
#     echo "Processing directory: $DIR"
#     # Iterate over each Python script in the subdirectory
#     for PY_FILE in "$DIR"/*.py; do
#       if [ -f "$PY_FILE" ]; then
#         echo "Running $PY_FILE with model_id=$MODEL_ID"
#         python "$PY_FILE" --model_id "$MODEL_ID"
#       fi
#     done
#   fi
# done


#!/bin/bash

# Check if model_id parameter was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <model_id>"
  exit 1
fi

MODEL_ID=$1
BASE_DIR="scripts/score/fairness"

# Find and process all Python files in BASE_DIR and its subdirectories
find "$BASE_DIR" -type f -name "*.py" | while read -r PY_FILE; do
    echo "Running $PY_FILE with model_id=$MODEL_ID"
    python "$PY_FILE" --model_id "$MODEL_ID" || {
        echo "Error executing $PY_FILE"
        exit 1
    }
done